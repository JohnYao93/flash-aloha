"""Task-independent async worker and configured-overlap chunk execution."""
import argparse
import threading
import time as time_v2
from queue import Queue, Empty, Full
from typing import Any
import numpy as np
from inference.policy_transport import WebsocketOpenPIClient, TimingStats, timing_now

class AsyncOpenPIQueryWorker:
    def __init__(
        self,
        *,
        policy_client: WebsocketOpenPIClient,
        query_actions,
        compress=None,
        timing_stats: TimingStats,
        compress_images: bool = False,
        jpeg_quality: int = 50,
        max_delay_s: float | None = None,
    ):
        self._policy_client = policy_client
        self._query_actions = query_actions
        self._compress = compress
        self._timing_stats = timing_stats
        self._compress_images = compress_images
        self._jpeg_quality = jpeg_quality
        if max_delay_s is not None and (not np.isfinite(max_delay_s) or max_delay_s <= 0):
            raise ValueError('Async delay budget must be finite and positive')
        self._max_delay_s = max_delay_s
        self._submitted_at = None
        self._requests: Queue[dict[str, Any]] = Queue(maxsize=1)
        self._results: Queue[dict[str, Any]] = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._in_flight = False
        self._result_available = False
        self._session_id = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="openpi-query-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def reset(self) -> None:
        drained_requests = self._drain(self._requests)
        with self._lock:
            self._drain(self._results)
            self._session_id += 1
            self._error = None
            self._submitted_at = None
            self._result_available = False
            if drained_requests > 0:
                self._in_flight = False

    def has_work_pending(self) -> bool:
        with self._lock:
            return self._in_flight or self._result_available

    def wait_until_idle(self, *, poll_interval_s: float = 0.01) -> None:
        while not self._stop_event.is_set():
            self.raise_if_failed()
            if not self.has_work_pending():
                return
            time_v2.sleep(poll_interval_s)

    def set_policy_client(self, policy_client: WebsocketOpenPIClient) -> None:
        """Switch servers only after the caller has reset and drained this worker."""
        if self.has_work_pending():
            raise RuntimeError("cannot switch OpenPI server while an async query is pending")
        with self._lock:
            self._policy_client = policy_client

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
            self._error = None
            if (error is None and self._submitted_at is not None and self._in_flight
                    and self._max_delay_s is not None
                    and timing_now() - self._submitted_at >= self._max_delay_s):
                error = InferenceDelayExceeded('Policy request exceeded configured overlap; increase overlap before resuming')
        if error is not None:
            if isinstance(error, InferenceDelayExceeded):
                raise error
            raise RuntimeError("Async OpenPI inference failed") from error

    def submit(
        self,
        *,
        image_dict: dict[str, np.ndarray],
        image_preprocessor=None,
        qpos: np.ndarray,
        prompt: str,
        query_index: int,
        submit_chunk_step: int,
        submit_rollout_index: int | None = None,
        rtc_metadata: dict[str, Any] | None = None,
        copy_inputs: bool = True,
        goal_generation: int = 0,
    ) -> bool:
        with self._lock:
            if self._in_flight or self._result_available:
                return False
            self._in_flight = True
            session_id = self._session_id
            submitted_at = self._submitted_at = timing_now()

        if copy_inputs:
            request_image_dict = {name: image.copy() for name, image in image_dict.items()}
        else:
            request_image_dict = image_dict
        request_qpos = np.asarray(qpos, dtype=np.float32).copy()

        request = {
            "submitted_at": submitted_at,
            "session_id": session_id,
            "image_dict": request_image_dict,
            "image_preprocessor": image_preprocessor,
            "qpos": request_qpos,
            "prompt": prompt,
            "query_index": query_index,
            "submit_chunk_step": int(submit_chunk_step),
            "submit_rollout_index": -1 if submit_rollout_index is None else int(submit_rollout_index),
            "rtc_metadata": rtc_metadata,
            "goal_generation": int(goal_generation),
        }
        try:
            self._requests.put_nowait(request)
            return True
        except Full:
            with self._lock:
                self._in_flight = False
            return False

    def poll_result(self) -> dict[str, Any] | None:
        with self._lock:
            try:
                result = self._results.get_nowait()
            except Empty:
                self._result_available = False
                return None
            self._result_available = False
        return result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.05)
            except Empty:
                continue

            try:
                image_dict = request["image_dict"]
                image_preprocessor = request.get("image_preprocessor")
                if image_preprocessor is not None:
                    prepare_start = timing_now()
                    image_dict = image_preprocessor(image_dict)
                    self._timing_stats.record(
                        "async_image_prepare",
                        timing_now() - prepare_start,
                    )
                if self._compress_images:
                    compress_start = timing_now()
                    image_dict = self._compress(
                        image_dict,
                        jpeg_quality=self._jpeg_quality,
                    )
                    self._timing_stats.record("async_jpeg_compress", timing_now() - compress_start)
                with self._lock:
                    active_session = request["session_id"] == self._session_id
                if not active_session:
                    print(f"[OpenPI Async] discarded stale query={request['query_index']} before send")
                    continue
                action_chunk, query_timing, raw_shape = self._query_actions(
                    self._policy_client,
                    image_dict,
                    request["qpos"],
                    request["prompt"],
                    query_index=request["query_index"],
                    rtc_metadata=request["rtc_metadata"],
                )
                for timing_name, timing_value in query_timing.items():
                    self._timing_stats.record(f"async_{timing_name}", timing_value)
                latency_s = timing_now() - request['submitted_at']
                if self._max_delay_s is not None and latency_s >= self._max_delay_s:
                    raise InferenceDelayExceeded(
                        f'Policy response took {latency_s * 1000:.1f} ms; overlap is '
                        f'{self._max_delay_s * 1000:.1f} ms. Increase overlap before resuming.')
                with self._lock:
                    active_session = request["session_id"] == self._session_id
                    if active_session:
                        self._drain(self._results)
                        self._results.put_nowait(
                            {
                                "latency_s": latency_s,
                                "query_index": request["query_index"],
                                "action_chunk": action_chunk,
                                "query_timing": query_timing,
                                "raw_shape": raw_shape,
                                "submit_chunk_step": request["submit_chunk_step"],
                                "submit_rollout_index": request["submit_rollout_index"],
                                "query_qpos": request["qpos"].copy(),
                                "goal_generation": request["goal_generation"],
                                "rtc_prefix_length": rtc_prefix_length(
                                    request["rtc_metadata"]
                                ),
                            }
                        )
                        self._result_available = True
                    else:
                        print(f"[OpenPI Async] discarded stale query={request['query_index']} after reset")
            except BaseException as exc:
                with self._lock:
                    if request["session_id"] == self._session_id:
                        self._error = exc
            finally:
                with self._lock:
                    self._in_flight = False

    @staticmethod
    def _drain(queue: Queue) -> int:
        count = 0
        while True:
            try:
                queue.get_nowait()
                count += 1
            except Empty:
                return count

def ensure_chunk_length(all_actions: np.ndarray, required_chunk_len: int) -> np.ndarray:
    if all_actions.shape[1] >= required_chunk_len:
        return all_actions
    last_action = all_actions[:, -1:]
    needed_actions = required_chunk_len - all_actions.shape[1]
    return np.concatenate([all_actions, np.repeat(last_action, needed_actions, axis=1)], axis=1)

def fit_chunk_length(all_actions: np.ndarray, required_chunk_len: int) -> np.ndarray:
    """Pad or trim a policy response to one explicit executable window."""
    return ensure_chunk_length(all_actions, required_chunk_len)[:, :required_chunk_len]

def is_sync_query_point(step: int, query_frequency: int) -> bool:
    return int(step) % int(query_frequency) == 0

def merge_openpi_chunk(
    *,
    new_action_chunk: np.ndarray,
    t: int,
    current_chunk_size: int,
    new_chunk_start_index: int = 0,
) -> tuple[np.ndarray, int, int]:
    next_chunk_start_index = min(
        max(0, int(new_chunk_start_index)),
        new_action_chunk.shape[1] - 1,
    )
    all_actions = fit_chunk_length(
        new_action_chunk,
        current_chunk_size + next_chunk_start_index,
    )
    return all_actions, next_chunk_start_index, t

def merge_sync_openpi_chunk(
    *,
    new_action_chunk: np.ndarray,
    current_chunk_size: int,
) -> tuple[np.ndarray, int]:
    """Replace a synchronous policy chunk and execute it from action zero."""
    return fit_chunk_length(new_action_chunk, current_chunk_size), 0

def select_action_from_openpi_chunk(
    *,
    all_actions: np.ndarray,
    elapsed_from_chunk_update: int,
    chunk_start_index: int,
) -> np.ndarray:
    chunk_length = all_actions.shape[1]
    action_ind = min(max(0, elapsed_from_chunk_update) + chunk_start_index, chunk_length - 1)
    return all_actions[:, action_ind].copy()

def rtc_prefix_length(rtc_metadata: dict[str, Any] | None) -> int:
    if not rtc_metadata:
        return 0
    if "action_prefix" in rtc_metadata:
        return int(rtc_metadata["action_prefix"]["prefix_length"])
    prefix = rtc_metadata.get("rtc/prev_chunk_left_over")
    return 0 if prefix is None else int(np.asarray(prefix).shape[0])

def get_unexecuted_chunk_suffix(
    all_actions: np.ndarray | None,
    *,
    current_chunk_step: int,
    current_chunk_start_index: int,
) -> np.ndarray | None:
    if all_actions is None:
        return None
    action_index = min(max(0, current_chunk_step + current_chunk_start_index), all_actions.shape[1])
    return all_actions[0, action_index:].copy()

def poll_async_result(
    async_worker: AsyncOpenPIQueryWorker,
    pending_async_result: dict[str, Any] | None,
    *,
    current_chunk_step: int,
    log_prefix: str,
    step: int | None = None,
    expected_goal_generation: int | None = None,
) -> dict[str, Any] | None:
    async_worker.raise_if_failed()
    result = async_worker.poll_result()
    if result is None:
        return pending_async_result
    if (
        expected_goal_generation is not None
        and int(result.get("goal_generation", 0)) != expected_goal_generation
    ):
        print(
            f"{log_prefix} discarded stale goal generation "
            f"query={result['query_index']} "
            f"result_generation={result.get('goal_generation', 0)} "
            f"active_generation={expected_goal_generation}"
        )
        return pending_async_result

    step_text = f" at step={step}" if step is not None else ""
    print(
        f"{log_prefix} received query={result['query_index']}{step_text} "
        f"elapsed={current_chunk_step - int(result['submit_chunk_step'])}"
    )
    result["measured_start_index"] = current_chunk_step - int(result["submit_chunk_step"])
    return result

def should_submit_async_query(
    async_worker: AsyncOpenPIQueryWorker | None,
    *,
    all_actions: np.ndarray | None,
    pending_async_result: dict[str, Any] | None,
    current_chunk_step: int,
    current_chunk_start_index: int,
    async_prefetch_steps: int,
    force_submit: bool = False,
) -> bool:
    if async_worker is None or pending_async_result is not None:
        return False
    if async_worker.has_work_pending():
        return False
    if force_submit:
        return True
    if all_actions is None:
        return False
    current_action_index = current_chunk_step + current_chunk_start_index
    remaining_actions = max(0, all_actions.shape[1] - current_action_index)
    return remaining_actions <= async_prefetch_steps


class InferenceDelayExceeded(RuntimeError):
    """The configured overlap cannot cover this policy response."""


def validate_overlap_budget(fps, overlap_steps, max_delay_ms=None):
    if (fps is None or not np.isfinite(fps) or fps <= 0 or overlap_steps is None
            or not np.isfinite(overlap_steps) or int(overlap_steps) != overlap_steps or overlap_steps <= 0):
        raise ValueError('FPS and configured overlap must be positive')
    if max_delay_ms is not None:
        if not np.isfinite(max_delay_ms) or max_delay_ms < 0:
            raise ValueError('Maximum inference delay must be finite and nonnegative')
        if overlap_steps / fps <= max_delay_ms / 1000:
            raise ValueError('Configured overlap must be greater than the maximum inference delay')


def maybe_merge_async_result(args, *, pending_async_result, all_actions, t,
        current_chunk_step, current_chunk_update_t, current_chunk_start_index,
        async_prefetch_steps, timing_stats=None, log_prefix='', include_query_total=False):
    """All policies resume at the configured overlap, never an observed delay.

    A latency budget overrun invalidates the response instead of silently
    executing a stale continuation or changing the configured overlap.
    """
    validate_overlap_budget(args.fps, async_prefetch_steps,
                            getattr(args, 'max_inference_delay_ms', None))
    result = pending_async_result
    unchanged = (result, all_actions, current_chunk_start_index,
                 current_chunk_update_t, current_chunk_step, False, 0, None)
    if result is None:
        return unchanged
    elapsed = current_chunk_step - int(result['submit_chunk_step'])
    if elapsed < async_prefetch_steps:
        return unchanged
    latency = result.get('latency_s', result.get('query_timing', {}).get('query_total'))
    if latency is not None and latency >= async_prefetch_steps / args.fps:
        raise InferenceDelayExceeded(
            f'Policy response took {latency * 1000:.1f} ms; configured overlap '
            f'is {async_prefetch_steps / args.fps * 1000:.1f} ms. Increase overlap before resuming.')
    chunk = result['action_chunk']
    if chunk.shape[1] < async_prefetch_steps + args.query_frequency:
        raise ValueError('Policy horizon must cover configured overlap plus the execution window')
    start = timing_now()
    window, offset, update = merge_openpi_chunk(new_action_chunk=chunk, t=t,
        current_chunk_size=args.query_frequency, new_chunk_start_index=async_prefetch_steps)
    if timing_stats is not None:
        timing_stats.record('async_chunk_update', timing_now() - start)
    measured = int(result.get('measured_start_index', elapsed))
    record = dict(query_index=int(result['query_index']), action_chunk=chunk,
        query_rollout_index=int(result.get('submit_rollout_index', -1)), merge_rollout_index=int(t),
        start_index=offset, submit_chunk_step=int(result['submit_chunk_step']),
        measured_start_index=measured, query_qpos=result.get('query_qpos'))
    print(f'{log_prefix} merged query={result["query_index"]} step={t} '
          f'configured_overlap={offset} observed_steps={measured}')
    return None, window[:, offset:offset + args.query_frequency], 0, update, 0, True, measured, record
