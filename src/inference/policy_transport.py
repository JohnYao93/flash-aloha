"""Shared policy transport and timing; no hardware or task orchestration."""
from collections import defaultdict
import functools
import threading
import time as time_v2
from typing import Any, Callable
import msgpack
import numpy as np
import websockets.sync.client

COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_BOLD = "\033[1m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

MsgpackNumpyPacker = functools.partial(msgpack.Packer, default=pack_array)
msgpack_numpy_unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

class TimingStats:
    def __init__(
        self,
        *,
        enabled: bool,
        interval: int,
        reporter: Callable[[dict[str, object]], None] | None = None,
    ):
        self.enabled = enabled
        self.interval = max(1, interval)
        self.reporter = reporter
        self._samples = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, name: str, duration_s: float) -> None:
        if self.enabled:
            with self._lock:
                self._samples[name].append(duration_s * 1000.0)

    def report_if_due(self, *, step: int, query_count: int, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and (step + 1) % self.interval != 0:
            return
        with self._lock:
            if not self._samples:
                return
            samples = {name: values[:] for name, values in self._samples.items()}
            self._samples.clear()

        rows = []
        for name in sorted(samples):
            values = np.asarray(samples[name], dtype=np.float32)
            rows.append(
                {
                    "name": name,
                    "avg": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "count": int(values.size),
                }
            )
        rows.sort(key=lambda row: row["max"], reverse=True)

        print(
            f"\n{COLOR_BOLD}{COLOR_CYAN}[Timing]{COLOR_RESET} "
            f"step={COLOR_BOLD}{step}{COLOR_RESET} "
            f"queries={COLOR_BOLD}{query_count}{COLOR_RESET} "
            f"{COLOR_DIM}(avg/max over last {self.interval} control steps){COLOR_RESET}"
        )
        print(f"{COLOR_DIM}{'-' * 72}{COLOR_RESET}")
        for index, row in enumerate(rows):
            if index == 0:
                color = COLOR_RED
            elif index < 3:
                color = COLOR_YELLOW
            else:
                color = COLOR_GREEN
            print(
                f"  {color}{row['name']:<22}{COLOR_RESET} "
                f"avg={row['avg']:>7.1f} ms  "
                f"max={COLOR_BOLD}{row['max']:>7.1f}{COLOR_RESET} ms  "
                f"{COLOR_DIM}n={row['count']}{COLOR_RESET}"
            )
        print(f"{COLOR_DIM}{'-' * 72}{COLOR_RESET}\n")
        if self.reporter is not None:
            report = {
                "step": int(step),
                "query_count": int(query_count),
                "interval": self.interval,
                "rows": [
                    {
                        "name": str(row["name"]),
                        "avg_ms": float(row["avg"]),
                        "max_ms": float(row["max"]),
                        "count": int(row["count"]),
                    }
                    for row in rows
                ],
            }
            try:
                self.reporter(report)
            except Exception as exc:
                print(f"[Timing] browser diagnostics publish skipped: {exc}")

def timing_now() -> float:
    return time_v2.perf_counter()

def wait_until_loop_deadline(loop_start: float, execution_dt: float) -> None:
    remaining = execution_dt - (timing_now() - loop_start)
    if remaining > 0:
        time_v2.sleep(remaining)

class WebsocketOpenPIClient:
    def __init__(self, host: str, port: int | None = None, api_key: str | None = None):
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._packer = MsgpackNumpyPacker()
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> dict:
        return self._server_metadata

    def _connect_once(self):
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        try:
            return websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
            )
        except TypeError:
            return websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                extra_headers=headers,
            )

    def _wait_for_server(self):
        print(f"[OpenPI] Waiting for policy server at {self._uri}...")
        while True:
            try:
                conn = self._connect_once()
                metadata = msgpack_numpy_unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("[OpenPI] Server not ready; retrying in 5s...")
                time_v2.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        response, _ = self.infer_with_timing(obs)
        return response

    def infer_with_timing(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        pack_start = timing_now()
        packed_obs = self._packer.pack(obs)
        pack_duration = timing_now() - pack_start

        send_start = timing_now()
        self._ws.send(packed_obs)
        send_duration = timing_now() - send_start

        recv_start = timing_now()
        response = self._ws.recv()
        recv_duration = timing_now() - recv_start
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")

        unpack_start = timing_now()
        unpacked_response = msgpack_numpy_unpackb(response)
        unpack_duration = timing_now() - unpack_start
        return (
            unpacked_response,
            {
                "request_pack": pack_duration,
                "ws_send": send_duration,
                "server_wait_recv": recv_duration,
                "response_unpack": unpack_duration,
            },
        )
