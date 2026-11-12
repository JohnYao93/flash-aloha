from __future__ import annotations

import sys
import json
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
import queue

import numpy as np
import zarr
from data_processing import episode_store
import cv2


SRC_ACT_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ACT_DIR))

from camera_utils.camera_server import CameraRecordingWriter  # noqa: E402
from recording.distributed_recording import (
    ControlWriter,
    publish_recording,
    camera_video_encoding_from_statuses,
    CameraBarrierClient,
    DistributedRecorder,
    RecorderError,
)
from recording.recording_utils import atomic_write_json
from recording import distributed_recording  # noqa: E402


class FakeRMQServer:
    def __init__(self):
        self.status = []

    def pop_data(self, topic, n):
        del topic, n
        return [], []

    def put_data(self, topic, data):
        del topic
        self.status.append(data)


class FakeProcess:
    def poll(self):
        return None


class FakeCameraClient:
    def __init__(self, client_name, server_endpoint, acknowledge=True):
        self.client_name = client_name
        self.server_endpoint = server_endpoint
        self.acknowledge = acknowledge
        self.info = []

    def get_topic_status(self, topic, timeout_s):
        del topic, timeout_s
        return len(self.info)

    def put_data(self, topic, data, timeout_s=1.0):
        del topic, timeout_s
        if not self.acknowledge:
            return
        from robotmq import deserialize, serialize

        command = deserialize(data)
        state = {"start": "ready", "pause": "paused", "resume": "ready", "stop": "stopped"}[
            command["type"]
        ]
        self.info.append(
            serialize(
                {
                    "state": state,
                    "session_id": command["session_id"],
                    "command_id": command["command_id"],
                    "fatal_error": None,
                }
            )
        )

    def pop_data(self, topic, n, timeout_s=1.0):
        del topic, n, timeout_s
        result, self.info = self.info, []
        return result, []


class FakeVideoLogger:
    def __init__(self, first_write_gate=None, fail_start_attempts=0):
        self.attr = {
            "camera_configs": {
                "main": {"width": 2, "height": 2, "fps": 50, "type": "rgb"}
            }
        }
        self.rmq_server = FakeRMQServer()
        self.ffmpeg_processes = {}
        self.zarr_group = None
        self._is_recording = False
        self.start_count = 0
        self.stop_count = 0
        self.first_write_gate = first_write_gate
        self.fail_start_attempts = int(fail_start_attempts)
        self.writes_this_start = 0
        self.write_started = threading.Event()

    def start_recording(self, episode_dir):
        self.start_count += 1
        self.writes_this_start = 0
        self._is_recording = True
        self.zarr_group = zarr.open_group(str(Path(episode_dir) / "video_test.zarr"), mode="w")
        self.zarr_group.create_dataset(
            "main_timestamps", shape=(0,), chunks=(100,), dtype=np.float32
        )
        self.ffmpeg_processes = {"main": FakeProcess()}

    def log_frame(self, *, camera_name, timestamp, frame):
        del camera_name, frame
        self.write_started.set()
        if self.first_write_gate is not None:
            gate = self.first_write_gate
            self.first_write_gate = None
            gate.wait(timeout=2.0)
        timestamps = self.zarr_group["main_timestamps"]
        timestamps.resize((timestamps.shape[0] + 1,))
        timestamps[-1] = timestamp
        self.writes_this_start += 1
        if self.start_count <= self.fail_start_attempts:
            self.ffmpeg_processes = {}
            raise RuntimeError("[video_test] FFmpeg process for 'main' closed unexpectedly")

    def stop_recording(self):
        self.stop_count += 1
        self._is_recording = False
        self.ffmpeg_processes = {}
        self.zarr_group = None


class FakeBufferedVideoLogger(FakeVideoLogger):
    """Matches robologger versions that create timestamps only on stop."""

    def __init__(self):
        super().__init__()
        self.data_lists = {}
        self.saved_timestamps = None

    def start_recording(self, episode_dir):
        self.start_count += 1
        self.writes_this_start = 0
        self._is_recording = True
        self.zarr_group = zarr.open_group(str(Path(episode_dir) / "video_test.zarr"), mode="w")
        self.data_lists["main_timestamps"] = []
        self.ffmpeg_processes = {"main": FakeProcess()}

    def log_frame(self, *, camera_name, timestamp, frame):
        del camera_name, frame
        self.write_started.set()
        self.data_lists["main_timestamps"].append(timestamp)
        self.writes_this_start += 1

    def stop_recording(self):
        self.saved_timestamps = np.asarray(
            self.data_lists["main_timestamps"], dtype=np.float64
        )
        self.zarr_group.create_dataset(
            "main_timestamps",
            data=self.saved_timestamps,
            shape=self.saved_timestamps.shape,
            chunks=(max(1, min(1000, len(self.saved_timestamps))),),
            dtype=np.float64,
            overwrite=True,
        )
        self.data_lists = {}
        super().stop_recording()


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class CameraRecordingWriterTest(unittest.TestCase):
    def test_buffered_timestamp_logger_starts_and_validates_before_flush(self):
        logger = FakeBufferedVideoLogger()
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraRecordingWriter(
                logger,
                "test",
                50,
                queue_size=16,
                warmup_frames=1,
                encoder_settle_seconds=0,
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "buffered-s1",
                    "command_id": "buffered-start-1",
                    "episode_dir": directory,
                }
            )
            wait_for(lambda: writer.accepting)
            writer.submit_frame(
                np.zeros((2, 2, 3), dtype=np.uint8),
                12.5,
                0,
                time.monotonic(),
            )
            wait_for(lambda: writer.state == "ready")
            self.assertNotIn("main_timestamps", logger.zarr_group)
            writer.submit_command(
                {
                    "type": "stop",
                    "session_id": "buffered-s1",
                    "command_id": "buffered-stop-1",
                }
            )
            wait_for(lambda: writer.state == "stopped")
            np.testing.assert_array_equal(logger.saved_timestamps, np.asarray([12.5]))
            self.assertIsNone(writer.fatal_error)
            writer.close()

    def test_blocked_cold_write_does_not_block_capture_and_delays_ready(self):
        gate = threading.Event()
        logger = FakeVideoLogger(first_write_gate=gate)
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraRecordingWriter(
                logger,
                "test",
                50,
                queue_size=16,
                warmup_frames=2,
                encoder_settle_seconds=0,
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "s1",
                    "command_id": "start-1",
                    "episode_dir": directory,
                }
            )
            wait_for(lambda: writer.accepting)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.submit_frame(frame, 1.0, 0, time.monotonic())
            self.assertTrue(logger.write_started.wait(timeout=1.0))
            started = time.monotonic()
            for index in range(1, 8):
                writer.submit_frame(frame, 1.0 + index / 50.0, index, time.monotonic())
            self.assertLess(time.monotonic() - started, 0.05)
            self.assertEqual(writer.state, "warming")
            gate.set()
            wait_for(lambda: writer.state == "ready")
            self.assertEqual(np.dtype(logger.zarr_group["main_timestamps"].dtype), np.dtype(np.float64))
            writer.submit_command(
                {"type": "stop", "session_id": "s1", "command_id": "stop-1"}
            )
            wait_for(lambda: writer.state == "stopped")
            writer.close()


class CameraBarrierClientTest(unittest.TestCase):
    def test_multi_camera_barrier_matches_session_and_command_ids(self):
        clients = []

        def factory(**kwargs):
            client = FakeCameraClient(**kwargs)
            clients.append(client)
            return client

        barrier = CameraBarrierClient(
            ["video_left", "video_overhead"], 55551, client_factory=factory
        )
        acknowledgements = barrier.command(
            "start",
            session_id="session-a",
            command_id="command-a",
            episode_dir="/tmp/episode-a",
            timeout=0.1,
        )
        self.assertEqual(set(acknowledgements), {"video_left", "video_overhead"})
        self.assertTrue(
            all(status["command_id"] == "command-a" for status in acknowledgements.values())
        )

    def test_barrier_timeout_is_fatal(self):
        barrier = CameraBarrierClient(
            ["video_left"],
            55551,
            client_factory=lambda **kwargs: FakeCameraClient(
                **kwargs, acknowledge=False
            ),
        )
        with self.assertRaises(RecorderError):
            barrier.command(
                "pause", session_id="session-a", command_id="command-b", timeout=0.02
            )


class DistributedRecorderFacadeTest(unittest.TestCase):
    def test_pause_ack_reports_every_control_committed_to_zarr(self):
        class FakeBarrier:
            def __init__(self, *_args, **_kwargs):
                pass

            def wait_until_connected(self, _timeout):
                pass

            def command(self, *_args, **_kwargs):
                if _args and _args[0] == "start":
                    return {
                        "video_left": {
                            "state": "ready",
                            "video_encoder": {
                                "configured_codec": "h264_nvenc",
                                "effective_codec": "libx264",
                                "backend": "software",
                                "fallback_used": True,
                            },
                        }
                    }
                return {}

            def assert_healthy(self):
                pass

        def wait_for_ack(events, command_id):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                event = events.get(timeout=0.1)
                if event.get("command_id") == command_id:
                    return event
            raise AssertionError(f"missing recorder ack {command_id}")

        with tempfile.TemporaryDirectory() as directory:
            controls = queue.Queue()
            commands = queue.Queue()
            events = queue.Queue()
            config = {
                "camera_names": [],
                "camera_logger_endpoint_base": 55551,
                "status_timeout": 0.5,
                "warmup_timeout": 0.5,
                "warmup_frames": 1,
            }
            with mock.patch.object(
                distributed_recording, "CameraBarrierClient", FakeBarrier
            ):
                coordinator = threading.Thread(
                    target=distributed_recording._coordinator_main,
                    args=(config, controls, commands, events),
                    daemon=True,
                )
                coordinator.start()
                request = {
                    "session_id": "committed-counter",
                    "data_dir": directory,
                    "camera_names": [],
                    "record_rollout": False,
                    "record_dagger": True,
                    "rollout_dir": directory,
                    "rollout_task_name": "test",
                    "dagger_dir": directory,
                    "dagger_task_name": "dagger_test",
                    "dagger_camera_names": [],
                    "is_double_arm": True,
                    "fps": 100.0,
                }
                commands.put(
                    {"type": "start", "command_id": "start", "request": request}
                )
                wait_for_ack(events, "start")
                with (
                    Path(directory) / "committed-counter" / "manifest.json"
                ).open(encoding="utf-8") as stream:
                    started_manifest = json.load(stream)
                self.assertEqual(
                    started_manifest["camera_video_encoding"]["video_left"][
                        "effective_codec"
                    ],
                    "libx264",
                )
                for index in range(3):
                    controls.put(
                        {
                            "timestamp": index / 100.0,
                            "qpos": np.asarray([index]),
                            "qvel": np.asarray([index]),
                            "commanded_action": np.asarray([index]),
                        }
                    )
                commands.put(
                    {
                        "type": "pause",
                        "command_id": "pause",
                        "expected_sample_count": 3,
                    }
                )
                pause_ack = wait_for_ack(events, "pause")
                self.assertEqual(pause_ack["sample_count"], 3)
                control = zarr.open_group(
                    str(Path(directory) / "committed-counter" / "control.zarr"),
                    mode="r",
                )
                self.assertEqual(control["commanded_action"].shape[0], 3)
                commands.put({"type": "shutdown"})
                coordinator.join(timeout=2.0)
                self.assertFalse(coordinator.is_alive())

    def test_committed_count_only_advances_from_coordinator_events(self):
        recorder = object.__new__(DistributedRecorder)
        recorder.event_queue = queue.Queue()
        recorder.fatal_error = None
        recorder.events = {}
        recorder.manifests = []
        recorder.enqueued_sample_count = 7
        recorder.committed_sample_count = 3

        recorder.event_queue.put(
            {"type": "health", "state": "recording", "sample_count": 5}
        )
        recorder._poll_events()

        self.assertEqual(recorder.enqueued_sample_count, 7)
        self.assertEqual(recorder.committed_sample_count, 5)

    def test_pause_and_stop_send_exact_sample_commit_barriers(self):
        recorder = object.__new__(DistributedRecorder)
        recorder.enqueued_sample_count = 11
        recorder._send_and_wait = mock.Mock(return_value={"state": "paused"})
        recorder.command_queue = mock.Mock()

        recorder.pause(2.0)
        pause_command = recorder._send_and_wait.call_args.args[0]
        self.assertEqual(pause_command["expected_sample_count"], 11)

        recorder.begin_stop_episode(
            retained=True,
            outputs={},
            timeout=2.0,
        )
        stop_command = recorder.command_queue.put.call_args.args[0]
        self.assertEqual(stop_command["expected_sample_count"], 11)

    def test_control_queue_overflow_is_fatal(self):
        recorder = DistributedRecorder.__new__(DistributedRecorder)
        recorder.control_queue = queue.Queue(maxsize=1)
        recorder.command_queue = queue.Queue(maxsize=1)
        recorder.event_queue = queue.Queue()
        recorder.events = {}
        recorder.fatal_error = None
        recorder.continuity_id = 0
        recorder.segment_id = 0
        recorder.metadata_id = 0
        recorder.enqueued_sample_count = 0
        recorder.committed_sample_count = 0
        recorder.process = mock.Mock()
        recorder.process.is_alive.return_value = True
        recorder.append_sample(
            {"timestamp": 0.0, "qpos": [0], "qvel": [0], "commanded_action": [0]}
        )
        with self.assertRaises(RecorderError):
            recorder.append_sample(
                {"timestamp": 0.1, "qpos": [0], "qvel": [0], "commanded_action": [0]}
            )

    def test_pause_resume_keeps_ffmpeg_and_resume_waits_for_written_frame(self):
        logger = FakeVideoLogger()
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraRecordingWriter(
                logger, "test", 50, warmup_frames=1, encoder_settle_seconds=0
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "s1",
                    "command_id": "start-1",
                    "episode_dir": directory,
                }
            )
            wait_for(lambda: writer.accepting)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.submit_frame(frame, 1.0, 0, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            processes = logger.ffmpeg_processes
            writer.submit_command(
                {"type": "pause", "session_id": "s1", "command_id": "pause-1"}
            )
            wait_for(lambda: writer.state == "paused")
            self.assertIs(logger.ffmpeg_processes, processes)
            writer.submit_command(
                {"type": "resume", "session_id": "s1", "command_id": "resume-1"}
            )
            self.assertEqual(writer.state, "resuming")
            writer.submit_frame(frame, 2.0, 1, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            self.assertEqual(logger.start_count, 1)
            writer.submit_command(
                {"type": "stop", "session_id": "s1", "command_id": "stop-1"}
            )
            wait_for(lambda: writer.state == "stopped")
            writer.close()

    def test_new_session_start_closes_stale_paused_recording(self):
        logger = FakeVideoLogger()
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            writer = CameraRecordingWriter(
                logger, "test", 50, warmup_frames=1, encoder_settle_seconds=0
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "stale-session",
                    "command_id": "start-1",
                    "episode_dir": str(first_dir),
                }
            )
            wait_for(lambda: writer.accepting)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.submit_frame(frame, 1.0, 0, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            writer.submit_command(
                {
                    "type": "pause",
                    "session_id": "stale-session",
                    "command_id": "pause-1",
                }
            )
            wait_for(lambda: writer.state == "paused")

            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "new-session",
                    "command_id": "start-2",
                    "episode_dir": str(second_dir),
                }
            )
            wait_for(lambda: writer.accepting)
            self.assertEqual(writer.state, "warming")
            self.assertEqual(logger.stop_count, 1)
            writer.submit_frame(frame, 2.0, 1, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            self.assertEqual(logger.start_count, 2)
            self.assertEqual(writer.status()["session_id"], "new-session")
            self.assertIsNone(writer.fatal_error)

            writer.submit_command(
                {"type": "stop", "session_id": "new-session", "command_id": "stop-2"}
            )
            wait_for(lambda: writer.state == "stopped")
            writer.close()

    def test_cold_ffmpeg_failure_retries_before_ready(self):
        logger = FakeVideoLogger(fail_start_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraRecordingWriter(
                logger,
                "test",
                50,
                warmup_frames=1,
                encoder_settle_seconds=0,
                startup_retry_limit=1,
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "s1",
                    "command_id": "start-1",
                    "episode_dir": directory,
                }
            )
            wait_for(lambda: writer.accepting)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.submit_frame(frame, 1.0, 0, time.monotonic())
            wait_for(lambda: logger.start_count == 2 and writer.accepting)
            self.assertNotEqual(writer.state, "ready")
            writer.submit_frame(frame, 2.0, 1, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            self.assertIsNone(writer.fatal_error)
            self.assertEqual(logger.stop_count, 1)

            writer.submit_command(
                {"type": "stop", "session_id": "s1", "command_id": "stop-1"}
            )
            wait_for(lambda: writer.state == "stopped")
            writer.close()

    def test_nvenc_oom_cold_start_retries_with_cpu_encoder(self):
        logger = FakeVideoLogger(fail_start_attempts=1)
        logger.codec = "h264_nvenc"
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraRecordingWriter(
                logger,
                "test",
                50,
                warmup_frames=1,
                encoder_settle_seconds=0,
                startup_retry_limit=1,
            )
            writer.start()
            writer.submit_command(
                {
                    "type": "start",
                    "session_id": "nvenc-oom",
                    "command_id": "nvenc-oom-start",
                    "episode_dir": directory,
                }
            )
            wait_for(lambda: writer.accepting)
            Path(directory, "video_test_ffmpeg_attempt1.log").write_text(
                "[h264_nvenc] InitializeEncoder failed: out of memory (10)\n"
            )
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.submit_frame(frame, 1.0, 0, time.monotonic())
            wait_for(lambda: logger.start_count == 2 and writer.accepting)

            self.assertEqual(logger.codec, "libx264")
            self.assertNotEqual(writer.state, "ready")
            writer.submit_frame(frame, 2.0, 1, time.monotonic())
            wait_for(lambda: writer.state == "ready")
            self.assertIsNone(writer.fatal_error)
            encoder = writer.status()["video_encoder"]
            self.assertEqual(encoder["configured_codec"], "h264_nvenc")
            self.assertEqual(encoder["effective_codec"], "libx264")
            self.assertEqual(encoder["backend"], "software")
            self.assertTrue(encoder["fallback_used"])
            self.assertEqual(
                encoder["fallback_reason"], "nvenc_resource_failure"
            )
            self.assertEqual(
                logger.zarr_group.attrs["video_encoder_effective_codec"],
                "libx264",
            )
            self.assertTrue(
                logger.zarr_group.attrs["video_encoder_fallback_used"]
            )

            writer.submit_command(
                {
                    "type": "stop",
                    "session_id": "nvenc-oom",
                    "command_id": "nvenc-oom-stop",
                }
            )
            wait_for(lambda: writer.state == "stopped")
            writer.close()


class RawControlTest(unittest.TestCase):
    def test_camera_video_encoding_from_statuses(self):
        statuses = {
            "left": {
                "state": "ready",
                "video_encoder": {
                    "configured_codec": "h264_nvenc",
                    "effective_codec": "libx264",
                    "fallback_used": True,
                },
            },
            "overhead": {"state": "ready"},
        }

        self.assertEqual(
            camera_video_encoding_from_statuses(statuses),
            {"left": statuses["left"]["video_encoder"]},
        )





    def test_float64_timestamps_and_numeric_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ControlWriter(Path(directory) / "control.zarr")
            writer.append(
                {
                    "timestamp": 1234.123456789,
                    "qpos": np.arange(7),
                    "qvel": np.arange(7) + 1,
                    "commanded_action": np.arange(7) + 2,
                    "continuity_id": 3,
                    "segment_id": 4,
                    "intervention_active": True,
                }
            )
            self.assertEqual(np.dtype(writer.group["timestamps"].dtype), np.dtype(np.float64))
            self.assertAlmostEqual(float(writer.group["timestamps"][0]), 1234.123456789)
            self.assertEqual(int(writer.group["continuity_id"][0]), 3)
            self.assertEqual(int(writer.group["segment_id"][0]), 4)






if __name__ == "__main__":
    unittest.main()
