"""Robot collector tests must never contact live camera recording endpoints."""
import os
from pathlib import Path

import pytest

# Test-only tasks (ARX, one and two arms, with and without cameras) resolve by name
# alongside the shipped examples. Set before any config module builds its task catalog.
TASK_FIXTURES = Path(__file__).resolve().parent / 'task_fixtures'
os.environ['FLASH_ALOHA_TASK_DIRS'] = str(TASK_FIXTURES)


@pytest.fixture(autouse=True)
def restore_garbage_collection():
    # Captures latch collection off for the process; keep tests independent.
    yield
    from contracts import capture_gc
    capture_gc._reset_for_tests()


@pytest.fixture(autouse=True)
def cleanup_episode_fixtures():
    import shutil
    from episode_fixtures import directories
    yield
    while directories:
        shutil.rmtree(directories.pop())


@pytest.fixture(autouse=True)
def forbid_live_recording_endpoints(monkeypatch):
    from recording import distributed_recording
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must inject synthetic streams, never start a live recorder coordinator')
    monkeypatch.setattr(distributed_recording.DistributedRecorder, '__init__', forbidden)


@pytest.fixture(autouse=True)
def independent_camera_streams(request, monkeypatch, tmp_path):
    # Only the collectors migrated to the shared raw buffer need this fake.
    # Lifecycle tests construct their own explicit fake stream implementation.
    if request.path.name not in {
        'test_shared_episode_recording.py', 'test_record_episode_single.py', 'test_recording_disk_buffer.py',
        'test_standard_recording.py', 'test_camera_capture_clock.py',
    }:
        if not (request.path.name == 'test_camera_observation.py' and request.node.name in {
            'test_multibox_export_follows_saved_frame_ids',
            'test_legacy_cli_export_preserves_robot_samples_and_frame_references'}):
            return
    from recording import episode_buffer
    from contracts import standard_capture
    from recording import record_episode_single as single
    import numpy as np
    from fake_streams import FakeStreams
    FakeStreams.instances = []
    original_capture = standard_capture.capture_observation
    def capture(read_state, names, **kwargs):
        clock = kwargs.get('clock') or single.time.monotonic
        def snapshot(names, *, copy_images):
            now = clock()
            stream = next((s for s in reversed(FakeStreams.instances) if s.names == list(names)), None)
            references = {name: dict(frame_number=1001 + (len(stream.timing[name]) if stream else 0),
                capture_timestamp=now-.00101, consumer_received_timestamp=now-.000009) for name in names}
            images = {name: np.full((48, 64, 3), 13, np.uint8) for name in names} if copy_images else {}
            return images, dict(snapshot_timestamp=now, cameras=references)
        kwargs['clock'] = clock
        if kwargs.get('snapshot_provider') is None:
            kwargs['snapshot_provider'] = snapshot
        return original_capture(read_state, names, **kwargs)
    monkeypatch.setattr(standard_capture, 'capture_observation', capture)
    monkeypatch.setattr(episode_buffer, 'DistributedRecorder', FakeStreams)
    original = episode_buffer.EpisodeBuffer.__init__
    def initialize(self, **kwargs):
        if kwargs.get('directory') is None:
            kwargs['directory'] = tmp_path
        original(self, **kwargs)
    monkeypatch.setattr(episode_buffer.EpisodeBuffer, '__init__', initialize)


@pytest.fixture(autouse=True)
def forbid_live_hardware(monkeypatch):
    """Runtime imports are allowed; actual SDK construction is not."""
    import sys
    from types import ModuleType
    sdk = ModuleType('arx5_interface')
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must inject simulated hardware')
    sdk.Arx5JointController = sdk.Arx5CartesianController = forbidden
    from types import SimpleNamespace
    import numpy as np
    sdk.JointState = lambda n: SimpleNamespace(pos=np.zeros(n), vel=np.zeros(n), torque=np.zeros(n), gripper_pos=0.)
    monkeypatch.setitem(sys.modules, 'arx5_interface', sdk)


@pytest.fixture(autouse=True)
def isolated_run_logs(monkeypatch, tmp_path):
    """Inference and replay run logs go to the test's temporary directory, never logs/."""
    import config.paths
    monkeypatch.setattr(config.paths, 'INFERENCE_LOG_DIR', tmp_path / 'logs' / 'inference')
    monkeypatch.setattr(config.paths, 'REPLAY_LOG_DIR', tmp_path / 'logs' / 'replay')
