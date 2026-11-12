"""The camera-less replay task, with simulated hardware and synthetic streams only."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from recording import record_episode_single as recorder, recording_replay


def arguments(*flags):
    return recorder.build_arg_parser().parse_args(['--task_name', 'arx_right_no_camera', *flags])


@pytest.fixture
def no_window(monkeypatch):
    import cv2
    def forbidden(*args, **kwargs):
        raise AssertionError('A camera-less replay must not open or poll a window')
    for name in ('imshow', 'waitKey', 'destroyWindow'):
        monkeypatch.setattr(cv2, name, forbidden)


def test_task_is_one_arm_with_no_cameras():
    ctx = recorder.resolve_run_context(arguments())
    assert ctx.camera_names == []
    assert ctx.robot_type == 'arx' and ctx.fps == 100


@pytest.mark.parametrize('key, result', [('h', 'cancelled'), ('q', 'quit')])
def test_camera_less_go_ahead_still_cancels_or_quits(key, result, no_window):
    backend = SimpleNamespace(ctx=SimpleNamespace(fps=100.0), move_to_pose=Mock(), replay_rows=Mock())
    keys = [key]
    assert recording_replay.replay_take(
        backend, np.zeros((3, 7)), None, None, poll_keys=lambda: [keys.pop(0)] if keys else [],
        live_frame=Mock(), notify=lambda *_: None, sleep=lambda _: None) == result
    backend.replay_rows.assert_not_called()


def test_unsaved_take_source_has_no_overlay_without_cameras(monkeypatch):
    monkeypatch.setattr(recording_replay, 'finish_take', lambda samples: None)
    take = [SimpleNamespace(qpos=np.full(7, float(i)), images={}) for i in range(3)]
    actions, camera, reference = recording_replay.take_replay_source(take, [])
    assert actions.shape == (3, 7) and camera is None and reference is None
