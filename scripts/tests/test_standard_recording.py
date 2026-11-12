"""Shared raw storage adapters, using synthetic cameras only."""
import numpy as np
import pytest

from contracts import standard_capture
from contracts import standard_recording as recording


def test_rollout_keeps_only_references_until_export(tmp_path):
    names = ['video_left']
    lists = dict(images={}, position=[], velocity=[], commanded_action=[],
                 action_timestamps=[], image_timestamps={'video_left': []})
    with pytest.raises(RuntimeError, match='cancel'):
        @recording.recording_scope
        def collect():
            recording.ensure_rollout_started(lists, tmp_path, names, 100, False)
            for i in range(3):
                observation = standard_capture.capture_observation(
                    lambda: (np.full(7, i), np.full(7, -i)), names, clock=lambda: 100.+i*.01)
                recording.append_rollout(lists, observation, np.zeros((1, 7)))
            assert lists['_episode_buffer'].readers == {}  # No decode/encoding on append.
            assert isinstance(lists['images']['video_left'], recording.RecordedImages)
            raise RuntimeError('cancel')
        collect()
    assert lists['_episode_buffer'].finished
    np.testing.assert_array_equal(np.asarray(lists['position'])[:, 0], [0, 1, 2])


def test_rollout_export_follows_saved_references(tmp_path):
    lists = dict(images={}, position=[], velocity=[], commanded_action=[],
                 action_timestamps=[], image_timestamps={'video_left': []})
    recording.ensure_rollout_started(lists, tmp_path, ['video_left'], 100, False)
    observation = standard_capture.capture_observation(
        lambda: (np.zeros(7), np.zeros(7)), ['video_left'], clock=lambda: 100.)
    recording.append_rollout(lists, observation, np.ones(7))
    buffer = recording.finish_rollout(lists)
    assert buffer.finished
    assert len(lists['images']['video_left']) == 1
    np.testing.assert_allclose(lists['image_timestamps']['video_left'], [99.99899])
    assert lists['images']['video_left'][0].shape == (48, 64, 3)
