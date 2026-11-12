"""visualize_episode --review_replay: the collector's review replay over chosen episodes, simulated."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recording import record_episode_single as collector
from recording import recording_replay
from visualization import visualize_episode as viewer


class FakeBackend:
    def __init__(self, events):
        self.events = events
        self.events.append('build')

    def setup_for_recording(self):
        self.events.append('start position')

    def damping_target(self):
        return None  # damping_exit re-raises without a target

    def shutdown(self, *, home=True):
        self.events.append(('shutdown', home))


@pytest.fixture
def review(monkeypatch):
    events, loaded, played, keys = [], [], [], []
    state = SimpleNamespace(events=events, loaded=loaded, played=played, keys=keys,
                            results=[], gains=[], replay_error=None)

    def load(path, *, camera_names, robot_contract, fps):
        assert 'build' not in events  # every episode is checked before anything moves
        assert camera_names == ['video_left'] and fps == 100
        loaded.append(path)
        return np.zeros((2, 7)), 'video_left', np.zeros((4, 6, 3), np.uint8)

    def replay(backend, actions, camera, reference, *, label, gain_scales, poll_keys, live_frame):
        if state.replay_error:
            raise state.replay_error
        played.append((label, gain_scales))
        return state.results.pop(0) if state.results else 'completed'

    class Thread:
        def join(self, timeout=None):
            events.append('cameras joined')

    class Stop:
        def set(self):
            events.append('cameras stopped')

    monkeypatch.setattr(recording_replay, 'load_replay_episode', load)
    monkeypatch.setattr(recording_replay, 'replay_take', replay)
    monkeypatch.setattr(collector, 'build_backend', lambda ctx: FakeBackend(events))
    monkeypatch.setattr(collector, 'start_camera_capture',
                        lambda ctx, use_camera_server: events.append('cameras') or (Stop(), Thread()))
    monkeypatch.setattr(collector, 'wait_for_cameras', lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, 'read_key_blocking', lambda: keys.pop(0))
    monkeypatch.setattr(collector, 'prompt_replay_gain_scales',
                        lambda: state.gains.pop(0) if state.gains else (1.0, 1.0))
    return state


def args(*episodes):
    return {'robot': 'arx', 'task_name': 'arx_left_wrist_only', 'task_config': None, 'dataset_dir': None,
            'review_replay': list(episodes)}


def test_replays_chosen_episodes_in_order_with_repeat_and_cycle(review):
    review.keys[:] = ['\n', 'r', '\n', 'x']
    review.gains[:] = [(1.0, 1.0), (0.5, 0.2), (0.5, 0.2), (0.5, 0.2)]
    viewer.run_review_replay(args(24, 27))
    assert [path.rsplit('/', 1)[-1] for path in review.loaded] == ['episode_00024.zarr', 'episode_00027.zarr']
    assert review.played == [('episode 24', (1.0, 1.0)), ('episode 27', (0.5, 0.2)),
                             ('episode 27', (0.5, 0.2)), ('episode 24', (0.5, 0.2))]
    assert review.events[:3] == ['cameras', 'build', 'start position']
    assert review.events[-1] == ('shutdown', True)  # normal exit homes, then disables


def test_unloadable_episode_stops_before_cameras_or_motors(review, monkeypatch):
    def fail(path, **kwargs):
        raise ValueError('rate differs')
    monkeypatch.setattr(recording_replay, 'load_replay_episode', fail)
    with pytest.raises(ValueError, match='rate'):
        viewer.run_review_replay(args(24, 27))
    assert review.events == []


def test_quit_at_the_overlay_ends_the_review_and_homes(review):
    review.results[:] = ['quit']
    viewer.run_review_replay(args(24, 27))
    assert [label for label, _ in review.played] == ['episode 24']
    assert review.events[-1] == ('shutdown', True)


def test_cancelled_gain_prompt_skips_motion_but_keeps_reviewing(review):
    review.gains[:] = [None]
    review.keys[:] = ['\n', 'x']
    viewer.run_review_replay(args(24, 27))
    assert [label for label, _ in review.played] == ['episode 27']


def test_failure_reaches_the_failure_exit_and_skips_the_normal_home(review):
    review.replay_error = RuntimeError('stale feedback')
    with pytest.raises(RuntimeError, match='stale'):
        viewer.run_review_replay(args(24))
    assert review.events[-1] == ('shutdown', False)
    assert 'cameras stopped' in review.events


@pytest.mark.parametrize('extra', [['--action_replay'], ['--fill_in'], ['--arm_kp', *['1'] * 7]])
def test_cli_rejects_flags_owned_by_other_replays(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['visualize_episode.py', '--robot', 'arx', '--task_name',
                                      'arx_left_wrist_only', '--review_replay', '24', '27', *extra])
    with pytest.raises(SystemExit):
        viewer.parse_args()


def test_cli_parses_episode_list_without_episode_idx(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['visualize_episode.py', '--robot', 'arx', '--task_name',
                                      'arx_left_wrist_only', '--review_replay', '24', '27'])
    assert viewer.parse_args()['review_replay'] == [24, 27]
