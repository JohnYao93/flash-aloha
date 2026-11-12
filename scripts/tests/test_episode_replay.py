"""Replay of a saved episode from the collector, with a simulated backend and cameras."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recording import recording_replay
from recording.recording_replay import blend_reference, latest_saved_episode, overlay_camera



def test_overlay_uses_overhead_else_the_single_wrist_camera():
    assert overlay_camera(['video_left']) == 'video_left'
    assert overlay_camera(['video_right']) == 'video_right'
    assert overlay_camera(['video_left', 'video_overhead']) == 'video_overhead'
    assert overlay_camera(['video_left', 'video_overhead', 'video_right']) == 'video_overhead'
    with pytest.raises(ValueError):
        overlay_camera([])


def test_blend_resizes_the_reference_and_mixes_with_transparency():
    live = np.zeros((4, 6, 3), np.uint8)
    reference = np.full((8, 12, 3), 200, np.uint8)
    blended = blend_reference(live, reference, alpha=0.5)
    assert blended.shape == live.shape
    assert np.all(blended == 100)


def test_overlay_window_is_kept_above_once_mapped(monkeypatch):
    import cv2
    from utils import window_utils
    attempts = []
    monkeypatch.setattr(window_utils, 'keep_window_above',
                        lambda title: attempts.append(title) or len(attempts) == 2)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    backend, queue = FakeBackend(), [None, None, None, None, '1']
    reference = np.zeros((4, 6, 3), np.uint8)
    recording_replay.replay_take(
        backend, np.zeros((3, 7)), 'video_left', reference,
        poll_keys=lambda: [queue.pop(0)] if queue else [], live_frame=lambda name: reference,
        notify=lambda *_: None, pump=lambda: [], close_window=lambda: None, sleep=lambda _: None)
    assert attempts == ['Replay alignment'] * 2  # retried until mapped, then not again


def test_keep_above_is_a_no_op_without_a_display(monkeypatch):
    from utils.window_utils import keep_window_above
    monkeypatch.delenv('DISPLAY', raising=False)
    assert keep_window_above('Replay alignment') is False


def test_latest_saved_episode_is_the_highest_index(tmp_path):
    assert latest_saved_episode(tmp_path) is None
    for index in (2, 10, 3):
        (tmp_path / f'episode_{index:05d}.zarr').mkdir()
    (tmp_path / 'episode_00099.zarr.tmp').mkdir()
    assert latest_saved_episode(tmp_path) == str(tmp_path / 'episode_00010.zarr')



@pytest.fixture(autouse=True)
def default_gain_prompt(request, monkeypatch):
    """Collector tests answer the replay gain prompt with Enter (full gains)."""
    if 'gain_prompt' not in request.node.name:
        from recording import record_episode_single as recorder
        monkeypatch.setattr(recorder, 'prompt_replay_gain_scales', lambda: (1.0, 1.0))


class FakeBackend:
    def __init__(self):
        self.ctx = SimpleNamespace(robot_contract={}, fps=100.0)
        self.events = []
        self.replay_result = True
        self.replay_error = None
        self.move_error = None

    def move_to_pose(self, qpos):
        self.events.append(('move', np.asarray(qpos).copy()))
        if self.move_error:
            raise self.move_error

    def replay_rows(self, actions, *, fps, should_stop, kp_scale=1.0, kd_scale=1.0, log=None, **recording):
        self.events.append(('replay', len(actions), fps))
        self.gain_scales = (kp_scale, kd_scale)
        if self.replay_error:
            raise self.replay_error
        for index, row in enumerate(actions):
            if 'on_tick' in recording:
                recording['on_tick'](index, row)
        return self.replay_result and not should_stop()

    def enter_safe_stop(self):
        self.events.append('safe stop')

    def setup_for_recording(self):
        self.events.append('start position')

    def enter_motion_mode(self):
        self.events.append('hold')


def run(backend, keys, monkeypatch, actions=None):
    actions = np.arange(21, dtype=float).reshape(3, 7) if actions is None else actions
    reference = np.full((4, 6, 3), 50, np.uint8)
    monkeypatch.setattr(recording_replay, 'load_replay_episode',
                        lambda *args, **kwargs: (actions, 'video_left', reference))
    shown, queue = [], list(keys)
    result = recording_replay.replay_episode(
        backend, 'episode', ['video_left'], poll_keys=lambda: [queue.pop(0)] if queue else [],
        live_frame=lambda name: np.zeros((4, 6, 3), np.uint8), notify=lambda *_: None,
        show=shown.append, pump=lambda: [], close_window=lambda: None, sleep=lambda _: None)
    return result, shown


def test_moves_to_recorded_start_shows_overlay_then_replays_on_3(monkeypatch):
    backend = FakeBackend()
    result, shown = run(backend, [None, None, '1'], monkeypatch)
    assert result == 'completed'
    assert backend.events[0][0] == 'move'
    np.testing.assert_array_equal(backend.events[0][1], np.arange(7))  # recorded first row
    assert backend.events[1] == ('replay', 3, 100.0)
    assert backend.events[2] == 'hold'
    assert len(shown) == 3 and shown[0].shape == (4, 6, 3)  # overlay drawn until the go-ahead


@pytest.mark.parametrize('key, result', [('h', 'cancelled'), ('n', 'cancelled'), ('q', 'quit')])
def test_overlay_can_cancel_or_quit_without_replaying(monkeypatch, key, result):
    backend = FakeBackend()
    assert run(backend, [key], monkeypatch)[0] == result
    assert [event[0] for event in backend.events] == ['move']


def test_stop_key_during_replay_safe_stops_then_h_returns_to_start(monkeypatch):
    backend = FakeBackend()
    backend.replay_result = False
    result, _ = run(backend, ['1', 'x', 'h'], monkeypatch)
    assert result == 'stopped'
    assert backend.events[-2:] == ['safe stop', 'start position']


@pytest.mark.parametrize('error', [RuntimeError('camera'), KeyboardInterrupt()])
def test_errors_or_ctrl_c_during_replay_reach_the_collector_failure_exit(monkeypatch, error):
    # The collector's damping_exit handles them: damping until 'h', then home.
    backend = FakeBackend()
    backend.replay_error = error
    with pytest.raises(type(error)):
        run(backend, ['1'], monkeypatch)
    assert 'safe stop' not in backend.events


def test_error_while_returning_to_start_reaches_the_failure_exit(monkeypatch):
    backend = FakeBackend()
    backend.move_error = ValueError('joint2 target is outside limits')
    with pytest.raises(ValueError, match='outside limits'):
        run(backend, [], monkeypatch)
    assert backend.events == [('move', backend.events[0][1])]


def type_keys(monkeypatch, text):
    from recording import record_episode_single as recorder
    keys = list(text)
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: keys.pop(0))
    monkeypatch.setattr(recorder, '_replay_gain_scales', [1.0, 1.0])
    return recorder


def test_gain_prompt_reads_typed_fractions_and_remembers_them(monkeypatch, capsys):
    recorder = type_keys(monkeypatch, '0.5\n0.3\n' + '\n\n' + 'x\x7f.8\n\n')
    assert recorder.prompt_replay_gain_scales() == (0.5, 0.3)
    assert recorder.prompt_replay_gain_scales() == (0.5, 0.3)  # Enter keeps both
    assert recorder.prompt_replay_gain_scales() == (0.8, 0.3)  # backspace edits


def test_gain_prompt_reasks_invalid_values_and_esc_cancels(monkeypatch, capsys):
    recorder = type_keys(monkeypatch, '3\nabc\n0.4\n\x1b')
    assert recorder.prompt_replay_gain_scales() is None
    assert 'kp fraction must be in' in capsys.readouterr().out
    assert recorder._replay_gain_scales == [1.0, 1.0]  # cancelled values are not kept



def test_prompt_replays_last_saved_episode_on_r(monkeypatch, tmp_path):
    from recording import record_episode_single as recorder
    keys = iter(['r', 'r', '\n'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    loaded, played = [], []
    monkeypatch.setattr(recording_replay, 'load_replay_episode',
                        lambda path, **kwargs: loaded.append(path) or (np.zeros((2, 7)), 'video_left', None))
    monkeypatch.setattr(recording_replay, 'replay_take',
                        lambda backend, actions, camera, reference, **kwargs: played.append(kwargs['label']) or 'completed')
    ctx = SimpleNamespace(dataset_dir=str(tmp_path), camera_names=['video_left'], robot_contract={}, fps=100.0)
    backend = SimpleNamespace(replay_rows=None)
    assert recorder.prompt_next_episode(ctx, backend, None) is True
    assert loaded == []  # no saved episode yet: both 'r' presses only print
    (tmp_path / 'episode_00004.zarr').mkdir()
    keys = iter(['r', 'x'])
    assert recorder.prompt_next_episode(ctx, backend, None) is False
    assert played == [str(tmp_path / 'episode_00004.zarr')]
    keys = iter(['r', '\n'])
    recorder.prompt_next_episode(ctx, backend, '/data/episode_00007.zarr')
    assert played[-1] == '/data/episode_00007.zarr'  # this session's last save wins
    keys = iter(['r', '\n'])
    assert recorder.prompt_next_episode(ctx, SimpleNamespace(), None) is True  # ARX: r ignored
    assert len(played) == 2


def test_unreadable_saved_episode_is_reported_without_moving(monkeypatch, tmp_path):
    from recording import record_episode_single as recorder
    keys = iter(['r', '\n'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)

    def fail(path, **kwargs):
        raise ValueError('rate differs')
    monkeypatch.setattr(recording_replay, 'load_replay_episode', fail)
    backend = SimpleNamespace(replay_rows=None, move_to_pose=lambda *_: pytest.fail('moved'))
    ctx = SimpleNamespace(dataset_dir=str(tmp_path), camera_names=['video_left'], robot_contract={}, fps=100.0)
    assert recorder.prompt_next_episode(ctx, backend, '/data/episode_00001.zarr') is True


class Sample:
    def __init__(self, qpos, images):
        self.qpos, self.images = np.asarray(qpos, float), images


class Take(list):
    finished = False

    def finish(self):
        self.finished = True


def test_unsaved_take_source_reads_rows_and_first_frame_after_finishing():
    first = np.full((4, 6, 3), 7, np.uint8)
    take = Take([Sample(np.arange(7) + i, {'video_left': first if i == 0 else None,
                                            'video_overhead': first + 1 if i == 0 else None})
                 for i in range(3)])
    actions, camera, reference = recording_replay.take_replay_source(take, ['video_left', 'video_overhead'])
    assert take.finished
    np.testing.assert_array_equal(actions, [np.arange(7) + i for i in range(3)])
    assert camera == 'video_overhead'
    np.testing.assert_array_equal(reference, first + 1)


def test_review_replays_the_unsaved_take_then_still_asks_to_save(monkeypatch):
    from recording import record_episode_single as recorder
    keys = iter(['r', 'r', '1'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    take = Take([Sample(np.zeros(7), {'video_left': np.zeros((4, 6, 3), np.uint8)}) for _ in range(3)])
    played = []
    monkeypatch.setattr(recording_replay, 'replay_take',
                        lambda backend, actions, camera, reference, **kwargs:
                        played.append((len(actions), camera, kwargs['label'])) or 'completed')
    backend = SimpleNamespace(replay_rows=None, ctx=SimpleNamespace(camera_names=['video_left']))
    assert recorder.prompt_accept_episode(backend=backend, samples=take) is True  # saved after 2 replays
    assert played == [(3, 'video_left', 'unsaved take')] * 2


@pytest.mark.parametrize('decision, saved', [('h', False), ('n', False)])
def test_review_can_discard_after_replaying(monkeypatch, decision, saved):
    from recording import record_episode_single as recorder
    keys = iter(['r', decision])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    monkeypatch.setattr(recording_replay, 'replay_take', lambda *args, **kwargs: 'stopped')
    take = Take([Sample(np.zeros(7), {'video_left': np.zeros((4, 6, 3), np.uint8)}) for _ in range(2)])
    backend = SimpleNamespace(replay_rows=None, ctx=SimpleNamespace(camera_names=['video_left']))
    assert recorder.prompt_accept_episode(backend=backend, samples=take) is saved


def test_quit_during_review_replay_shuts_down(monkeypatch):
    from recording import record_episode_single as recorder
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: 'r')
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    monkeypatch.setattr(recording_replay, 'replay_take', lambda *args, **kwargs: 'quit')
    take = Take([Sample(np.zeros(7), {'video_left': np.zeros((4, 6, 3), np.uint8)}) for _ in range(2)])
    backend = SimpleNamespace(replay_rows=None, ctx=SimpleNamespace(camera_names=['video_left']))
    with pytest.raises(KeyboardInterrupt):
        recorder.prompt_accept_episode(backend=backend, samples=take)


def test_unreplayable_take_keeps_the_review_open(monkeypatch):
    from recording import record_episode_single as recorder
    keys = iter(['r', '1'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)

    class Broken(Take):
        def finish(self):
            raise RuntimeError('Raw stream timing failed validation')
    backend = SimpleNamespace(replay_rows=None, ctx=SimpleNamespace(camera_names=['video_left']),
                              move_to_pose=lambda *_: pytest.fail('moved'))
    assert recorder.prompt_accept_episode(backend=backend, samples=Broken([1, 2])) is True


def test_queued_keys_are_discarded_after_replay(monkeypatch):
    backend = FakeBackend()
    # Go-ahead, a stop press during playback, a stray press in safe stop, 'h',
    # then two more stray presses that must not reach the save prompt.
    queue = ['1', '1', '1', 'h', '1', '1']
    reference = np.zeros((4, 6, 3), np.uint8)
    result = recording_replay.replay_take(
        backend, np.zeros((3, 7)), 'video_left', reference,
        poll_keys=lambda: [queue.pop(0)] if queue else [],
        live_frame=lambda name: reference, notify=lambda *_: None, show=lambda _: None,
        pump=lambda: [], close_window=lambda: None, sleep=lambda _: None)
    assert result == 'stopped'
    assert backend.events[-2:] == ['safe stop', 'start position']
    assert queue == []  # nothing left to reach the save prompt



def test_gain_prompt_fractions_reach_the_replay_and_esc_skips_it(monkeypatch):
    from recording import record_episode_single as recorder
    answers = iter([(0.6, 0.4), None])
    monkeypatch.setattr(recorder, 'prompt_replay_gain_scales', lambda: next(answers))
    monkeypatch.setattr(recorder, 'wait_for_cameras', lambda *args, **kwargs: None)
    keys = iter(['r', 'r', '1'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    played = []
    monkeypatch.setattr(recording_replay, 'replay_take',
                        lambda *args, **kwargs: played.append(kwargs['gain_scales']) or 'completed')
    take = Take([Sample(np.zeros(7), {'video_left': np.zeros((4, 6, 3), np.uint8)}) for _ in range(2)])
    backend = SimpleNamespace(replay_rows=None, ctx=SimpleNamespace(camera_names=['video_left']))
    assert recorder.prompt_accept_episode(backend=backend, samples=take) is True
    assert played == [(0.6, 0.4)]  # the second 'r' was cancelled at the prompt


def test_gain_prompt_fractions_are_passed_to_the_backend(monkeypatch):
    backend = FakeBackend()
    run_keys = ['1']
    reference = np.full((4, 6, 3), 50, np.uint8)
    assert recording_replay.replay_take(
        backend, np.zeros((3, 7)), 'video_left', reference, gain_scales=(0.7, 1.5),
        poll_keys=lambda: [run_keys.pop(0)] if run_keys else [],
        live_frame=lambda name: np.zeros((4, 6, 3), np.uint8), notify=lambda *_: None,
        show=lambda _: None, pump=lambda: [], close_window=lambda: None,
        sleep=lambda _: None) == 'completed'
    assert backend.gain_scales == (0.7, 1.5)


def test_review_uses_the_two_pedal_keys(monkeypatch):
    from recording import record_episode_single as recorder
    for presses, saved in ((['3', '1'], True), (['3', 'h'], False)):
        keys = iter(presses)  # the old '3' key is ignored
        monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
        assert recorder.prompt_accept_episode() is saved


class Recording:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append('record start')

    def sample(self, index, row):
        self.events.append(('record', index))

    def stop(self):
        self.events.append('record stop')


def replay_recorded(backend, keys):
    reference = np.zeros((4, 6, 3), np.uint8)
    return recording_replay.replay_take(
        backend, np.zeros((3, 7)), 'video_left', reference, record=Recording(backend.events),
        poll_keys=lambda: [keys.pop(0)] if keys else [],
        live_frame=lambda name: reference, notify=lambda *_: None, show=lambda _: None,
        pump=lambda: [], close_window=lambda: None, sleep=lambda _: None)


def test_recorded_replay_opens_streams_after_go_and_drains_when_rows_end():
    backend = FakeBackend()
    assert replay_recorded(backend, ['1']) == 'completed'
    assert [event if isinstance(event, str) else event[:2] for event in backend.events[1:]] == [
        'record start', ('replay', 3), ('record', 0), ('record', 1), ('record', 2), 'record stop', 'hold']


def test_recorded_replay_drains_before_the_stop_damping():
    backend = FakeBackend()
    backend.replay_result = False
    assert replay_recorded(backend, ['1', 'h']) == 'stopped'
    assert backend.events[-3:] == ['record stop', 'safe stop', 'start position']


@pytest.mark.parametrize('key', ['h', 'q'])
def test_cancelled_overlay_never_opens_replay_streams(key):
    backend = FakeBackend()
    replay_recorded(backend, [key])
    assert 'record start' not in backend.events


def test_replay_error_leaves_draining_to_the_recording_scope():
    backend = FakeBackend()
    backend.replay_error = RuntimeError('stale feedback')
    with pytest.raises(RuntimeError, match='stale'):
        replay_recorded(backend, ['1'])
    assert 'record start' in backend.events and 'record stop' not in backend.events


def test_next_prompt_records_one_replay_on_go_then_prepares_another(monkeypatch):
    from recording import record_episode_single as recorder
    keys = iter(['1', '1'])
    monkeypatch.setattr(recorder, 'read_key_blocking', lambda: next(keys))
    recorded = []
    ctx = SimpleNamespace(dataset_dir='/data', camera_names=['video_left'], robot_contract={}, fps=100.0)
    backend = SimpleNamespace(replay_rows=None)
    assert recorder.prompt_next_episode(ctx, backend, '/data/episode_00003.zarr',
                                        record_replay=lambda: recorded.append(1) or False) is True
    assert recorded == [1]
    keys = iter(['1'])
    assert recorder.prompt_next_episode(ctx, backend, '/data/episode_00003.zarr',
                                        record_replay=lambda: True) is False  # quit during it
    keys = iter(['\n'])
    assert recorder.prompt_next_episode(ctx, backend, '/data/episode_00003.zarr',
                                        record_replay=lambda: pytest.fail('replayed')) is True
