"""Pedal record/replay test tool with a fake arm, scripted pedals and a fake clock."""
import numpy as np
import pytest

from recording import replay_practice as tool
from recording.record_episode_single import EpisodeCapture


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 1e-4)


class Arm:
    """Hand-guided fake: position advances on every state read."""

    def __init__(self, dims):
        self.dims, self.reads, self.passive_calls = dims, 0, 0

    def passive_step(self):
        pass

    def enable_passive_compensation(self):
        self.passive_calls += 1

    def read_state(self):
        self.reads += 1
        return (np.full(self.dims, self.reads / 1000, dtype=np.float32),
                np.zeros(self.dims, dtype=np.float32))


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """The capture loop sleeps on the real clock; advance the fake one instead."""
    clock = Clock()
    monkeypatch.setattr('recording.record_episode_single.time.sleep', clock.sleep)
    return clock


def pedals(script):
    """Yield one scripted key per poll; None means no pedal that tick."""
    keys = iter(script)
    return lambda: next(keys)


@pytest.mark.parametrize('dims', [7, 14])
def test_record_until_pedal_then_replay_and_rerecord(dims, clock):
    arm, replays, log = Arm(dims), [], []

    def replay(backend, samples, gains, read_key):
        replays.append(np.asarray([sample.qpos for sample in samples]))
        return 'completed'

    script = ([None, '2', '3'] + [None] * 249 + ['3']      # no take yet; record 250 samples
              + ['2', '1', '2']                              # replay, release, replay the kept take
              + ['3'] + [None] * 4 + ['3', '2', 'q'])        # record 5 more, replay, quit
    tool.run(arm, pedals(script), fps=100,
             make_capture=lambda *a, **k: EpisodeCapture(*a, clock=clock, **k),
             replay=replay, notify=log.append, sleep=clock.sleep)

    assert [take.shape for take in replays] == [(250, dims), (250, dims), (5, dims)]
    np.testing.assert_array_equal(replays[0], replays[1])
    assert np.all(np.diff(replays[0][:, 0]) > 0)            # every tick sampled, in order
    assert arm.passive_calls == 3                            # two recordings and one release
    assert 'Nothing to replay yet; pedal 3 records.' in log
    assert any('250 samples' in line and '100.0 Hz' in line for line in log)


def test_quit_while_recording_and_from_replay(clock):
    capture = lambda *a, **k: EpisodeCapture(*a, clock=clock, **k)
    arm = Arm(7)
    tool.run(arm, pedals(['3', None, 'q']), fps=100, make_capture=capture,
             replay=None, notify=lambda _: None, sleep=clock.sleep)
    arm = Arm(7)
    tool.run(arm, pedals(['3', None, '3', '2']), fps=100, make_capture=capture,
             replay=lambda *a: 'quit', notify=lambda _: None, sleep=clock.sleep)


def test_take_replays_through_the_collectors_review_replay(clock):
    """The tool hands its in-memory take, row for row, to the collector's replay_take."""
    from types import SimpleNamespace
    arm = Arm(7)
    samples, _ = tool.record_take(arm, pedals([None] * 9 + ['3']), fps=100, notify=lambda _: None,
                                  make_capture=lambda *a, **k: EpisodeCapture(*a, clock=clock, **k))
    events = []

    class Backend:
        ctx = SimpleNamespace(fps=100.0)

        def move_to_pose(self, qpos):
            events.append(('move', np.asarray(qpos).copy()))

        def replay_rows(self, actions, *, fps, should_stop, kp_scale, kd_scale, log=None):
            events.append(('replay', np.asarray(actions).copy(), fps, kp_scale, kd_scale))
            return True

        def enter_motion_mode(self):
            events.append('hold')

    keys = iter([None, '1'])                         # the go-ahead once positioned
    result = tool.replay_in_memory_take(Backend(), samples, tool.ReplayGains(kp=0.5, kd=0.75),
                                        lambda: next(keys, None))
    assert result == 'completed'
    recorded = np.asarray([sample.qpos for sample in samples], dtype=np.float64)
    np.testing.assert_allclose(events[0][1], recorded[0])
    kind, rows, fps, kp_scale, kd_scale = events[1]
    np.testing.assert_allclose(rows, recorded)
    assert (kind, fps, kp_scale, kd_scale) == ('replay', 100.0, 0.5, 0.75) and events[-1] == 'hold'
    keys = iter(['h'])                               # cancelled at the go-ahead: nothing replays
    assert tool.replay_in_memory_take(Backend(), samples, tool.ReplayGains(), lambda: next(keys, None)) == 'cancelled'
    assert [event[0] for event in events if isinstance(event, tuple)].count('replay') == 1


def test_gain_keys_step_and_clamp_the_two_numbers():
    gains = tool.ReplayGains()
    for key in ('w', '\x1b[a', 's', 'a', '\x1b[d'):
        assert gains.adjust(key)
    assert (gains.kp, gains.kd) == (1.05, 0.9)
    assert not gains.adjust('2') and not gains.adjust(None)
    for _ in range(100):
        gains.adjust('s'), gains.adjust('d')
    assert (gains.kp, gains.kd) == (0.1, 2.0)
    with pytest.raises(Exception):
        tool.ReplayGains(kp=0.0)


def test_gain_keys_outside_a_replay_only_change_the_numbers(clock):
    arm, gains, log = Arm(7), tool.ReplayGains(), []
    tool.run(arm, pedals(['w', 'w', 'a', 'q']), fps=100, make_capture=None, replay=None,
             gains=gains, notify=log.append, sleep=clock.sleep)
    assert (gains.kp, gains.kd) == (1.1, 0.95)
    assert arm.passive_calls == 0 and 'kp x1.10, kd x0.95' in log[-1]
