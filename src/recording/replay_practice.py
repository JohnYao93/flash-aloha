#!/usr/bin/env python3
"""Pedal-driven record/replay test for one or both ARX arms.

No task, no cameras, no time limit, no episode written. The take is kept in
memory until the next recording replaces it.

    idle (arms passive)   3 = start recording          q = quit
    recording             3 = stop
    review                2 = replay   3 = record again   1 = release arms
    replay                1 = play once positioned; 1 again = stop (damping; h returns)
                          h / n = cancel before playing    q = quit

Replay gains are two numbers, each a multiple of the tracking kp and kd:

    up / down arrow  (or w / s)   kp x +0.05 / -0.05
    right / left arrow (or d / a) kd x +0.05 / -0.05

They can be changed at any time except while recording or replaying, apply to
the next replay, and are not saved.

This is an operator of the collector's own pieces, used unchanged: the ARX
backend, `standard_capture` via `EpisodeCapture`, and the collector's review
replay (`recording_replay.replay_take`), including its tracking log under
logs/replay and its failure exit (damping until `h`, then home).
"""
from __future__ import annotations

# Support direct script execution and package imports from any working directory.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

import argparse
import time

import numpy as np

from config.constants import ARX_SINGLE_ARM_CAN_BY_SIDE
from utils.hardware_utils import DEFAULT_ARM_KD, DEFAULT_ARM_KP

QUIT_KEYS = ('q', 'x', '\x1b', '\x03')
IDLE_POLL_SECONDS = 0.01
GAIN_STEP = 0.05
GAIN_SCALE_RANGE = (0.1, 2.0)
# The key watcher lowercases keys, so arrows arrive as '\x1b[a' .. '\x1b[d'.
GAIN_KEYS = {'\x1b[a': ('kp', 1), 'w': ('kp', 1), '\x1b[b': ('kp', -1), 's': ('kp', -1),
             '\x1b[c': ('kd', 1), 'd': ('kd', 1), '\x1b[d': ('kd', -1), 'a': ('kd', -1)}


def gain_scale(text):
    value = float(text)
    low, high = GAIN_SCALE_RANGE
    if not low <= value <= high:
        raise argparse.ArgumentTypeError(f'gain scale must be in [{low:g}, {high:g}]')
    return value


class ReplayGains:
    """The two replay numbers: multiples of the default joint kp and kd."""

    def __init__(self, kp=1.0, kd=1.0):
        self.kp, self.kd = gain_scale(kp), gain_scale(kd)

    def __str__(self):
        kp, kd = np.asarray(DEFAULT_ARM_KP[:6]) * self.kp, np.asarray(DEFAULT_ARM_KD[:6]) * self.kd
        return (f'Replay gains: kp x{self.kp:.2f}, kd x{self.kd:.2f}  '
                f'(joints 1-3 kp {kp[0]:g} kd {kd[0]:g}; joints 4-6 kp {kp[3]:g} kd {kd[3]:g})')

    def adjust(self, key):
        """Step one number for a gain key. Return whether `key` was a gain key."""
        if key not in GAIN_KEYS:
            return False
        name, direction = GAIN_KEYS[key]
        low, high = GAIN_SCALE_RANGE
        value = round(getattr(self, name) + direction * GAIN_STEP, 2)
        setattr(self, name, min(high, max(low, value)))
        return True


def replay_in_memory_take(backend, samples, gains, read_key):
    """The collector's review replay of the unsaved take; returns its result string."""
    from recording.recording_replay import replay_take, take_replay_source
    actions, camera, reference = take_replay_source(samples, [])
    return replay_take(backend, actions, camera, reference, label='practice take',
                       gain_scales=(gains.kp, gains.kd),
                       poll_keys=lambda: [key for key in (read_key(),) if key],
                       live_frame=lambda name: None)


def build_context(arms, *, fps, can_left, can_right, start_position_path, move_duration):
    from recording.record_episode_single import RunContext
    both = arms == 'both'
    return RunContext(
        task_name='replay_practice', dataset_dir='', max_timesteps=0, camera_names=[],
        robot_type='arx', single_arm=not both, double_arm=both,
        # In single-arm mode the backend drives `can_interface_left` as the active arm.
        can_interface_left=can_left if arms != 'right' else can_right,
        can_interface_right=can_right if both else None,
        arx_single_arm_side=None if both else arms,
        fps=float(fps), use_start_position=start_position_path is not None,
        start_position_path=start_position_path,
        start_position_move_duration=float(move_duration))


def record_take(backend, read_key, *, fps, make_capture, notify=print):
    """Sample proprio at `fps` until pedal 3. Return (samples, quit_requested)."""
    capture = make_capture(backend, [], fps=fps)
    samples = []
    notify('Recording... pedal 3 stops.')
    while True:
        samples.append(capture.wait_and_sample())
        key = read_key()
        if key == '3':
            break
        if key in QUIT_KEYS:
            return samples, True
    seconds = samples[-1].timestamp - samples[0].timestamp
    rate = (len(samples) - 1) / seconds if seconds > 0 else 0.0
    notify(f'Stopped: {len(samples)} samples, {seconds:.2f} s, {rate:.1f} Hz.')
    return samples, False


def run(backend, read_key, *, fps, make_capture, replay, gains=None, notify=print, sleep=time.sleep):
    """Pedal loop. Returns when the operator quits; the caller owns shutdown."""
    samples = None
    gains = gains or ReplayGains()
    notify('Arms are passive. Pedal 3 starts recording; q quits.')
    notify(f'{gains}. Arrows or w/s (kp) and d/a (kd) change them.')
    while True:
        backend.passive_step()
        key = read_key()
        if key is None:
            sleep(IDLE_POLL_SECONDS)
        elif key in QUIT_KEYS:
            return
        elif key == '3':
            # Review/replay may have left the arms holding; hand them back first.
            backend.enable_passive_compensation()
            samples, quit_requested = record_take(
                backend, read_key, fps=fps, make_capture=make_capture, notify=notify)
            if quit_requested:
                return
            notify('Review: pedal 2 replays, 3 records again, 1 releases the arms; q quits.')
        elif key == '2':
            if samples is None or len(samples) < 2:
                notify('Nothing to replay yet; pedal 3 records.')
            elif replay(backend, samples, gains, read_key) == 'quit':
                return
            else:
                notify('Review: pedal 2 replays, 3 records again, 1 releases the arms; q quits.')
        elif key == '1':
            backend.enable_passive_compensation()
            notify('Arms released. The last take is kept: 2 replays it, 3 records a new one.')
        elif gains.adjust(key):
            # The arms are passive or holding here; the next replay applies the new fractions.
            notify(str(gains))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', choices=['left', 'right', 'both'], required=True)
    parser.add_argument('--fps', type=float, default=100.0, help='Proprio sample and replay rate (default: 100).')
    parser.add_argument('--can-left', default=ARX_SINGLE_ARM_CAN_BY_SIDE['left'])
    parser.add_argument('--can-right', default=ARX_SINGLE_ARM_CAN_BY_SIDE['right'])
    parser.add_argument('--start-position', dest='start_position_path', nargs='?', default=None,
                        const='calibration/start_position.npz',
                        help='Move to this saved start pose first; without it the arms go passive where they are.')
    parser.add_argument('--kp-scale', type=gain_scale, default=1.0,
                        help='Starting replay kp as a multiple of the default joint kp (default: 1).')
    parser.add_argument('--kd-scale', type=gain_scale, default=1.0,
                        help='Starting replay kd as a multiple of the default joint kd (default: 1).')
    parser.add_argument('--move-duration', type=float, default=3.0,
                        help='Seconds to reach the start pose and the first replay pose (default: 3).')
    return parser


def main(args):
    from hardware.safe_exit import QuitRequested, damping_exit, raising_signals
    from recording.record_episode_single import ARXBackend, EpisodeCapture, home_key_pressed, read_key_nonblocking
    ctx = build_context(args.arms, fps=args.fps, can_left=args.can_left, can_right=args.can_right,
                        start_position_path=args.start_position_path, move_duration=args.move_duration)
    print(f'[Config] arms={args.arms} fps={ctx.fps:g} cameras=none; no episode is saved')
    backend = None
    failed = False
    try:
        with raising_signals():
            # Any failure, including Ctrl+C, damps until 'h', homes, then exits.
            with damping_exit(lambda: None if backend is None else backend.damping_target(),
                              home_requested=home_key_pressed):
                backend = ARXBackend(ctx)
                backend.setup_for_recording()
                backend.enable_passive_compensation()  # Practice starts with the arms released.
                gains = ReplayGains(args.kp_scale, args.kd_scale)
                try:
                    run(backend, read_key_nonblocking, fps=ctx.fps, make_capture=EpisodeCapture,
                        replay=replay_in_memory_take, gains=gains)
                finally:
                    print(f'Last used. {gains}')
    except QuitRequested:
        pass
    except BaseException:
        failed = True  # The failure exit already homed or disabled in place.
        raise
    finally:
        if backend is not None:
            backend.shutdown(home=not failed)
        print('Shutdown complete.')


if __name__ == '__main__':
    main(build_arg_parser().parse_args())
