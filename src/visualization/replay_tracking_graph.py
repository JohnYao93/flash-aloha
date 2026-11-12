#!/usr/bin/env python3
"""Graph of commanded and achieved positions for one physical replay log.

Reads a replay's step log (``logs/replay/<date>/<time>_<name>/steps_*.npz``,
written by `utils.run_log.RunLog`): the row sent on each tick and the positions
measured right after sending it. One panel per joint and the gripper, one
column per arm. Saves ``tracking.png`` in the log directory and shows it.

    uv run src/visualization/replay_tracking_graph.py logs/replay/2026-11-12/101500_practice_take

It only reads the log. Replay practice's ``--graph`` runs it in a separate
process after each replay, so nothing is drawn in the process that drives the arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRAPH_NAME = 'tracking.png'
JOINTS_PER_ARM = 7
PANEL_NAMES = [f'Joint {index}' for index in range(1, 7)] + ['Gripper']


def load_tracking(directory):
    """Return (seconds, commanded, achieved, settings) for a replay log directory."""
    directory = Path(directory)
    files = sorted(directory.glob('steps_*.npz'))
    if not files:
        raise FileNotFoundError(f'No steps_*.npz replay log in {directory}')
    times, commanded, achieved = [], [], []
    for path in files:
        with np.load(path) as steps:
            times.append(steps['time'])
            commanded.append(steps['target'])
            achieved.append(steps['qpos'])
    times = np.concatenate(times).astype(np.float64)
    commanded, achieved = np.concatenate(commanded), np.concatenate(achieved)
    if commanded.ndim != 2 or commanded.shape != achieved.shape or commanded.shape[1] % JOINTS_PER_ARM:
        raise ValueError(f'Unexpected replay log shapes {commanded.shape} and {achieved.shape}')
    settings_path = directory / 'run.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    return times - times[0], commanded, achieved, settings


def arm_labels(arm_count, arms=None):
    if arms and len(arms) == arm_count:
        return list(arms)
    return ['Arm'] if arm_count == 1 else ['Left arm', 'Right arm'][:arm_count]


def draw_tracking(seconds, commanded, achieved, settings=None, arms=None):
    """Build the figure: commanded and achieved position per joint, one column per arm."""
    import matplotlib.pyplot as plt
    settings = settings or {}
    arm_count = commanded.shape[1] // JOINTS_PER_ARM
    labels = arm_labels(arm_count, arms)
    figure, axes = plt.subplots(JOINTS_PER_ARM, arm_count, sharex=True, squeeze=False,
                                figsize=(7.5 * arm_count, 13))
    for arm in range(arm_count):
        for joint in range(JOINTS_PER_ARM):
            column = arm * JOINTS_PER_ARM + joint
            axis = axes[joint][arm]
            error = achieved[:, column] - commanded[:, column]
            axis.plot(seconds, commanded[:, column], color='tab:blue', linewidth=1.2, label='commanded')
            axis.plot(seconds, achieved[:, column], color='tab:orange', linewidth=1.2, label='achieved')
            unit = ' rad' if joint < 6 else ''
            axis.set_title(f'{labels[arm]}, {PANEL_NAMES[joint]}: error RMS {np.sqrt(np.mean(error ** 2)):.4f}'
                           f'{unit}, max {np.abs(error).max():.4f}{unit}', fontsize=9)
            axis.set_ylabel('position' if joint == 6 else 'rad', fontsize=8)
            axis.grid(True, alpha=0.3)
        axes[-1][arm].set_xlabel('seconds')
    axes[0][0].legend(loc='upper right', fontsize=8)
    title = 'Replay tracking: commanded and achieved position'
    if 'kp_scale' in settings and 'kd_scale' in settings:
        title += f"  (kp x{float(settings['kp_scale']):.2f}, kd x{float(settings['kd_scale']):.2f}"
        title += f", {float(settings['fps']):g} Hz)" if 'fps' in settings else ')'
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', help='Replay log directory holding steps_*.npz.')
    parser.add_argument('--arms', nargs='+', default=None, help='Column titles, one per arm.')
    parser.add_argument('--no-show', action='store_true', help='Save the image without opening a window.')
    args = parser.parse_args(argv)
    seconds, commanded, achieved, settings = load_tracking(args.directory)
    import matplotlib.pyplot as plt
    figure = draw_tracking(seconds, commanded, achieved, settings, args.arms)
    path = Path(args.directory) / GRAPH_NAME
    figure.savefig(path, dpi=120)
    print(f'Saved {path}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
