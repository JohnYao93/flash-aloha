#!/usr/bin/env python3
"""Demonstration recording and offline publication of stopped recordings.

    uv run scripts/record.py single --task_name single_arm
    uv run scripts/record.py publish --data-dir RECORDING_ROOT

scripts/run.py resolves these arguments from a task name.
"""
from _launcher import main

MODES = {
    'single': ('src/recording/record_episode_single.py', 'hand-guided ARX collection, with review replay'),
    'publish': ('src/recording/distributed_recording.py',
                'publish completed rollout/DAgger recordings as episodes (no robot)'),
}

if __name__ == '__main__':
    main(__doc__.splitlines()[0], MODES)
