#!/usr/bin/env python3
"""Episode viewing and physical replay.

    python scripts/visualize.py episode --task_name single_arm --episode_idx 2 --rerun
    python scripts/visualize.py episode --task_name single_arm --review_replay 24 27

scripts/run.py resolves the episode arguments from a task name.
"""
from _launcher import main

MODES = {
    'episode': ('src/visualization/visualize_episode.py',
                'Rerun/video viewing and physical replay of recorded episodes'),
}

if __name__ == '__main__':
    main(__doc__.splitlines()[0], MODES)
