"""Select replay or hand-guided episodes by their recording metadata.

Hand-guided takes record the measured qpos as their action; replays (metadata
``replay_recording``) record the rows commanded while the arm tracked them.
"""
from data_processing.episode_data import open_episode


def is_replay_episode(path):
    with open_episode(path) as root:
        return bool(root.attrs.get('replay_recording', False))


def replay_filter(mode):
    """Episode path predicate for data.replay_episodes: all, only (replays) or exclude (hand-guided)."""
    if mode in (None, 'all'):
        return lambda path: True
    if mode not in ('only', 'exclude'):
        raise ValueError(f'replay_episodes must be all, only or exclude, got {mode!r}')
    cache = {}

    def keep(path):
        if path not in cache:
            cache[path] = is_replay_episode(path)
        return cache[path] if mode == 'only' else not cache[path]
    return keep
