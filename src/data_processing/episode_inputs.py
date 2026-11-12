"""Episode inputs shared by policy validation."""
from pathlib import Path
import numpy as np
from data_processing.episode_data import find_episodes, is_episode, open_episode, read_json
from contracts.action_rate_contract import resolve_rate


def discover_episodes(directory, fps=None):
    paths = [path for path in find_episodes(directory) if is_episode(path)]
    if not paths:
        raise FileNotFoundError(f'No complete episodes found in {directory}')
    if fps is not None:
        # Episode uses this same authoritative declaration. Read only the
        # manifests here; opening every episode would parse unused camera logs.
        for path in paths:
            resolve_rate(read_json(Path(path) / 'manifest.json'), fps, context='Raw episode FPS')
    return paths


def read_robot_array(root, path, action_dim):
    value = np.asarray(root[path], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < action_dim or not np.isfinite(value).all():
        raise ValueError(f'{root.filename}:{path} must contain finite rows with at least {action_dim} values')
    return value[:, :action_dim]


def compute_stats(paths, action_dim):
    qpos, action = [], []
    for path in paths:
        with open_episode(path) as root:
            qpos.append(read_robot_array(root, '/observations/qpos', action_dim))
            action.append(read_robot_array(root, '/action', action_dim))
    qpos, action = np.concatenate(qpos), np.concatenate(action)
    return dict(qpos=qpos, action=action, action_for_range=action,
                action_min=action.min(axis=0), action_max=action.max(axis=0))


def row_metadata(root, index):
    """Resolve metadata by its saved ID, never by a nearby timestamp."""
    metadata = dict(root.attrs)
    if 'metadata_id' in root.control and 'metadata_segments' in root.manifest:
        raw_index = int(root.retained_indices[index])
        identifier = int(_metadata_ids(root)[index])
        segments = root.manifest.get('metadata_segments', [])
        if identifier >= 0:
            if identifier >= len(segments):
                raise ValueError(f'Missing metadata segment {identifier} for row {raw_index}')
            metadata.update(segments[identifier])
    return metadata


def _metadata_ids(root):
    if not hasattr(root, '_dry_metadata_ids'):
        root._dry_metadata_ids = np.asarray(root.control['metadata_id'])[root.retained_indices]
    return root._dry_metadata_ids


def causal_goal_row(root, index, rng):
    """Sample a saved earlier row within this retained primitive and continuity run."""
    if 'metadata_id' not in root.control:
        raise ValueError('Historical task goals require recorded metadata IDs')
    if not hasattr(root, '_dry_goal_starts'):
        identifiers = _metadata_ids(root)
        changes = np.r_[True, (identifiers[1:] != identifiers[:-1]) | (root.runs[1:] != root.runs[:-1])]
        root._dry_goal_starts = np.maximum.accumulate(np.where(changes, np.arange(len(identifiers)), 0))
    start = int(root._dry_goal_starts[index])
    return int(rng.integers(start, index+1))
