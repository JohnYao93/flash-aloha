"""Task-independent continuity for retained trajectories and training windows.

Recording gaps and excluded rows always break a sequence. Task code may supply
additional boundary labels; this module never interprets goals or subtasks.
"""
import numpy as np

OBSERVATION_CONTINUITY_PATHS = (
    'observations/capture_continuity_index', 'recording_timing/continuity_id', 'continuity_id')


def contiguous_runs(indices, *boundary_labels):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError('Retained indices must be a strictly increasing integer vector')
    changes = np.diff(indices) != 1
    for labels in boundary_labels:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.dtype.kind not in 'iub' or (len(indices) and indices[-1] >= len(labels)):
            raise ValueError('Boundary labels must be integer vectors covering retained indices')
        changes |= labels[indices[1:]] != labels[indices[:-1]]
    return np.cumsum(np.r_[True, changes]).astype(np.int64) - 1 if len(indices) else np.empty(0, np.int64)


def training_runs(root, count, *, raw=False, boundary_provider=None):
    """Read physical continuity, optionally composing caller-supplied boundaries.

    By default, task metadata has no effect. An explicit provider returns a
    sequence of integer label vectors, each with one label per sample. Physical
    continuity is mandatory regardless of the provider.
    """
    paths = ['action_raw_capture_continuity_index'] if raw else list(OBSERVATION_CONTINUITY_PATHS)
    if raw and 'action_raw_capture_continuity_index' not in root:
        for path in OBSERVATION_CONTINUITY_PATHS:
            if path in root and np.any(np.diff(np.asarray(root[path])) != 0):
                raise ValueError('Raw actions need action_raw_capture_continuity_index when observations contain gaps')
    labels = []
    for path in paths:
        if path in root:
            value = np.asarray(root[path])
            if value.shape != (count,):
                raise ValueError(f'{path} must have one boundary label per sample')
            labels.append(value)
    if boundary_provider is not None:
        for value in boundary_provider(root, count, raw=raw):
            value = np.asarray(value)
            if value.shape != (count,):
                raise ValueError('Optional boundaries must have one label per sample')
            labels.append(value)
    return contiguous_runs(np.arange(count), *labels)


def run_bounds(runs, index):
    """Half-open interval containing index; run IDs must be nondecreasing."""
    return (int(np.searchsorted(runs, runs[index], side='left')),
            int(np.searchsorted(runs, runs[index], side='right')))
