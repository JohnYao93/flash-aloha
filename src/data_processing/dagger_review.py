"""Reversible raw-row selection for offline DAgger review and training readers.

Never edits observations, timestamps or camera references. Segment IDs are raw
half-open row ranges, stable across discards and restores.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from contracts.sequence_contract import contiguous_runs


def segment_table(control):
    times = np.asarray(control['timestamps'], dtype=np.float64)
    metadata = np.asarray(control['metadata_id'], dtype=np.int64)
    manual = np.asarray(control['intervention_active'], dtype=np.uint8)
    continuity = np.asarray(control['continuity_id'], dtype=np.int64)
    count = len(times)
    if any(v.shape != (count,) for v in (times, metadata, manual, continuity)):
        raise ValueError('Invalid DAgger segment arrays')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(manual > 1):
        raise ValueError('Invalid DAgger timestamps or intervention flags')
    if not count:
        return []
    starts = np.flatnonzero(np.r_[True, (np.diff(metadata) != 0) |
        (np.diff(manual.astype(int)) != 0) | (np.diff(continuity) != 0)])
    return [dict(id=f'{start}:{end}', raw_start=int(start), raw_end_exclusive=int(end),
                 start_seconds=float(times[start]-times[0]),
                 end_seconds=float(times[end]-times[0]) if end<count else float(times[-1]-times[0]),
                 sample_count=int(end-start), metadata_id=int(metadata[start]),
                 continuity_id=int(continuity[start]), manual=bool(manual[start]))
            for start, end in zip(starts, np.r_[starts[1:], count])]


def source_fingerprint(control):
    digest = hashlib.sha256()
    for name in ('timestamps', 'metadata_id', 'intervention_active', 'continuity_id'):
        value = np.asarray(control[name], dtype='<f8' if name=='timestamps' else '<i8')
        digest.update(name.encode());digest.update(value.tobytes())
    return digest.hexdigest()


def retained_indices(control, review):
    """Exact raw observation indices; no joining, timestamp repair or resampling."""
    count = len(control['timestamps'])
    keep = np.ones(count, dtype=bool)
    excluded = review.get('discarded_segments', [])
    if not isinstance(excluded, list) or any(not isinstance(s,str) for s in excluded) or len(excluded)!=len(set(excluded)):
        raise ValueError('Invalid discarded segment list')
    if excluded:
        if review.get('source_fingerprint') != source_fingerprint(control):
            raise ValueError('Recording changed after segment review; refusing stale exclusions')
        segments = {s['id']:s for s in segment_table(control)}
        for sid in excluded:
            if sid not in segments:
                raise ValueError('Discarded segment does not match a recorded boundary')
            s=segments[sid];keep[s['raw_start']:s['raw_end_exclusive']]=False
    if review.get('discarded'):
        keep[:]=False
    return np.flatnonzero(keep)


def load_selection(directory, control=None):
    """Training-reader entry point; keep raw_indices and split chunks by run_id.

    A one-step target r+1 is allowed only if both rows are retained and belong
    to the same recorded continuity. Camera lookup still uses original rows.
    """
    import zarr
    directory=Path(directory)
    if control is None:control=zarr.open_group(str(directory/'control.zarr'),mode='r')
    path=directory/'review.json'
    review=json.loads(path.read_text()) if path.exists() else {}
    indices=retained_indices(control,review)
    continuity=np.asarray(control['continuity_id'])
    runs=contiguous_runs(indices, continuity)
    transitions=indices[:-1][(np.diff(indices)==1) & (np.diff(continuity[indices])==0)]
    return dict(raw_indices=indices,run_id=runs,transition_indices=transitions)
