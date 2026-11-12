"""Shared episode indexing, atomic metadata writes, and recording diagnostics.

These helpers do not capture observations, control hardware, or encode video.
Capture and policy invariants remain in the contracts package.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def get_auto_index(dataset_dir: str, dataset_name_prefix: str = "", data_suffix: str = "zarr") -> int:
    max_idx = 1000
    os.makedirs(dataset_dir, exist_ok=True)
    for i in range(max_idx + 1):
        name = f"{dataset_name_prefix}episode_{i:05d}.{data_suffix}"
        if not os.path.exists(os.path.join(dataset_dir, name)):
            return i
    raise RuntimeError(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history: list[tuple[float, float]]) -> float:
    if not actual_dt_history:
        return 0.0
    actual_dt = np.asarray(actual_dt_history, dtype=np.float64)
    total_time = actual_dt[:, 1] - actual_dt[:, 0]
    mean_dt = float(np.mean(total_time))
    freq_mean = 1.0 / mean_dt if mean_dt > 0 else 0.0
    print(f"Avg freq: {freq_mean:.2f} Hz, average sample time: {mean_dt:.4f}s")
    return freq_mean


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def summarize_recording_rate(nominal_hz, intervals, *, sample_count):
    nominal_hz = float(nominal_hz) if nominal_hz is not None else None
    if nominal_hz is not None and (not math.isfinite(nominal_hz) or nominal_hz <= 0):
        raise ValueError('Nominal recording frequency must be finite and positive')
    intervals = np.asarray(intervals, dtype=np.float64)
    valid = intervals[np.isfinite(intervals) & (intervals > 0)]
    measured = float(len(valid) / valid.sum()) if len(valid) else None
    return dict(
        schema_version=1,
        nominal_recording_hz=nominal_hz,
        measured_recording_hz=measured,
        median_recording_hz=float(1 / np.median(valid)) if len(valid) else None,
        sample_count=int(sample_count),
        measured_interval_count=int(len(valid)),
        measured_duration_seconds=float(valid.sum()),
        relative_error=(measured / nominal_hz - 1) if measured is not None and nominal_hz else None,
        measurement='positive capture intervals; pauses, resume boundaries and forced samples excluded',
    )


def read_recording_rate(root):
    """Read a new recording or explicitly mark a legacy estimate as such."""
    nominal = root.attrs.get('nominal_recording_hz', root.attrs.get('fps'))
    count = len(root['observations/qpos'])
    if 'recording_timing/interval_seconds' in root:
        return summarize_recording_rate(nominal, root['recording_timing/interval_seconds'][:], sample_count=count)
    # Older recordings' timestamps remove operator/encoding pauses, but contain
    # synthetic resume gaps and forced samples. They are only an approximation.
    intervals = np.diff(root['timestamps'][:]) if 'timestamps' in root else []
    report = summarize_recording_rate(nominal, intervals, sample_count=count)
    report['measurement'] = ('legacy trajectory timestamp estimate; resume/forced samples not identifiable'
                             if 'timestamps' in root else 'unavailable: recording has no capture timing')
    return report


def save_recording_rate(root):
    report = read_recording_rate(root)
    root.attrs['recording_rate_json'] = json.dumps(report, allow_nan=False)
    if report['nominal_recording_hz'] is not None:
        root.attrs['nominal_recording_hz'] = report['nominal_recording_hz']
    # NaN explicitly denotes an unavailable measurement (e.g. one sample).
    root.attrs['measured_recording_hz'] = report['measured_recording_hz'] or float('nan')
    return report

