"""Audit independent streams without decoding video or changing their clocks."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import zarr

from camera_utils.camera_timing import TimingLimits, driver_timestamp, selected_frame_indices


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (dict(min_ms=float(values.min()*1000), median_ms=float(np.median(values)*1000),
                 p99_ms=float(np.quantile(values, .99)*1000), max_ms=float(values.max()*1000))
            if len(values) else None)


def audit_episode_data(directory, *, limits=None):
    """Check exact consumer references, retaining legacy matching diagnostics.

    Camera-ready matching is only an approximation of consumer availability.
    New samples name the selected frames and their consumer publication times.
    Edits and continuity boundaries are honored; no interpolation or endpoint
    clamping hides missing coverage.
    """
    directory = Path(directory)
    limits = limits or TimingLimits.from_env()
    manifest = json.loads((directory / 'manifest.json').read_text())
    control = zarr.open_group(str(directory / 'control.zarr'), mode='r')
    if 'timestamps' not in control:
        return dict(schema_version=1, alignable=False, limits=asdict(limits), retained_robot_samples=0,
                    cameras={}, cross_camera={}, errors=['No robot samples were recorded'])
    t = np.asarray(control['timestamps'], dtype=float)
    continuity = np.asarray(control['continuity_id'], dtype=int)
    errors = []
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        errors.append('Robot timestamps must be nonempty, finite and strictly increasing')
    edit_file = directory / 'edits.json'
    retained = (json.loads(edit_file.read_text())['retained_robot_indices'] if edit_file.exists()
                else list(range(len(t))))
    retained = np.asarray(retained, dtype=int)
    if not len(retained) or np.any(retained < 0) or np.any(retained >= len(t)) or np.any(np.diff(retained) <= 0):
        errors.append('Invalid/empty retained robot indices')
        retained = np.array([], dtype=int)
    t, continuity = t[retained], continuity[retained]
    selected = {'capture': [], 'available': []}
    has_references = 'camera_observation' in control
    version = manifest.get('capture_contract_version')
    if version is not None:
        if version != 1 or control.attrs.get('capture_contract_version') != version:
            errors.append('Missing/unsupported standard capture contract version')
        try:
            started = np.asarray(control['robot_read_started_at'])[retained]
            if (not np.isfinite(started).all() or np.any(started > t)
                    or np.any(t - started > limits.max_age_seconds)):
                errors.append('Invalid/excessive robot state read interval')
        except (KeyError, IndexError) as exc:
            errors.append(f'Missing/incomplete robot read timing: {exc}')
    if manifest.get('observation_alignment') == 'selected_consumer_frame_ids' and not has_references:
        errors.append('Recording requires consumer frame references, but they are missing')
    snapshots = None
    if has_references:
        selected['consumer'] = []
        try:
            snapshots = np.asarray(control['camera_observation/snapshot_timestamp'])[retained]
            if (not np.isfinite(snapshots).all() or np.any(np.diff(snapshots) <= 0)
                    or np.any(snapshots < t) or np.any(snapshots - t > limits.max_age_seconds)):
                errors.append('Invalid observation cutoff or excessive proprio-to-image-snapshot delay')
        except (KeyError, IndexError) as exc:
            errors.append(f'Missing/incomplete observation cutoffs: {exc}')
            snapshots = np.full(len(t), np.nan)
    cameras = {}
    for name in manifest['camera_names']:
        try:
            path = directory / f'{name}.timing.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            capture = np.asarray([r['capture_timestamp'] for r in rows], dtype=float)
            ready = np.asarray([r['frame_ready_timestamp'] for r in rows], dtype=float)
            ids = np.asarray([r['continuity_id'] for r in rows], dtype=int)
            if not len(rows) or [r['video_frame'] for r in rows] != list(range(len(rows))):
                raise ValueError('missing/non-contiguous frame indices')
            group = zarr.open_group(str(directory / f'{name}.zarr'), mode='r')
            if len(group['main_timestamps']) != len(rows):
                raise ValueError('sidecar/video timestamp count mismatch')
            for times in (capture, ready):
                if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError('non-finite/non-increasing camera timestamps')
            for row in rows:
                if row.get('buffer_flags') is None:
                    raise ValueError('driver clock flags were not recorded; clock provenance unverified')
                driver_timestamp(row['capture_timestamp'], row['buffer_flags'])
            latency = ready - capture
            camera_errors = []
            if np.any(latency < -limits.max_future_seconds):
                camera_errors.append('capture clock is ahead of host frame-ready clock')
            if np.any(latency > limits.max_age_seconds):
                camera_errors.append('capture-to-ready delay exceeds age limit')
            info = dict(frames=len(rows), capture_to_ready=_summary(latency), errors=camera_errors)
            for mode, clock in (('capture', capture), ('available', ready)):
                indices = np.searchsorted(clock, t, side='right') - 1
                safe = np.clip(indices, 0, len(rows)-1)
                valid = (indices >= 0) & (ids[safe] == continuity)
                stamps = np.where(valid, capture[safe], np.nan)
                age = t - stamps
                bad = ~valid | (age > limits.max_age_seconds) | (age < -limits.max_future_seconds)
                info[mode] = dict(bad_samples=int(bad.sum()), age=_summary(age))
                selected[mode].append(stamps)
                if bad.any() and not has_references:
                    camera_errors.append(f'{mode}: {bad.sum()} robot samples lack a fresh frame in the same continuity')
            if has_references:
                indices = selected_frame_indices(directory, name, control, sample_indices=retained)
                received = np.asarray(control[f'camera_observation/{name}/consumer_received_timestamp'])[retained]
                age = snapshots - capture[indices]
                bad = (~np.isfinite(received) | (received > snapshots)
                       | (received < ready[indices])
                       | (age > limits.max_age_seconds) | (age < -limits.max_future_seconds))
                info['consumer'] = dict(bad_samples=int(bad.sum()), age=_summary(age))
                selected['consumer'].append(capture[indices])
                if bad.any():
                    camera_errors.append(f'consumer: {bad.sum()} samples have invalid publication times or stale referenced frames')
            cameras[name] = info
            errors.extend(f'{name}: {error}' for error in camera_errors)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            errors.append(f'{name}: {exc}')
            cameras[name] = dict(errors=[str(exc)])
    skew = {}
    for mode, values in selected.items():
        if len(values) == len(manifest['camera_names']) and values and len(t):
            spread = np.ptp(np.stack(values), axis=0)
            bad = ~np.isfinite(spread) | (spread > limits.max_skew_seconds)
            skew[mode] = dict(bad_samples=int(bad.sum()), spread=_summary(spread))
            if bad.any() and (not has_references or mode == 'consumer'):
                errors.append(f'{mode}: {bad.sum()} samples have excessive/undefined cross-camera skew')
    return dict(schema_version=1, alignable=not errors, limits=asdict(limits),
                alignment_rule='selected_consumer_frame_ids' if has_references else 'legacy_timestamp_matching',
                retained_robot_samples=len(t), cameras=cameras, cross_camera=skew, errors=errors)
