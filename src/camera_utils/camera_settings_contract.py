"""Optional saved camera-tuning contract; never opens or changes a camera."""

# Support direct execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "src")
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))
from collections.abc import Mapping
from copy import deepcopy
import json
import math


def normalize_camera_settings(value):
    """Accept JSON Zarr attributes and native checkpoint/YAML mappings."""
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if value is None or (isinstance(value, str) and not value.strip()):
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('camera_settings must be a mapping or empty')
    result = {}
    for camera, record in value.items():
        if not isinstance(camera, str) or not camera:
            raise ValueError('camera_settings camera names must be nonempty strings')
        if not isinstance(record, Mapping) or set(record) - {'settings', 'mode'}:
            raise ValueError(f'Invalid camera_settings entry for {camera}')
        controls = record.get('settings', {})
        if not isinstance(controls, Mapping):
            raise ValueError(f'{camera}.settings must be a mapping')
        entry = {}
        if controls:
            if any(not isinstance(key, str) or type(v) not in (int, float) or not math.isfinite(v)
                   for key, v in controls.items()):
                raise ValueError(f'{camera}.settings must contain finite numeric controls')
            entry['settings'] = dict(controls)
        mode = record.get('mode')
        if mode is not None and mode != []:
            if (not isinstance(mode, (list, tuple)) or len(mode) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in mode)):
                raise ValueError(f'{camera}.mode must be [width, height, fps] with positive finite values')
            entry['mode'] = list(mode)
        if entry:
            result[camera] = entry
    return result


def camera_settings_json(settings):
    return json.dumps(normalize_camera_settings(settings), sort_keys=True, separators=(',', ':'))


def snapshot_camera_settings(camera_names=None, path=None):
    """Read tune_cameras' NPZ using the same resolution and parser as camera_server.

    Paths and device IDs are not part of equality. Missing files/records give an
    empty, unrestricted contract; malformed existing files still report errors.
    """
    if camera_names is not None and not camera_names:
        return {}
    from camera_utils.camera_server import camera_match_keys, load_camera_param_records
    records = load_camera_param_records(path, strict=True)
    if camera_names is None:
        camera_names = [record['name'] if record['name'].startswith('video_')
                        else 'video_' + record['name'] for record in records]
    result = {}
    for name in camera_names:
        keys = camera_match_keys(name, '/dev/' + name)
        matches = [record for record in records
                   if keys & camera_match_keys(record['name'], record['device'])]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous saved camera settings for {name}')
        if matches:
            record = matches[0]
            result[name] = {'settings': record['settings'], 'mode': record['mode']}
    return normalize_camera_settings(result)


def merge_camera_settings(*settings, context='camera settings'):
    """Combine declarations; missing values add no constraint, conflicts fail."""
    result = {}
    for value in settings:
        for camera, record in normalize_camera_settings(value).items():
            target = result.setdefault(camera, {})
            for field, incoming in record.items():
                if field == 'settings':
                    controls = target.setdefault(field, {})
                    for key, val in incoming.items():
                        if key in controls and controls[key] != val:
                            raise ValueError(f'{context}: camera settings mismatch for {camera}.{key}: {controls[key]} != {val}')
                        controls[key] = val
                else:
                    if field in target and target[field] != incoming:
                        raise ValueError(f'{context}: camera settings mismatch for {camera}.{field}: {target[field]} != {incoming}')
                    target[field] = deepcopy(incoming)
    return result


def metadata_camera_settings(metadata):
    # An explicitly empty top-level contract is an intentional opt-out.
    if 'camera_settings' in metadata:
        return normalize_camera_settings(metadata['camera_settings'])
    return merge_camera_settings(*(section.get('camera_settings') for section in
        (metadata.get('policy_config', {}), metadata.get('task_config', {}))
        if isinstance(section, Mapping)))


def check_inference_camera_settings(metadata, *, path=None, context='inference'):
    expected = metadata_camera_settings(metadata)
    if expected:
        actual = snapshot_camera_settings(list(expected), path)
        merge_camera_settings(expected, actual, context=context)
    return expected


def recording_camera_metadata(metadata, camera_names):
    result = dict(metadata or {})
    result['camera_settings'] = (normalize_camera_settings(result['camera_settings'])
        if 'camera_settings' in result else snapshot_camera_settings(camera_names))
    return result


def dataset_camera_settings(paths):
    """Propagate known episode settings; legacy/empty episodes add no constraint."""
    from data_processing.episode_data import open_episode
    settings = {}
    for path in paths:
        with open_episode(path) as episode:
            settings = merge_camera_settings(settings, metadata_camera_settings(episode.attrs), context=str(path))
    return settings
