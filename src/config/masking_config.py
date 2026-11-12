"""The masking configuration: calibration, arms, cameras and the strategies themselves.

One YAML file (default ``calibration/masking/masking.yaml``; ``FLASH_ALOHA_MASKING_CONFIG``
selects another) declares everything ``utils.masking.Masker`` needs, so masks are data
rather than code. This module only reads and validates it: no torch, numpy or OpenCV.

    calibration:
      image_size: [320, 180]        # width, height the intrinsics were calibrated at
      urdf: ../../src/hardware/arx5-sdk/models/X5.urdf
      links: {wrist: link6, end_effector: eef_link}
      overhead_camera: {name: video_overhead, intrinsics: <file>}
      arms:                         # in state order; each arm has `joints` joints, then a gripper
        left:  {wrist_camera: video_left, intrinsics: <file>, camera_on_end_effector: <file>,
                base_in_overhead: <file>, mirror_y: true}
        right: {...}
    definitions: {...}              # optional: YAML anchors for operations reused below
    strategies:
      NAME:                         # a list of operations, applied in order
        - project: {camera: overhead | wrist, frame: base | wrist | end_effector,
                    arm: self | other, corners: [[x, y, z], ...]}   # metres
        - region: {camera: overhead | wrist | <camera name>, x: [0, 0.25], y: [0, 1]}  # image fractions
        - blank: wrist | overhead | <camera name>
        - zero: proprioception

``project`` runs once per arm ("self"): the polygon's corners are given in a frame located
at a link of that arm or of the other arm (``arm``), with the axes of that arm's base, and
drawn in the overhead camera or in self's wrist camera. Corners are written for an
unmirrored arm; arms with ``mirror_y: true`` negate y. File paths are relative to the YAML.
"""
import hashlib
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASKING_CONFIG = REPO_ROOT / 'calibration/masking/masking.yaml'
ENVIRONMENT_VARIABLE = 'FLASH_ALOHA_MASKING_CONFIG'
FRAMES = ('base', 'wrist', 'end_effector')
OPERATIONS = ('project', 'region', 'blank', 'zero')


def masking_config_path(path=None):
    return Path(path or os.environ.get(ENVIRONMENT_VARIABLE) or DEFAULT_MASKING_CONFIG).expanduser().resolve()


def _fail(source, message):
    raise ValueError(f'{source}: {message}')


def _keys(source, where, mapping, required, optional=()):
    if not isinstance(mapping, dict):
        _fail(source, f'{where} must be a mapping')
    unknown, missing = set(mapping) - set(required) - set(optional), set(required) - set(mapping)
    if unknown or missing:
        _fail(source, f'{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}')


def _interval(source, where, value):
    if (not isinstance(value, list) or len(value) != 2 or not all(type(v) in (int, float) for v in value)
            or not 0 <= value[0] < value[1] <= 1):
        _fail(source, f'{where} must be [start, stop] image fractions with 0 <= start < stop <= 1')


def _validate_operation(source, where, operation, calibrated):
    if not isinstance(operation, dict) or len(operation) != 1 or next(iter(operation)) not in OPERATIONS:
        _fail(source, f'{where} must be one of {OPERATIONS}')
    kind, body = next(iter(operation.items()))
    if kind == 'project':
        _keys(source, where, body, ('camera', 'frame', 'arm', 'corners'))
        if body['camera'] not in ('overhead', 'wrist') or body['frame'] not in FRAMES or body['arm'] not in ('self', 'other'):
            _fail(source, f'{where}: camera is overhead|wrist, frame is one of {FRAMES}, arm is self|other')
        corners = body['corners']
        if (not isinstance(corners, list) or len(corners) < 3 or any(
                not isinstance(c, list) or len(c) != 3 or not all(type(v) in (int, float) for v in c) for c in corners)):
            _fail(source, f'{where}: corners must be at least three [x, y, z] points in metres')
        if not calibrated:
            _fail(source, f'{where}: projected masks need a calibration section')
    elif kind == 'region':
        _keys(source, where, body, ('camera', 'x', 'y'))
        _interval(source, f'{where}.x', body['x'])
        _interval(source, f'{where}.y', body['y'])
    elif kind == 'blank':
        if not isinstance(body, str):
            _fail(source, f'{where}: blank names wrist, overhead or one camera')
    elif body != 'proprioception':
        _fail(source, f'{where}: zero supports only proprioception')


def load_masking_config(path=None):
    """The validated document, with ``source`` (its path) and absolute calibration file paths."""
    source = masking_config_path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Masking config not found: {source} (set {ENVIRONMENT_VARIABLE} or see '
                                'calibration/masking/README.md)')
    document = yaml.safe_load(source.read_text()) or {}
    _keys(source, 'masking config', document, ('strategies',), ('calibration', 'definitions'))
    calibration = document.get('calibration')
    if calibration is not None:
        _keys(source, 'calibration', calibration, ('image_size', 'urdf', 'links', 'overhead_camera', 'arms'),
              ('joints_per_arm',))
        size = calibration['image_size']
        if not isinstance(size, list) or len(size) != 2 or not all(type(v) is int and v > 0 for v in size):
            _fail(source, 'calibration.image_size must be [width, height] in pixels')
        _keys(source, 'calibration.links', calibration['links'], ('wrist', 'end_effector'))
        _keys(source, 'calibration.overhead_camera', calibration['overhead_camera'], ('name', 'intrinsics'))
        arms = calibration['arms']
        if not isinstance(arms, dict) or not arms:
            _fail(source, 'calibration.arms must name at least one arm')
        for name, arm in arms.items():
            _keys(source, f'calibration.arms.{name}', arm,
                  ('wrist_camera', 'intrinsics', 'camera_on_end_effector', 'base_in_overhead'), ('mirror_y',))
            if type(arm.get('mirror_y', False)) is not bool:
                _fail(source, f'calibration.arms.{name}.mirror_y must be a boolean')
        if type(calibration.get('joints_per_arm', 6)) is not int or calibration.get('joints_per_arm', 6) <= 0:
            _fail(source, 'calibration.joints_per_arm must be a positive integer')

        def resolve(value):
            file = Path(value).expanduser()
            return str(file if file.is_absolute() else (source.parent / file).resolve())
        calibration['urdf'] = resolve(calibration['urdf'])
        calibration['overhead_camera']['intrinsics'] = resolve(calibration['overhead_camera']['intrinsics'])
        for arm in arms.values():
            for key in ('intrinsics', 'camera_on_end_effector', 'base_in_overhead'):
                arm[key] = resolve(arm[key])
    strategies = document['strategies']
    if not isinstance(strategies, dict) or not strategies:
        _fail(source, 'strategies must define at least one named strategy')
    for name, operations in strategies.items():
        if not isinstance(name, str) or not isinstance(operations, list) or not operations:
            _fail(source, f'strategies.{name} must be a non-empty list of operations')
        for index, operation in enumerate(operations):
            _validate_operation(source, f'strategies.{name}[{index}]', operation, calibration is not None)
    document['source'] = str(source)
    return document


def masking_strategy_names(path=None):
    return tuple(load_masking_config(path)['strategies'])


def require_masking_strategy(strategy, path=None):
    """None or a strategy the masking config defines; unknown names fail instead of running unmasked."""
    if strategy is None:
        return None
    names = masking_strategy_names(path)
    if strategy not in names:
        raise ValueError(f'Unknown masking strategy: {strategy!r} (defined in {masking_config_path(path)}: '
                         f'{", ".join(names)})')
    return strategy


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masking_contract(strategy, path=None):
    """What a checkpoint stores about its mask: the operations and, for projected masks, the calibration.

    Deployment rebuilds it from its own masking config and requires equality, so a changed
    strategy definition or calibration file is refused instead of silently changing the inputs.
    """
    document = load_masking_config(path)
    require_masking_strategy(strategy, path)
    operations = document['strategies'][strategy]
    content = {'strategy': strategy, 'operations': operations}
    if any('project' in operation for operation in operations):
        calibration = document['calibration']
        content['calibration'] = {
            'image_size': calibration['image_size'], 'links': calibration['links'],
            'joints_per_arm': calibration.get('joints_per_arm', 6), 'urdf': _file_digest(calibration['urdf']),
            'overhead_camera': {'name': calibration['overhead_camera']['name'],
                                'intrinsics': _file_digest(calibration['overhead_camera']['intrinsics'])},
            'arms': {name: {'wrist_camera': arm['wrist_camera'], 'mirror_y': arm.get('mirror_y', False),
                            **{key: _file_digest(arm[key])
                               for key in ('intrinsics', 'camera_on_end_effector', 'base_in_overhead')}}
                     for name, arm in calibration['arms'].items()}}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return {'strategy': strategy, 'sha256': digest}


def require_masking_contract(saved, strategy, path=None):
    """Fail unless the checkpoint's saved mask is the one this masking config would apply."""
    if strategy is None:
        if saved is not None:
            raise ValueError('Checkpoint stores a masking contract but no masking strategy')
        return None
    current = masking_contract(strategy, path)
    if saved is None:
        raise ValueError(f'Checkpoint was trained with masking strategy {strategy!r} but stores no masking '
                         'contract; retrain so the mask definition and calibration can be checked')
    if dict(saved) != current:
        raise ValueError(f'Masking contract mismatch for strategy {strategy!r}: the masking config or calibration '
                         f'in {masking_config_path(path)} differs from the one the checkpoint was trained with')
    return current
