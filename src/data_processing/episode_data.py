"""Offline episode access shared by training and visualization.

Raw control arrays and exact camera references are authoritative. The mapping
interface keeps the established dataset/preprocessing code storage-independent;
Camera frames are decoded lazily from their saved references.
"""
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
import zarr

from contracts.action_contract import require_absolute_actions
from contracts.action_rate_contract import resolve_rate
from camera_utils.camera_timing import CameraTimingError, selected_frame_indices
from data_processing.dagger_review import retained_indices
from data_processing.video_processing import ExactVideo
from contracts.sequence_contract import contiguous_runs


def read_json(path):
    return json.loads(Path(path).read_text())


def is_episode(path):
    return (Path(path) / 'control.zarr').is_dir() or (Path(path) / 'episode.json').is_file()


def resolve_episode_directory(directory):
    directory = Path(directory).expanduser().resolve()
    if is_episode(directory):
        return directory
    candidates = sorted(p.parent for p in directory.rglob('control.zarr') if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(f'Expected one raw segment in {directory}; found {len(candidates)}. '
                         'Pass the specific raw stream directory with --data-dir.')
    return candidates[0]


def find_episodes(directory, skip_mirrored_data=False):
    """Find complete, retained Zarr/video episodes."""
    root = Path(directory).expanduser()
    episodes = []
    sources = set()
    for parent, dirs, files in os.walk(root):
        has_control = 'control.zarr' in dirs
        published = 'episode.json' in files
        if has_control or published:
            dirs[:] = []
            path = Path(parent)
            if published:
                sources.update(str((path / part['directory']).resolve())
                               for part in read_json(path / 'episode.json')['sources'])
            manifest = read_json(path / 'manifest.json')
            review = read_json(path / 'review.json') if (path / 'review.json').exists() else {}
            requested_views = any(value.get('requested') for value in manifest.get('outputs', {}).values() if isinstance(value, dict))
            if (not (has_control and requested_views) and manifest.get('complete') is True and manifest.get('retained') is not False
                    and not review.get('discarded') and (not skip_mirrored_data or 'mirror' not in str(path))):
                episodes.append(str(path))
        else:
            dirs[:] = [name for name in dirs if name not in ('discard', 'discarded')
                       and not name.startswith('.') and (not name.endswith('.zarr') or is_episode(Path(parent) / name))]
    # Accepted views can live beside their source streams. Discover each sample
    # through the accepted view, rather than training on both representations.
    return sorted(path for path in episodes if str(Path(path).resolve()) not in sources)


class Group(Mapping):
    def __init__(self, fields, prefix=''):
        self.fields, self.prefix = fields, prefix
        if not hasattr(self, 'attrs'):
            self.attrs = {}

    def __getitem__(self, key):
        key = self.prefix + str(key).strip('/')
        if key in self.fields:
            return self.fields[key]
        if any(name.startswith(key + '/') for name in self.fields):
            return Group(self.fields, key + '/')
        raise KeyError(key)

    def __iter__(self):
        return iter(dict.fromkeys(name[len(self.prefix):].split('/')[0] for name in self.fields if name.startswith(self.prefix)))

    def __len__(self):
        return sum(1 for _ in self)


class ReferencedFrames:
    def __init__(self, path, indices, *, transform=None, cache_bytes=16 * 1024**2):
        self.indices = indices
        self.video = ExactVideo(path, cache_bytes=cache_bytes)
        self.transform = transform
        self.image = None
        self.attrs = dict(color_space='BGR', encoding='video_reference')

    @property
    def shape(self):
        return (len(self), *self[0].shape)

    @property
    def dtype(self):
        return np.dtype('uint8')

    @property
    def capture(self):
        return self.video.container

    @property
    def position(self):
        return self.video.position

    def __len__(self):
        return len(self.indices)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if not index:
                index = slice(None)
            else:
                row, *rest = index
                value = self[row]
                return value[(slice(None), *rest)] if isinstance(row, slice) else value[tuple(rest)]
        if isinstance(index, slice):
            return np.stack([self[i] for i in range(*index.indices(len(self)))])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        image = self.video.get(self.indices[index])
        self.image = self.transform(image) if self.transform is not None else image
        return self.image.copy()

    def close(self):
        self.video.close()
        self.image = None


class SelectedFrames:
    """A lazy selection across source videos, sharing their exact frame readers."""
    def __init__(self, parts):
        self.parts = parts
        self.attrs = dict(color_space='BGR', encoding='video_reference')
        self.ends = np.cumsum([len(indices) for _, indices in parts])

    def __len__(self):
        return int(self.ends[-1])

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    @property
    def shape(self):
        return (len(self), *self[0].shape)

    @property
    def dtype(self):
        return np.dtype('uint8')

    def __getitem__(self, index):
        if isinstance(index, tuple) and not index:
            index = slice(None)
        if isinstance(index, tuple):
            frame, *rest = index
            value = self[frame]
            return value[tuple(rest)] if rest else value
        if isinstance(index, slice):
            return np.stack([self[i] for i in range(*index.indices(len(self)))])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = int(np.searchsorted(self.ends, index, side='right'))
        start = 0 if part == 0 else self.ends[part-1]
        frames, indices = self.parts[part]
        return frames[int(indices[index-start])]

    def close(self):
        pass  # Source Episode owns each decoder.


class Episode(Group):
    def __init__(self, directory, *, inspection=False, transform=None):
        self.directory = resolve_episode_directory(directory)
        self.filename = str(self.directory)
        self.manifest = manifest = read_json(self.directory / 'manifest.json')
        if (self.directory / 'episode.json').is_file():
            self._load_published(inspection=inspection, transform=transform)
            return
        report = read_json(self.directory / 'alignment_report.json')
        task_path = self.directory / 'task_manifest.json'
        task = read_json(task_path) if task_path.exists() else {}
        review_path = self.directory / 'review.json'
        review = read_json(review_path) if review_path.exists() else {}
        if (manifest.get('fatal_error') or manifest.get('alignable') is not True
                or report.get('alignable') is not True or report.get('errors')):
            raise ValueError(f'Raw episode failed alignment checks: {directory}')
        if not inspection and (manifest.get('complete') is not True
                or manifest.get('retained') is False or review.get('discarded')):
            raise ValueError(f'Training requires a complete, retained raw episode: {directory}')
        task_metadata_path = self.directory / 'task_metadata.json'
        task_metadata = read_json(task_metadata_path).get('metadata', {}) if task_metadata_path.exists() else {}
        self.control = control = zarr.open_group(str(self.directory / 'control.zarr'), mode='r')
        require_absolute_actions(manifest, manifest.get('metadata', {}), control.attrs)
        count = len(control['timestamps'])
        if manifest.get('sample_count') != count:
            raise ValueError('Raw sample count differs from manifest')
        stamps = np.asarray(control['timestamps'], dtype=np.float64)
        if stamps.shape != (count,) or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError('Raw robot timestamps must increase strictly')
        keep = np.arange(count, dtype=np.int64)
        edits_path = self.directory / 'edits.json'
        if edits_path.exists():
            keep = np.asarray(read_json(edits_path)['retained_robot_indices'])
            if (keep.ndim != 1 or keep.dtype.kind not in 'iu' or np.any(keep < 0)
                    or np.any(keep >= count) or np.any(np.diff(keep.astype(np.int64)) <= 0)
                    or report.get('retained_robot_samples', len(keep)) != len(keep)):
                raise ValueError('Invalid retained raw observation indices')
        if review:
            keep = np.intersect1d(keep, retained_indices(control, review))
        if not len(keep):
            raise ValueError('Raw episode has no retained observations')
        self.retained_indices = keep
        continuity = np.asarray(control['continuity_id'])
        if continuity.shape != (count,):
            raise ValueError('Invalid raw continuity labels')
        self.runs = contiguous_runs(keep, continuity)
        arrays = {}
        for name in ('qpos', 'qvel', 'commanded_action'):
            value = np.asarray(control[name])
            if len(value) != count or not np.isfinite(value[keep]).all() or value.ndim != 2:
                raise ValueError(f'Invalid raw {name} samples')
            arrays[name] = value[keep]
        self.qpos, self.qvel, self.actions = (arrays[k] for k in ('qpos', 'qvel', 'commanded_action'))
        if self.qpos.shape != self.qvel.shape:
            raise ValueError('Raw position/velocity shapes differ')
        # Keep action labeling explicit: the recorded commanded_action is used
        # without an implicit next-row shift. Task-specific expert labels belong
        # in their own adapter, not in generic observation reconstruction.
        elapsed = np.asarray(control['elapsed_seconds'] if 'elapsed_seconds' in control else stamps)
        times = elapsed[keep]
        if times.shape != (len(keep),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Raw observation timestamps must increase strictly')
        self.timestamps = times - times[0]
        self.fps = resolve_rate(manifest, context=self.filename)
        self.attrs = {**manifest.get('metadata', {}), **task_metadata, **dict(control.attrs), 'fps': self.fps,
            'action_rate_hz': self.fps, 'raw_action_rate_hz': self.fps, 'compress': False,
            'camera_alignment_rule': 'selected_consumer_frame_ids', 'recording_data_directory': self.filename}
        fields = {'observations/qpos': self.qpos, 'observations/qvel': self.qvel,
                  'action': self.actions, 'action_raw': self.actions,
                  'action_timestamps': stamps[keep], 'action_raw_timestamps': stamps[keep],
                  'observation_timestamps': stamps[keep], 'timestamps': self.timestamps,
                  'continuity_id': self.runs, 'action_raw_capture_continuity_index': self.runs}
        fields['recording_timing/robot_timestamp'] = stamps[keep]
        fields['recording_timing/robot_index'] = keep
        fields['recording_timing/timestamps'] = times
        for source, target in (('capture_interval', 'interval_seconds'),
                               ('robot_read_started_at', 'robot_read_started_at'),
                               ('sampled_at', 'sampled_at')):
            if source in control:
                fields[f'recording_timing/{target}'] = np.asarray(control[source])[keep]
        self.images = {}
        self._executor = None
        names = manifest.get('camera_names')
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError('Raw manifest camera names must be a unique list')
        cache_bytes = int(os.environ.get('FLASH_ALOHA_FRAME_CACHE_MB', '64')) * 1024**2 // max(1, len(names))
        for camera in names:
            try:
                indices = selected_frame_indices(self.directory, camera, control, sample_indices=keep)
            except CameraTimingError as exc:
                raise ValueError(str(exc)) from exc
            if indices is None:
                raise ValueError(f'{camera}: exact camera frame references are required')
            video = self.directory / f'{camera}.zarr/main.mp4'
            if not video.is_file():
                raise ValueError(f'Missing raw camera video: {video}')
            frames = ReferencedFrames(video, indices, transform=transform, cache_bytes=cache_bytes)
            self.images[camera] = frames
            fields[f'observations/images/{camera}'] = frames
            fields[f'observations/image_timestamps/{camera}'] = np.asarray(control[f'camera_observation/{camera}/capture_timestamp'])[keep]
            for key in ('frame_number', 'capture_timestamp', 'consumer_received_timestamp'):
                fields[f'camera_observation/{camera}/{key}'] = np.asarray(control[f'camera_observation/{camera}/{key}'])[keep]
        if 'camera_observation/snapshot_timestamp' in control:
            fields['camera_observation/snapshot_timestamp'] = np.asarray(control['camera_observation/snapshot_timestamp'])[keep]
        # Optional declared labels remain generic; no subtask interpretation.
        for attribute in ('policy_training_mask_path', 'training_run_id_path', 'raw_training_run_id_path'):
            path = self.attrs.get(attribute)
            if path:
                # Task boundaries are checked only when opted in. Policy masks
                # remain mandatory even in generic training.
                if path not in control and attribute != 'policy_training_mask_path':
                    continue
                value = np.asarray(control[path])
                if value.shape != (count,):
                    raise ValueError(f'Invalid declared sample labels: {path}')
                fields[path.strip('/')] = value[keep]
        self._load_annotations(fields)
        super().__init__(fields)

    def _load_annotations(self, fields):
        if (self.directory / '.zgroup').exists():
            labels = zarr.open_group(str(self.directory), mode='r')
            self.attrs.update(dict(labels.attrs))
            require_absolute_actions(self.attrs)
            def visit(name, value):
                from data_processing.episode_store import Array, Store
                if isinstance(value, zarr.Array):
                    fields[name] = Array(value)
                    if name.startswith('observations/images/'):
                        camera = name.removeprefix('observations/images/')
                        previous = self.images.get(camera)
                        if hasattr(previous, 'close'):
                            previous.close()
                        self.images[camera] = fields[name]
                elif isinstance(value, zarr.Group):
                    fields[name] = Store(self.filename, 'r', _group=value, _base=Group(fields, name+'/'))
            labels.visititems(visit)
            if 'action' in fields:
                self.actions = np.asarray(fields['action'])
            if 'continuity_id' in fields:
                self.runs = np.asarray(fields['continuity_id'])
            if 'timestamps' in fields:
                self.timestamps = np.asarray(fields['timestamps'])

    def _load_published(self, *, inspection, transform):
        if not inspection and (self.manifest.get('complete') is not True or self.manifest.get('retained') is False):
            raise ValueError('Training requires a complete retained episode')
        specification = read_json(self.directory / 'episode.json')
        if specification.get('schema_version') != 1:
            raise ValueError('Unsupported episode reference schema')
        self._sources, selections = [], []
        self.images, self._executor = {}, None
        fields, arrays, control_arrays, metadata_segments = {}, {}, {}, []
        group_attrs, static_fields = {}, {}
        self.attrs = dict(self.manifest.get('metadata', {}))
        self.fps = resolve_rate(self.manifest, context=self.filename)
        previous_run = -1
        for entry in specification['sources']:
            source = Episode(self.directory / entry['directory'], inspection=True, transform=transform)
            self._sources.append(source)
            if len(self._sources) == 1:
                self.attrs = {**source.attrs, **self.attrs}
            resolve_rate(source.manifest, self.fps, context='Episode source FPS')
            indices = np.asarray(entry['indices'], dtype=np.int64)
            positions = np.searchsorted(source.retained_indices, indices)
            if np.any(positions >= len(source.retained_indices)) or not np.array_equal(source.retained_indices[positions], indices):
                raise ValueError('Published episode references unavailable/discarded observations')
            selections.append((source, positions))
            for key, value in source.fields.items():
                if key.startswith('observations/images/'):
                    continue
                if isinstance(value, Mapping):
                    group_attrs.setdefault(key, dict(getattr(value, 'attrs', {})))
                    continue
                value = np.asarray(value)
                if value.ndim and len(value) == len(source.retained_indices):
                    arrays.setdefault(key, []).append(value[positions])
                else:
                    static_fields.setdefault(key, value)
            runs = contiguous_runs(indices, np.asarray(source.control['continuity_id']))
            arrays.setdefault('_runs', []).append(runs + previous_run + 1)
            previous_run += int(runs[-1]) + 1
            for key in ('metadata_id', 'continuity_id', 'segment_id', 'intervention_active'):
                if key in source.control:
                    value = np.asarray(source.control[key])[indices]
                    if key == 'metadata_id':
                        value = np.where(value >= 0, value + len(metadata_segments), value)
                    control_arrays.setdefault(key, []).append(value)
            metadata_segments.extend(source.manifest.get('metadata_segments', []))
        for key, values in arrays.items():
            if len(values) == len(selections):
                fields[key] = np.concatenate(values)
        fields.update(static_fields)
        self.runs = fields.pop('_runs')
        fields['continuity_id'] = self.runs
        fields['action_raw_capture_continuity_index'] = self.runs
        self.qpos, self.qvel, self.actions = (fields[k] for k in ('observations/qpos', 'observations/qvel', 'action'))
        self.retained_indices = np.arange(len(self.qpos))
        self.timestamps = fields['observation_timestamps'] - fields['observation_timestamps'][0]
        fields['timestamps'] = self.timestamps
        self.manifest['metadata_segments'] = metadata_segments
        self.control = {key: np.concatenate(values) for key, values in control_arrays.items() if len(values) == len(selections)}
        self.control['continuity_id'] = self.runs
        for name in self.manifest['camera_names']:
            frames = SelectedFrames([(source.images[name], positions) for source, positions in selections])
            self.images[name] = frames
            fields[f'observations/images/{name}'] = frames
        self.attrs.update(fps=self.fps, action_rate_hz=self.fps, compress=False)
        for name, attrs in group_attrs.items():
            fields[name] = group = Group(fields, name + '/')
            group.attrs = attrs
        self._load_annotations(fields)
        Group.__init__(self, fields)

    def frame_bundle(self, index, camera_names=None):
        """Decode independent cameras concurrently, preserving each saved reference."""
        names = list(self.images) if camera_names is None else list(camera_names)
        if len(names) < 2:
            return {name: self.images[name][index] for name in names}
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=len(self.images))
        return dict(zip(names, self._executor.map(lambda name: self.images[name][index], names)))

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for frames in self.images.values():
            if hasattr(frames, 'close'):
                frames.close()
        for source in getattr(self, '_sources', []):
            source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_cache, _pid = OrderedDict(), os.getpid()


def clear_episode_cache():
    for _, episode in _cache.values():
        episode.close()
    _cache.clear()


def _fingerprint(path):
    names = ('manifest.json', 'alignment_report.json', 'edits.json', 'review.json',
             'task_metadata.json', 'task_manifest.json', 'collection_labels.jsonl',
             'episode.json', '.zattrs', '.annotation_revision')
    result = tuple((name, (path/name).stat().st_mtime_ns) for name in names if (path/name).exists())
    if (path/'episode.json').exists():
        result += tuple((entry['directory'], _fingerprint(path/entry['directory']))
                        for entry in read_json(path/'episode.json')['sources'])
    return result


@contextmanager
def open_episode(path):
    global _pid
    path = Path(path).expanduser()
    if not is_episode(path):
        raise ValueError(f'Expected a Zarr/video episode directory: {path}')
    if _pid != os.getpid():
        clear_episode_cache()
        _pid = os.getpid()
    key, signature = str(path.resolve()), _fingerprint(path)
    capacity = int(os.environ.get('FLASH_ALOHA_EPISODE_CACHE_SIZE', '4'))
    if capacity < 1:
        raise ValueError('FLASH_ALOHA_EPISODE_CACHE_SIZE must be positive')
    cached = _cache.get(key)
    if cached is None or cached[0] != signature:
        if cached is not None:
            cached[1].close()
        _cache[key] = (signature, Episode(path))
    _cache.move_to_end(key)
    while len(_cache) > capacity:
        _, (_, old) = _cache.popitem(last=False)
        old.close()
    yield _cache[key][1]

# Dataset construction reads a debug sample in the parent. Never inherit an
# active decoder or camera thread pool when PyTorch forks its data workers.
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=clear_episode_cache)
