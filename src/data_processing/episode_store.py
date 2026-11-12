"""Native episode publication and Zarr annotations.

A published episode selects recorded observations from completed stream directories.
Video and control data stay in those directories; publication never decodes images.
Task labels and inference diagnostics are ordinary Zarr arrays beside episode.json.
"""
from collections.abc import Mapping, MutableMapping
from functools import wraps
from pathlib import Path
import json
import os
import shutil
import uuid

import numpy as np
import zarr

Group = Mapping
Dataset = (zarr.Array, np.ndarray)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


class Attributes(MutableMapping):
    def __init__(self, group, inherited=None):
        self.group, self.inherited = group, inherited or {}

    def __getitem__(self, key):
        if key in self.group.attrs:
            return self.group.attrs[key]
        return self.inherited[key]

    def __setitem__(self, key, value):
        self.group.attrs[key] = json_value(value)

    def __delitem__(self, key):
        del self.group.attrs[key]

    def __iter__(self):
        return iter(dict(self.inherited, **dict(self.group.attrs)))

    def __len__(self):
        return len(list(iter(self)))


class Array:
    """Zarr array with NumPy conversion and explicit string access."""
    def __init__(self, array):
        self.array = array

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value

    def asstr(self):
        return self

    def resize(self, shape, axis=None):
        if axis is not None:
            dims = list(self.shape)
            dims[axis] = shape
            shape = tuple(dims)
        self.array.resize(shape)


class Store(Mapping):
    """Context-managed Zarr labels, optionally over a referenced stream episode."""
    def __init__(self, path, mode='r', *, _group=None, _base=None):
        self.filename = str(path)
        self.mode = mode
        self._owner = _group is None
        self._episode = None
        self._base = _base
        self._group = _group
        if _group is None:
            path = Path(path).expanduser()
            if path.is_file():
                raise ValueError(f'Episodes must be Zarr/video directories: {path}')
            if mode == 'r' and not path.exists():
                raise FileNotFoundError(path)
            if mode in ('r', 'r+', 'a') and ((path / 'episode.json').exists() or (path / 'control.zarr').exists()):
                from data_processing.episode_data import Episode
                self._episode = Episode(path, inspection=True)
                self._base = self._episode
            if mode == 'r' and not (path / '.zgroup').exists():
                self._group = zarr.group(store=zarr.MemoryStore())
            else:
                self._group = zarr.open_group(str(path), mode=mode)
        self.attrs = Attributes(self._group, getattr(self._base, 'attrs', {}))

    def __getitem__(self, key):
        key = str(key).strip('/')
        if key in self._group:
            value = self._group[key]
            if isinstance(value, zarr.Group):
                base = self._base.get(key) if self._base is not None else None
                return Store(self.filename, self.mode, _group=value, _base=base)
            return Array(value)
        if self._base is not None:
            value = self._base[key]
            if isinstance(value, Mapping):
                group = self._group.require_group(key) if self.mode != 'r' else zarr.group(store=zarr.MemoryStore())
                return Store(self.filename, self.mode, _group=group, _base=value)
            return value
        raise KeyError(key)

    def __iter__(self):
        return iter(dict.fromkeys([*self._group.keys(), *(self._base.keys() if self._base is not None else [])]))

    def __len__(self):
        return sum(1 for _ in self)

    def __setitem__(self, key, value):
        self.create_dataset(key, data=value, overwrite=True)

    def __delitem__(self, key):
        del self._group[str(key).strip('/')]

    def move(self, source, destination):
        self._group.move(source, destination)

    def create_group(self, name):
        return Store(self.filename, self.mode, _group=self._group.create_group(name.strip('/')))

    def require_group(self, name):
        return Store(self.filename, self.mode, _group=self._group.require_group(name.strip('/')))

    def create_dataset(self, name, shape=None, *, data=None, dtype=None, chunks=None, **kwargs):
        # Annotations use Zarr's compression defaults; there is no image export.
        if dtype is str:
            dtype = None if data is not None else '<U4096'
        if data is not None:
            data = np.asarray(data, dtype=dtype)
            if data.dtype.kind == 'O':
                data = data.astype(str)
            dtype = data.dtype
        options = dict(data=data, dtype=dtype)
        if shape is not None:
            options['shape'] = shape
        if chunks is not None:
            options['chunks'] = chunks
        for key in ('fill_value', 'overwrite', 'compressor', 'object_codec'):
            if key in kwargs:
                options[key] = kwargs[key]
        return Array(self._group.create_dataset(name.strip('/'), **options))

    def copy(self, source, destination, name=None):
        value = self[source]
        name = source if name is None else name
        self._copy_value(value, destination, name)

    @staticmethod
    def _copy_value(value, destination, name):
        if isinstance(value, Mapping):
            target = destination.create_group(name)
            target.attrs.update(getattr(value, 'attrs', {}))
            for key in value:
                Store._copy_value(value[key], target, key)
        else:
            target = destination.create_dataset(name, data=np.asarray(value))
            target.attrs.update(dict(getattr(value, 'attrs', {})))

    def flush(self):
        pass  # DirectoryStore writes each array/attribute update synchronously.

    def close(self):
        if self._owner and self.mode != 'r':
            (Path(self.filename) / '.annotation_revision').write_text(uuid.uuid4().hex)
        if self._episode is not None:
            self._episode.close()
            self._episode = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


open_store = Store


def atomic_episode_save(function):
    """Publish a task save only after all of its annotations have succeeded."""
    @wraps(function)
    def save(dataset_path, *args, **kwargs):
        destination = Path(dataset_path)
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.pending')
        try:
            result = function(temporary, *args, **kwargs)
            temporary.rename(destination)
            return result
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return save


def copy_references(source, destination):
    """Carry stream provenance into a derived annotation directory."""
    from data_processing.episode_data import Episode, is_episode
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not is_episode(source):
        return  # Standalone annotation artifacts have no recorded streams.
    if (source / 'episode.json').exists():
        specification = json.loads((source / 'episode.json').read_text())
        for part in specification['sources']:
            part['directory'] = os.path.relpath(source / part['directory'], destination)
    else:
        with Episode(source, inspection=True) as episode:
            specification = dict(schema_version=1, sources=[dict(
                directory=os.path.relpath(source, destination), indices=episode.retained_indices.tolist())])
    (destination / 'episode.json').write_text(json.dumps(specification))
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['sample_count'] = sum(len(part['indices']) for part in specification['sources'])
    (destination / 'manifest.json').write_text(json.dumps(manifest))


def replace_directory(temporary, destination):
    """Replace an explicitly overwrite-enabled artifact, restoring on failure."""
    temporary, destination = Path(temporary), Path(destination)
    backup = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.backup')
    existed = destination.exists()
    if existed:
        destination.rename(backup)
    try:
        temporary.rename(destination)
    except BaseException:
        if existed:
            backup.rename(destination)
        raise
    if existed:
        shutil.rmtree(backup)


def next_episode_path(directory):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while (directory / f'episode_{index:05d}.zarr').exists():
        index += 1
    return directory / f'episode_{index:05d}.zarr'


def publish_episode(path, sources, *, metadata=None, arrays=None, annotate=None):
    """Atomically publish exact source-row selections with task annotations.

    sources is a list of (stream_directory, original_control_row_indices).
    Source streams must be drained and pass their existing alignment report.
    """
    from recording.recording_utils import atomic_write_json
    path = Path(path).expanduser().absolute()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.{path.name}.{uuid.uuid4().hex}.pending'
    temporary.mkdir()
    try:
        selections, rate, cameras, inherited = [], None, None, None
        for directory, indices in sources:
            directory = Path(directory).resolve()
            manifest = json.loads((directory / 'manifest.json').read_text())
            report = json.loads((directory / 'alignment_report.json').read_text())
            if not manifest.get('complete') or manifest.get('alignable') is not True or report.get('alignable') is not True or report.get('errors'):
                raise ValueError(f'Cannot publish incomplete or misaligned streams: {directory}')
            from contracts.action_rate_contract import resolve_rate
            source_rate = resolve_rate(manifest, rate, context=str(directory))
            rate = source_rate
            names = manifest['camera_names']
            if cameras is not None and cameras != names:
                raise ValueError('Episode segments must have identical camera names and ordering')
            cameras = names
            indices = np.asarray(indices)
            count = int(manifest['sample_count'])
            if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or np.any(indices < 0) or np.any(indices >= count) or np.any(np.diff(indices) <= 0):
                raise ValueError('Episode source indices must be nonempty, increasing recorded row indices')
            from data_processing.episode_data import Episode
            with Episode(directory, inspection=True) as source:
                if not np.isin(indices, source.retained_indices).all():
                    raise ValueError('Episode references unavailable/discarded observations')
                if inherited is None:
                    inherited = dict(source.attrs)
            selections.append(dict(directory=os.path.relpath(directory, path), indices=indices.tolist()))
        if not selections:
            raise ValueError('An episode requires at least one recorded stream')
        attrs = inherited
        attrs.update(metadata or {})
        from contracts.action_contract import require_absolute_actions
        require_absolute_actions(attrs)
        attrs.update(fps=rate, action_rate_hz=rate, raw_action_rate_hz=rate, compress=False,
                     camera_names=cameras, camera_alignment_rule='selected_consumer_frame_ids')
        with Store(temporary, 'a') as store:
            store.attrs.update(attrs)
            for name, value in (arrays or {}).items():
                store.create_dataset(name, data=value)
            if annotate is not None:
                annotate(store)
        atomic_write_json(temporary / 'episode.json', dict(schema_version=1, sources=selections))
        atomic_write_json(temporary / 'manifest.json', dict(complete=True, retained=True, alignable=True,
            fps=rate, camera_names=cameras, sample_count=sum(len(s['indices']) for s in selections), metadata=attrs))
        os.rename(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(path)


def publish_samples(path, samples, *, metadata=None, events=None, annotate=None):
    sources = []
    for sample in samples:
        images = sample.images
        buffer = getattr(images, 'buffer', None)
        if buffer is not None:
            buffer.finish()
            directory, index = buffer.data_dir, sample.robot_index
        else:
            directory, index = getattr(sample, 'episode_directory', None), getattr(sample, 'robot_index', None)
        if directory is None or index is None:
            raise ValueError('Episode samples must retain their original stream directory and frame references')
        directory = str(Path(directory).resolve())
        if sources and sources[-1][0] == directory:
            sources[-1][1].append(int(index))
        else:
            sources.append((directory, [int(index)]))
    arrays = {'timestamps': np.asarray([sample.timestamp for sample in samples], dtype=np.float64)}
    if events is not None:
        from dataclasses import asdict
        for name, value in asdict(events).items():
            if value is not None:
                arrays[f'recording_events/{name}'] = np.asarray(value)
    return publish_episode(path, sources, metadata=metadata, arrays=arrays, annotate=annotate)


class SampleImages(Mapping):
    """Lazy checkpoint images with the original recording reference attached."""
    def __init__(self, episode, index):
        self.episode, self.index = episode, index

    def __getitem__(self, name):
        return self.episode.images[name][self.index]

    def __iter__(self):
        return iter(self.episode.images)

    def __len__(self):
        return len(self.episode.images)


def referenced_samples(path, *, timestamps=None):
    """Restore checkpoint samples without decoding or duplicating their video."""
    from types import SimpleNamespace
    from data_processing.episode_data import Episode
    episode = Episode(path, inspection=True)
    samples = []
    times = episode.timestamps if timestamps is None else timestamps
    spec = json.loads((Path(path) / 'episode.json').read_text())
    rows = [(Path(path) / part['directory'], row) for part in spec['sources'] for row in part['indices']]
    for index, (directory, row) in enumerate(rows):
        samples.append(SimpleNamespace(images=SampleImages(episode, index),
            qpos=episode.qpos[index].copy(), qvel=episode.qvel[index].copy(),
            timestamp=float(times[index]), episode_directory=str(directory.resolve()),
            robot_index=int(row), capture_interval=None))
    return samples
