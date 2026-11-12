"""Training-only store of deterministically preprocessed frames.

Random access into long-GOP raw video decodes ~100 frames per sample. The store
decodes each episode once, sequentially, through the same exact-frame reader and
the same shared deterministic preprocessing as the live path
(``prepare_policy_frame``), and saves the uint8 results as memory-mapped arrays.
Rows are keyed by the dataset's own sample indices, so frame references,
alignment and fail-closed checks are those of the reader; raw recordings are
never modified. Entries are keyed by the episode fingerprint and preprocessing
parameters and rebuilt when either changes.
"""
import fcntl
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

STORE_VERSION = 1


def default_frame_store_dir():
    return Path(os.environ.get('FLASH_ALOHA_FRAME_STORE_DIR', '~/.cache/flash-aloha/frame-store')).expanduser()


def _entry_key(path, camera, indices, width, height):
    from data_processing.episode_data import _fingerprint
    from contracts import standard_image_processing
    resolved = Path(path).expanduser().resolve()
    identity = [STORE_VERSION, str(resolved), repr(_fingerprint(resolved)), camera, width, height,
                getattr(standard_image_processing, '__file__', ''),
                hashlib.sha256(np.ascontiguousarray(indices, np.int64).tobytes()).hexdigest()]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _build_entry(path, camera, indices, width, height, target):
    """Decode one camera of one episode in ascending order into ``target``."""
    from data_processing.episode_data import open_episode
    from policy.training_utils import prepare_policy_frame
    target = Path(target)
    with open(str(target) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return str(target)
        with open_episode(path) as root:
            compressed = root.attrs.get('compress', False)
            frames = root[f'/observations/images/{camera}']
            first = prepare_policy_frame(frames[int(indices[0])], compressed, width, height)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.npy', delete=False) as handle:
                temporary = Path(handle.name)
            try:
                out = np.lib.format.open_memmap(temporary, mode='w+', dtype=first.dtype,
                                                shape=(len(indices), *first.shape))
                out[0] = first
                for row, ts in enumerate(indices[1:], start=1):
                    out[row] = prepare_policy_frame(frames[int(ts)], compressed, width, height)
                out.flush()
                del out
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return str(target)


class FrameStore:
    """Read-only view of built entries; pickles as paths so workers share pages."""

    def __init__(self, entries, start_indices):
        self.entries = entries  # {(episode_id, camera): path}
        self.start_indices = {int(k): np.asarray(v, np.int64) for k, v in start_indices.items()}
        self._arrays = {}

    def __getstate__(self):
        return {'entries': self.entries, 'start_indices': self.start_indices}

    def __setstate__(self, state):
        self.__init__(state['entries'], state['start_indices'])

    def get(self, episode_id, camera, ts):
        key = (int(episode_id), camera)
        array = self._arrays.get(key)
        if array is None:
            array = self._arrays[key] = np.load(self.entries[key], mmap_mode='r')
        indices = self.start_indices[int(episode_id)]
        row = int(np.searchsorted(indices, ts))
        if row >= len(indices) or indices[row] != ts:
            raise KeyError(f'Frame store has no row for episode {episode_id} step {ts}')
        return np.array(array[row])


def build_frame_store(dataset_path_list, start_indices, camera_names, width, height, *,
                      directory=None, workers=None):
    directory = Path(directory) if directory else default_frame_store_dir()
    directory.mkdir(parents=True, exist_ok=True)
    jobs, entries = [], {}
    for episode_id, indices in start_indices.items():
        if len(indices) == 0:
            continue
        path = dataset_path_list[int(episode_id)]
        for camera in camera_names:
            target = directory / (_entry_key(path, camera, indices, width, height) + '.npy')
            entries[(int(episode_id), camera)] = str(target)
            if not target.exists():
                jobs.append((path, camera, np.asarray(indices, np.int64), width, height, str(target)))
    if jobs:
        workers = max(1, min(len(jobs), workers or os.cpu_count() or 1))
        print(f'Building frame store: {len(jobs)} episode-camera entries with {workers} processes in {directory}')
        if workers == 1:
            for job in jobs:
                _build_entry(*job)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
                for future in [pool.submit(_build_entry, *job) for job in jobs]:
                    future.result()
    return FrameStore(entries, start_indices)
