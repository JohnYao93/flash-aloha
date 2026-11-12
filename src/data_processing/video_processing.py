"""Exact ordinal video decoding with bounded caches and timestamp-verified seeks.

Video PTS are used only to seek an already-selected encoded frame. They never
select which camera observation belongs to a robot row.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import av
import numpy as np


class ExactVideo:
    def __init__(self, path, *, cache_bytes=16 * 1024**2):
        self.path = Path(path)
        self.cache_bytes = int(cache_bytes)
        self.cache, self.bytes = OrderedDict(), 0
        self.container = self.iterator = None
        self.position = -1
        self.pts = None
        self.decoded_frames = 0

    def _open(self):
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 1
        self.iterator = iter(self.container.decode(self.stream))
        self.position = -1

    def _index(self):
        if self.pts is not None:
            return
        stat = self.path.stat()
        identity = json.dumps([str(self.path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino, 1])
        key = hashlib.sha256(identity.encode()).hexdigest()
        directory = Path(os.environ.get('FLASH_ALOHA_VIDEO_INDEX_CACHE', '~/.cache/flash-aloha/video-index')).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (key + '.npy')
        # Several data workers can reach a video at once. Build its tiny index
        # only once; no decoded image cache is written to disk.
        import fcntl
        with (directory / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with av.open(str(self.path)) as source:
                    stream = source.streams.video[0]
                    stream.codec_context.thread_count = 1
                    stamps = [frame.pts for frame in source.decode(stream)]
                if not stamps or any(value is None for value in stamps):
                    raise ValueError(f'Video has missing presentation timestamps: {self.path}')
                values = np.asarray(stamps, np.int64)
                if np.any(np.diff(values) <= 0):
                    raise ValueError(f'Video presentation timestamps are not unique and increasing: {self.path}')
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    try:
                        np.save(temporary, values, allow_pickle=False)
                        temporary.flush()
                        os.replace(temporary_path, target)
                    finally:
                        temporary_path.unlink(missing_ok=True)
            self.pts = np.load(target, allow_pickle=False, mmap_mode='r')

    def get(self, index):
        index = int(index)
        if index < 0:
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index].copy()
        try:
            if self.container is None:
                self._open()
            seeking = index <= self.position or index - self.position > 64
            if seeking:
                self._index()
                if index >= len(self.pts):
                    raise IndexError(index)
                wanted_pts = int(self.pts[index])
                self.container.seek(wanted_pts, stream=self.stream, backward=True, any_frame=False)
                self.iterator = iter(self.container.decode(self.stream))
            for frame in self.iterator:
                self.decoded_frames += 1
                if seeking:
                    ordinal = int(np.searchsorted(self.pts, frame.pts))
                    if ordinal >= len(self.pts) or int(self.pts[ordinal]) != frame.pts:
                        raise ValueError(f'Video seek returned an unindexed frame: {self.path}')
                    self.position = ordinal
                else:
                    self.position += 1
                if self.position > index:
                    raise ValueError(f'Video seek skipped referenced frame {index}: {self.path}')
                if self.position == index:
                    image = frame.to_ndarray(format='bgr24')
                    self.cache[index] = image
                    self.bytes += image.nbytes
                    while self.bytes > self.cache_bytes and self.cache:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old.nbytes
                    return image.copy()
            raise IndexError(index)
        except (IndexError, av.error.FFmpegError) as exc:
            self.close()
            raise ValueError(f'Missing referenced video frame {index}: {self.path}') from exc
        except Exception:
            self.close()
            raise

    def close(self):
        if self.container is not None:
            self.container.close()
        self.container = self.iterator = None
        self.position = -1
        self.cache.clear()
        self.bytes = 0
