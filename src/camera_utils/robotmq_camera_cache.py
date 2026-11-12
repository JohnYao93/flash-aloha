"""Process-isolated camera consumers with a native RobotMQ metadata cache.

No robot SDK, Torch, camera acquisition, encoding, or recording imports belong
here. The helper reads camera-server shared memory in parallel. Only small
coherent reference bundles travel through RMQ; pixels stay in bounded mmap rings.
"""
from __future__ import annotations

# Support direct execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "src")
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

import atexit
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

from camera_utils.camera_timing import CameraTimingError, require_camera_timestamps, validate_shared_clock_contract
from contracts import capture_gc

SOURCE_HEADER = struct.Struct('Qd')
SLOT_HEADER = struct.Struct('QQdd')
TOPIC = 'observations'
RING_SLOTS = 8


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: str
    width: int
    height: int

    @property
    def frame_bytes(self):
        return self.width * self.height * 3


class ImageRing:
    """A single helper writer and serialized consumer readers, never waiting.

    Byte-range locks protect each slot across processes. Writers skip pinned
    slots; readers verify generation and complete frame metadata before copying.
    The owning cache serializes consumer threads because POSIX locks are per
    process. No reader ever falls forward to a newer frame.
    """
    def __init__(self, path, spec, *, create=False, slots=RING_SLOTS):
        self.spec, self.slots = spec, slots
        self.stride = SLOT_HEADER.size + spec.frame_bytes
        self.fd = os.open(path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
        self.mapping = None
        try:
            if create:
                # Reserve RAM/disk up front: a later mmap write must not SIGBUS.
                os.posix_fallocate(self.fd, 0, self.stride * slots)
            if os.fstat(self.fd).st_size != self.stride * slots:
                raise RuntimeError('Camera cache ring size mismatch')
            self.mapping = mmap.mmap(self.fd, self.stride * slots)
        except BaseException:
            os.close(self.fd)
            raise
        self.generation = 0
        self.next_slot = 0

    def _lock(self, slot, exclusive):
        try:
            fcntl.lockf(self.fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
                        1, slot * self.stride)
            return True
        except BlockingIOError:
            return False

    def _unlock(self, slot):
        fcntl.lockf(self.fd, fcntl.LOCK_UN, 1, slot * self.stride)

    def write(self, pixels, number, stamp):
        if len(pixels) != self.spec.frame_bytes:
            raise ValueError('Camera frame size differs from configured native dimensions')
        for step in range(self.slots):
            slot = (self.next_slot + step) % self.slots
            if not self._lock(slot, True):
                continue
            try:
                offset = slot * self.stride
                self.mapping[offset + SLOT_HEADER.size:offset + self.stride] = pixels
                self.generation += 1
                received = time.monotonic()
                self.mapping[offset:offset + SLOT_HEADER.size] = SLOT_HEADER.pack(
                    self.generation, number, stamp, received)
                self.next_slot = (slot + 1) % self.slots
                return dict(slot=slot, generation=self.generation, frame_number=number,
                            capture_timestamp=stamp, consumer_received_timestamp=received)
            finally:
                self._unlock(slot)
        raise CameraTimingError(f'{self.spec.name}: all camera cache slots are pinned')

    def copy(self, reference):
        slot = int(reference['slot'])
        if not 0 <= slot < self.slots or not self._lock(slot, False):
            raise CameraTimingError(f'{self.spec.name}: selected camera cache slot is unavailable')
        try:
            offset = slot * self.stride
            expected = SLOT_HEADER.pack(reference['generation'], reference['frame_number'],
                                        reference['capture_timestamp'], reference['consumer_received_timestamp'])
            if self.mapping[offset:offset + SLOT_HEADER.size] != expected:
                raise CameraTimingError(f'{self.spec.name}: selected camera frame was overwritten')
            # One owned, writable copy; the slot stays pinned if memcpy releases
            # the GIL or this process is paused during the copy.
            return np.ndarray((self.spec.height, self.spec.width, 3), np.uint8,
                              buffer=self.mapping, offset=offset + SLOT_HEADER.size).copy()
        finally:
            self._unlock(slot)

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            os.close(self.fd)
            self.mapping = None


class SourceReader:
    """Read the existing producer's seqlock and verify its driver-clock inode."""
    def __init__(self, spec):
        self.spec = spec
        self.fd = None
        self.mapping = None
        self.last = None
        self.identity = None
        self.next_identity_check = 0.

    def read(self):
        if self.fd is None:
            try:
                self.fd = os.open(self.spec.source, os.O_RDONLY)
            except FileNotFoundError:
                return None
            try:
                stat = os.fstat(self.fd)
                if stat.st_size != SOURCE_HEADER.size + self.spec.frame_bytes:
                    raise CameraTimingError(f'{self.spec.name}: source dimensions do not match camera configuration')
                validate_shared_clock_contract(SimpleNamespace(_fd=self.fd, name=Path(self.spec.source).name),
                                               directory=str(Path(self.spec.source).parent))
                self.identity = stat.st_dev, stat.st_ino
                self.mapping = mmap.mmap(self.fd, stat.st_size, access=mmap.ACCESS_READ)
            except BaseException:
                self.close()
                raise
        now = time.monotonic()
        if now >= self.next_identity_check:
            stat = os.stat(self.spec.source)
            if (stat.st_dev, stat.st_ino) != self.identity:
                raise CameraTimingError(f'{self.spec.name}: camera server restarted; restart camera consumers')
            self.next_identity_check = now + .05
        for _ in range(3):
            header = self.mapping[:SOURCE_HEADER.size]
            number, stamp = SOURCE_HEADER.unpack(header)
            if number & (1 << 63) or not math.isfinite(stamp) or stamp <= 0:
                continue
            if self.last == (number, stamp):
                return None
            pixels = self.mapping[SOURCE_HEADER.size:]
            if self.mapping[:SOURCE_HEADER.size] == header:
                if self.last is not None and (number <= self.last[0] or stamp <= self.last[1]):
                    raise CameraTimingError(f'{self.spec.name}: camera frame ID/clock moved backwards or was reused')
                self.last = number, stamp
                return pixels, number, stamp
        return None

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def _helper(config_path):
    from robotmq import RMQClient, RMQLogLevel
    capture_gc.pause_for_stream()  # Owns only bounded rings/current references; never collects.
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    parent = config['parent_pid']
    stop, changed = threading.Event(), threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    lock = threading.Lock()
    packets, errors, workers = {}, {}, {}
    client = RMQClient('camera-cache-publisher', config['endpoint'], RMQLogLevel.ERROR)

    def consume(item):
        spec = CameraSpec(**item['spec'])
        reader, ring = SourceReader(spec), None
        try:
            ring = ImageRing(item['ring'], spec)
            while not stop.is_set():
                frame = reader.read()
                if frame is not None:
                    reference = ring.write(*frame)
                    with lock:
                        packets[spec.name] = reference
                        changed.set()
                stop.wait(.001)
        except Exception as exc:
            with lock:
                errors[spec.name] = str(exc)
                changed.set()
        finally:
            reader.close()
            if ring is not None:
                ring.close()

    next_config = 0.
    try:
        while not stop.is_set() and os.getppid() == parent:
            now = time.monotonic()
            if now >= next_config:
                config = json.loads(config_path.read_text())
                for name, item in config['cameras'].items():
                    if name not in workers:
                        thread = threading.Thread(target=consume, args=(item,), name=f'camera-cache-{name}', daemon=True)
                        workers[name] = thread
                        thread.start()
                next_config = now + .02
            if changed.wait(.001):
                with lock:
                    payload = json.dumps(dict(cameras=packets, errors=errors), separators=(',', ':')).encode()
                    changed.clear()
                # Network waits are confined to this helper, never the robot.
                client.put_data(TOPIC, payload, timeout_s=.1, automatic_resend=False)
    finally:
        stop.set()
        for worker in workers.values():
            worker.join(timeout=1.)
        # SIGKILL of the consumer cannot leave a reader running forever.
        if os.getppid() != parent:
            import shutil
            shutil.rmtree(config['image_directory'], ignore_errors=True)
            shutil.rmtree(config_path.parent, ignore_errors=True)


class RobotMQCameraCache:
    """One helper/native receiver per consumer process; cameras added on demand.

    Configured rings/readers live until the final lease closes, so adding a camera
    never interrupts existing leases. All snapshot image copies are serialized.
    ``image_directory`` is for synthetic tests; production uses RAM in /dev/shm.
    """
    def __init__(self, *, image_directory='/dev/shm'):
        from robotmq import RMQServer, RMQLogLevel
        self.lock = threading.RLock()
        self.rings, self.specs = {}, {}
        self.closed = False
        self.process = self.server = self.log = None
        self.directory = tempfile.TemporaryDirectory(prefix='aloha-rmq-')
        self.images = None
        try:
            self.images = tempfile.TemporaryDirectory(prefix='aloha-camera-cache-', dir=image_directory)
            self.endpoint = 'ipc://' + self.directory.name + '/cache'
            self.config_path = Path(self.directory.name) / 'config.json'
            self.server = RMQServer('camera-cache-' + Path(self.directory.name).name,
                                    self.endpoint, RMQLogLevel.ERROR)
            self.server.add_topic(TOPIC, .2)
            self._write_config()
            self.log = open(Path(self.directory.name) / 'helper.log', 'w+')
            # Exec a minimal module, not multiprocessing-spawn of the collector
            # __main__, which would import its Torch/UI/SDK graph into the helper.
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(self.config_path)],
                                            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
            print(f'[Camera cache] RobotMQ reader process started (PID {self.process.pid})', flush=True)
        except BaseException:
            self.close()
            raise

    def _write_config(self):
        config = dict(parent_pid=os.getpid(), endpoint=self.endpoint, image_directory=self.images.name,
                      cameras={name: dict(spec=asdict(spec), ring=str(Path(self.images.name) / f'{index}.ring'))
                               for index, (name, spec) in enumerate(self.specs.items())})
        temporary = self.config_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(config))
        temporary.replace(self.config_path)

    def add(self, specs):
        with self.lock:
            self._check_process()
            added = []
            try:
                for spec in specs:
                    if spec.name in self.specs:
                        if self.specs[spec.name] != spec:
                            raise ValueError(f'{spec.name}: conflicting camera cache configuration')
                        continue
                    if min(spec.width, spec.height) <= 0:
                        raise ValueError('Camera dimensions must be positive')
                    path = Path(self.images.name) / f'{len(self.specs)}.ring'
                    try:
                        ring = ImageRing(path, spec, create=True)
                    except BaseException:
                        path.unlink(missing_ok=True)
                        raise
                    self.rings[spec.name] = ring
                    self.specs[spec.name] = spec
                    added.append((spec.name, path))
                self._write_config()
            except BaseException:
                # Existing leases keep their working configuration if adding
                # another camera fails (for example, insufficient /dev/shm).
                for name, path in reversed(added):
                    self.rings.pop(name).close()
                    del self.specs[name]
                    path.unlink(missing_ok=True)
                raise

    def _check_process(self):
        if self.closed:
            raise CameraTimingError('Camera cache is closed')
        if self.process is not None and self.process.poll() is not None:
            self.log.seek(0)
            detail = self.log.read()[-2000:]
            raise CameraTimingError(f'Camera cache helper exited ({self.process.returncode}): {detail}')

    def _bundle(self, names):
        self._check_process()
        messages, _transport_times = self.server.peek_data(TOPIC, -1)
        cutoff = time.monotonic()
        bundle = json.loads(messages[-1]) if messages else dict(cameras={}, errors={})
        for name in names:
            if name in bundle['errors']:
                raise CameraTimingError(f"Camera cache {name}: {bundle['errors'][name]}")
        return bundle['cameras'], cutoff

    def check_health(self, names):
        with self.lock:
            self._bundle(names)

    def snapshot(self, names, *, copy_images=False, validate=True):
        names = list(names)
        with self.lock:
            packets, cutoff = self._bundle(names)
            missing = [name for name in names if name not in packets or name not in self.rings]
            if missing:
                raise CameraTimingError(f'Required timestamped camera frames are not available: {missing}')
            references = {name: {key: packets[name][key] for key in (
                'frame_number', 'capture_timestamp', 'consumer_received_timestamp')} for name in names}
            if validate:
                require_camera_timestamps({name: ref['capture_timestamp'] for name, ref in references.items()}, now=cutoff)
            observation = dict(snapshot_timestamp=cutoff, cameras=references)
            images = {name: self.rings[name].copy(packets[name]) for name in names} if copy_images else {}
            return images, observation

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.)
            self.server = None  # joins the native receiver before unlinking IPC
            for ring in self.rings.values():
                ring.close()
            self.rings.clear()
            if self.log is not None:
                self.log.close()
            if self.images is not None:
                self.images.cleanup()
            self.directory.cleanup()


_pool_lock = threading.RLock()
_cache = None
_users = 0


class CameraLease:
    def __init__(self, specs):
        global _cache, _users
        specs = list(specs)
        if not specs:
            raise ValueError('A camera reader lease needs at least one camera')
        self.names = [spec.name for spec in specs]
        self.closed = False
        with _pool_lock:
            created = _cache is None
            if created:
                _cache = RobotMQCameraCache()
            try:
                _cache.add(specs)
            except BaseException:
                if created:
                    _cache.close()
                    _cache = None
                raise
            self.cache = _cache
            _users += 1

    def check_health(self):
        self.cache.check_health(self.names)

    def wait_ready(self, timeout=5.):
        deadline = time.monotonic() + timeout
        while True:
            self.check_health()
            try:
                self.cache.snapshot(self.names)
                return
            except CameraTimingError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.005)

    def close(self):
        global _cache, _users
        with _pool_lock:
            if self.closed:
                return
            self.closed = True
            _users -= 1
            if not _users:
                self.cache.close()
                _cache = None


def active_cache():
    return _cache


def _close_at_exit():
    if _cache is not None:
        _cache.close()


atexit.register(_close_at_exit)


if __name__ == '__main__':
    _helper(sys.argv[1])
