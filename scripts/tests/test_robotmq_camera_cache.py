"""Real RobotMQ + synthetic mmap cameras only; no live endpoints or hardware."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import camera_utils.robotmq_camera_cache as native
from camera_utils.camera_timing import CameraTimingError, publish_shared_clock_contract


PUBLISHER = r'''
import json, mmap, os, pathlib, struct, sys, time
root = pathlib.Path(sys.argv[1])
config = json.loads((root / 'sources.json').read_text())
sources = []
for spec in config:
    f = open(spec['source'], 'r+b', buffering=0)
    sources.append((spec['name'], f, mmap.mmap(f.fileno(), 0)))
number = 0
while True:
    flags = json.loads((root / 'flags.json').read_text())
    for name, f, image in sources:
        if name in flags.get('frozen', []):
            continue
        image[:16] = struct.pack('Qd', number | (1 << 63), float('nan'))
        image[16:] = bytes([number % 251]) * (len(image) - 16)
        stamp = time.monotonic() - .002 - flags.get('delay', {}).get(name, 0)
        image[:16] = struct.pack('Qd', number, stamp)
    number += 1
    time.sleep(.01)
'''


class Sources:
    def __init__(self, root, *, width=64, height=48):
        self.root = root
        self.specs = [native.CameraSpec(name, str(root / name), width, height)
                      for name in ('video_left', 'video_overhead', 'video_right')]
        for spec in self.specs:
            with open(spec.source, 'w+b') as f:
                f.truncate(16 + spec.frame_bytes)
                publish_shared_clock_contract(SimpleNamespace(_fd=f.fileno(), name=spec.name), directory=root)
        (root / 'sources.json').write_text(json.dumps([native.asdict(s) for s in self.specs]))
        self.flags()
        self.process = subprocess.Popen([sys.executable, '-c', PUBLISHER, str(root)])

    def flags(self, **flags):
        path = self.root / 'flags.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(flags))
        temporary.replace(path)

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=5)


@pytest.fixture
def sources(tmp_path):
    result = Sources(tmp_path)
    try:
        yield result
    finally:
        result.close()


@pytest.fixture
def cache(sources, tmp_path):
    result = native.RobotMQCameraCache(image_directory=tmp_path)
    try:
        assert result.process.args[1] == str(Path(native.__file__).resolve())
        result.add(sources.specs)
        ready(result)
        yield result
    finally:
        result.close()


def ready(cache, names=None):
    names = names or list(cache.specs)
    deadline = time.monotonic() + 5
    while True:
        cache.check_health(names)
        try:
            return cache.snapshot(names)
        except CameraTimingError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def freeze(cache, sources):
    sources.flags(frozen=[s.name for s in sources.specs])
    # Allow the publisher to finish an iteration and helper to publish its tail.
    time.sleep(.025)


def test_default_lease_shares_native_cache_and_adds_cameras_without_restart(sources, tmp_path, monkeypatch):
    import camera_utils.capture as cameras
    factory = native.RobotMQCameraCache
    monkeypatch.setattr(native, 'RobotMQCameraCache', lambda: factory(image_directory=tmp_path))
    by_short = {s.name.removeprefix('video_'): s for s in sources.specs}
    monkeypatch.setattr(native, 'CameraSpec', lambda name, source, width, height:
                        by_short[name.removeprefix('video_')])
    monkeypatch.delenv('FLASH_ALOHA_CAMERA_CACHE', raising=False)
    first = cameras.CameraReaderLease(['video_left'])
    second = None
    try:
        first.wait_ready()
        pid = first.cache.process.pid
        second = cameras.CameraReaderLease(['video_left', 'video_right'])
        second.wait_ready()
        assert first.cache is second.cache and second.cache.process.pid == pid
        first.close()
        images, observation = cameras.get_camera_observation(['video_right'], copy_images=True)
        assert np.all(images['video_right'] == observation['cameras']['video_right']['frame_number'] % 251)
        assert cameras.get_last_camera_observation() is observation
        assert cameras.get_latest_camera_frame_number('video_right') >= observation['cameras']['video_right']['frame_number']
        assert second.cache.process.poll() is None
    finally:
        first.close()
        if second is not None:
            second.close()
    assert native.active_cache() is None
    assert native._users == 0
    assert first.cache.process.poll() is not None
    assert not Path(first.cache.directory.name).exists()
    assert not Path(first.cache.images.name).exists()


def test_recording_metadata_and_inference_pixels_are_the_same_snapshot(cache, sources, monkeypatch, tmp_path):
    from contracts import standard_capture
    from policy import training_utils  # Import heavy consumers before freezing synthetic cameras.
    from recording.distributed_recording import ControlWriter
    freeze(cache, sources)
    monkeypatch.setattr(native, '_cache', cache)
    names = list(cache.specs)
    qpos = np.arange(14, dtype=np.float32)
    recorded = standard_capture.capture_observation(lambda: (qpos, -qpos), names)
    inferred = standard_capture.capture_observation(lambda: (qpos, -qpos), names, include_images=True)
    assert not recorded.images
    assert recorded.camera_observation['cameras'] == inferred.camera_observation['cameras']
    writer = ControlWriter(tmp_path / 'controls.zarr')
    writer.append(dict(recorded.recording_fields(), commanded_action=qpos))
    for name, frame in inferred.images.items():
        assert np.all(frame == recorded.camera_observation['cameras'][name]['frame_number'] % 251)
        assert frame.flags.writeable
        for key, value in recorded.camera_observation['cameras'][name].items():
            assert writer.group[f'camera_observation/{name}/{key}'][0] == value


def test_failed_startup_releases_last_lease(tmp_path, monkeypatch):
    import camera_utils.capture as cameras
    factory = native.RobotMQCameraCache
    created = []
    def build():
        cache = factory(image_directory=tmp_path)
        created.append(cache)
        return cache
    monkeypatch.setattr(native, 'RobotMQCameraCache', build)
    missing = native.CameraSpec('video_left', str(tmp_path / 'missing'), 64, 48)
    monkeypatch.setattr(native, 'CameraSpec', lambda *args: missing)
    monkeypatch.delenv('FLASH_ALOHA_CAMERA_CACHE', raising=False)
    with pytest.raises(CameraTimingError, match='not available'):
        cameras.acquire_camera_readers(['video_left'], timeout=.05)
    assert native.active_cache() is None and native._users == 0
    assert created[0].process.poll() is not None
    assert not Path(created[0].images.name).exists()


@pytest.mark.parametrize('width,height', [(64, 48), (1280, 800)])
def test_cache_advances_while_consumer_gil_is_blocked(tmp_path, width, height):
    sources = Sources(tmp_path, width=width, height=height)
    # Full-resolution case also exercises production tmpfs allocation/cleanup.
    cache = native.RobotMQCameraCache(image_directory='/dev/shm' if width == 1280 else tmp_path)
    try:
        cache.add(sources.specs)
        ready(cache)
        usleep = ctypes.PyDLL(None).usleep  # Intentionally retains the Python GIL.
        usleep.argtypes, usleep.restype = [ctypes.c_uint], ctypes.c_int
        for _ in range(3):
            before = cache.snapshot(list(cache.specs))[1]
            assert usleep(150000) == 0
            images, after = cache.snapshot(list(cache.specs), copy_images=True)
            for name, ref in after['cameras'].items():
                assert ref['frame_number'] > before['cameras'][name]['frame_number']
                assert after['snapshot_timestamp'] - ref['capture_timestamp'] < .05
                assert np.all(images[name] == ref['frame_number'] % 251)
    finally:
        cache.close()
        sources.close()


def test_stopped_camera_fails_closed_while_other_readers_advance(cache, sources):
    names = list(cache.specs)
    before = cache.snapshot(names)[1]
    sources.flags(frozen=['video_left'])
    time.sleep(.09)
    with pytest.raises(CameraTimingError, match='video_left: frame age'):
        cache.snapshot(names)
    after = cache.snapshot(['video_right'])[1]
    assert after['cameras']['video_right']['frame_number'] > before['cameras']['video_right']['frame_number']


def test_missing_camera_does_not_block_existing_readers(cache, tmp_path):
    missing = native.CameraSpec('video_missing', str(tmp_path / 'does-not-exist'), 64, 48)
    cache.add([missing])
    with pytest.raises(CameraTimingError, match='not available'):
        cache.snapshot([missing.name])
    assert cache.snapshot(['video_right'])[1]['cameras']['video_right']


def test_failed_camera_addition_does_not_disturb_existing_leases(cache, tmp_path):
    before = dict(cache.specs)
    extra = native.CameraSpec('video_extra', str(tmp_path / 'extra'), 64, 48)
    invalid = native.CameraSpec('video_invalid', str(tmp_path / 'invalid'), 0, 48)
    with pytest.raises(ValueError, match='positive'):
        cache.add([extra, invalid])
    assert cache.specs == before
    assert len(list(Path(cache.images.name).glob('*.ring'))) == len(before)
    assert cache.snapshot(['video_right'])[1]['cameras']['video_right']


def test_skew_is_rejected_without_relaxing_age(cache, sources, monkeypatch):
    monkeypatch.setenv('CAMERA_MAX_SKEW_MS', '10')
    sources.flags(frozen=['video_right'])
    time.sleep(.04)
    sources.flags(delay={'video_right': .025})
    time.sleep(.03)
    with pytest.raises(CameraTimingError, match='spread'):
        cache.snapshot(list(cache.specs))


def test_helper_failure_is_reported_even_if_its_last_frame_is_fresh(cache):
    cache.process.terminate()
    cache.process.wait(timeout=3)
    with pytest.raises(CameraTimingError, match='helper exited'):
        cache.snapshot(['video_left'])


def test_source_restart_is_rejected(cache, sources):
    spec = sources.specs[0]
    replacement = Path(spec.source).with_suffix('.new')
    replacement.write_bytes(Path(spec.source).read_bytes())
    replacement.replace(spec.source)
    deadline = time.monotonic() + 2
    while True:
        try:
            cache.check_health([spec.name])
        except CameraTimingError as exc:
            assert 'restarted' in str(exc)
            break
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_source_rejects_unverified_clock_and_reused_frame_id(tmp_path):
    spec = native.CameraSpec('video_test', str(tmp_path / 'video_test'), 2, 2)
    with open(spec.source, 'w+b') as f:
        f.truncate(28)
        reader = native.SourceReader(spec)
        with pytest.raises(CameraTimingError, match='clock contract'):
            reader.read()
        publish_shared_clock_contract(SimpleNamespace(_fd=f.fileno(), name=spec.name), directory=tmp_path)
        now = time.monotonic()
        f.seek(0)
        f.write(struct.pack('Qd', 7, now) + bytes([7])*12)
        f.flush()
        assert reader.read()[1] == 7
        f.seek(0)
        f.write(struct.pack('Qd', 7, now+.01))
        f.flush()
        try:
            with pytest.raises(CameraTimingError, match='reused'):
                reader.read()
        finally:
            reader.close()


def test_overwritten_slot_never_substitutes_a_new_frame(tmp_path):
    spec = native.CameraSpec('video_test', '', 2, 2)
    ring = native.ImageRing(tmp_path / 'ring', spec, create=True, slots=2)
    try:
        first = ring.write(bytes([1])*12, 1, time.monotonic())
        image = ring.copy(first)
        ring.write(bytes([2])*12, 2, time.monotonic())
        assert np.all(ring.copy(first) == 1)
        ring.write(bytes([3])*12, 3, time.monotonic())
        assert np.all(image == 1), 'Returned images must own their pixels'
        with pytest.raises(CameraTimingError, match='overwritten'):
            ring.copy(first)
    finally:
        ring.close()


def test_inference_keeps_selected_pixels_during_new_bundle_publication(cache, monkeypatch):
    original = cache.rings['video_left'].copy
    selected = []
    def copy(reference):
        selected.append(reference.copy())
        time.sleep(.025)  # New native bundles arrive before selected pixels copy.
        return original(reference)
    monkeypatch.setattr(cache.rings['video_left'], 'copy', copy)
    images, observation = cache.snapshot(['video_left'], copy_images=True)
    assert observation['cameras']['video_left']['frame_number'] == selected[0]['frame_number']
    assert np.all(images['video_left'] == selected[0]['frame_number'] % 251)


def test_recording_snapshot_never_reads_pixels(cache, monkeypatch):
    for ring in cache.rings.values():
        monkeypatch.setattr(ring, 'copy', lambda *_: pytest.fail('recording copied pixels'))
    assert cache.snapshot(list(cache.specs))[0] == {}


def test_ring_slot_pin_survives_a_parent_python_pause(cache):
    packets, _ = cache._bundle(['video_left'])
    ref = packets['video_left']
    ring = cache.rings['video_left']
    assert ring._lock(ref['slot'], False)
    try:
        usleep = ctypes.PyDLL(None).usleep
        usleep.argtypes, usleep.restype = [ctypes.c_uint], ctypes.c_int
        usleep(200000)  # Enough frames to wrap an eight-slot ring twice.
        assert np.all(ring.copy(ref) == ref['frame_number'] % 251)
    finally:
        ring._unlock(ref['slot'])
    assert cache.snapshot(['video_left'])[1]['cameras']['video_left']['frame_number'] > ref['frame_number']
