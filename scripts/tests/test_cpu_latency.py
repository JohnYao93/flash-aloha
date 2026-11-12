"""CPU wake-up latency hold for camera servers, with a fake device file."""
import os
import struct

from camera_utils.cpu_latency import cpu_latency_limit


def test_holds_the_requested_limit_until_released(tmp_path):
    device = tmp_path / 'cpu_dma_latency'
    device.write_bytes(b'')
    messages = []
    with cpu_latency_limit(20, path=str(device), notify=messages.append) as active:
        assert active
        assert struct.unpack('i', device.read_bytes()) == (20,)
        assert 'limited to 20 us' in messages[-1]
    assert not any(str(device) in link for link in os.listdir('/proc/self/fd'))


def test_missing_permission_warns_and_lets_the_camera_run(tmp_path):
    messages = []
    with cpu_latency_limit(20, path=str(tmp_path / 'missing'), notify=messages.append) as active:
        assert not active
    assert 'WARNING' in messages[0] and 'README' in messages[0]


def test_negative_limit_disables_the_request(tmp_path):
    device = tmp_path / 'cpu_dma_latency'
    device.write_bytes(b'')
    with cpu_latency_limit(-1, path=str(device), notify=lambda *_: None) as active:
        assert not active
    assert device.read_bytes() == b''


def test_camera_server_cli_exposes_the_limit():
    import subprocess, sys
    out = subprocess.run([sys.executable, 'src/camera_utils/camera_server.py', '--help'],
                         capture_output=True, text=True, check=True).stdout
    assert '--cpu-latency-us' in out
