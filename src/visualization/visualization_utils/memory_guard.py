"""RAM guard for offline Rerun exports and the viewer that opens them.

Viewing runs on the robot host. If an export exhausts RAM, the kernel swaps or
OOM-kills, which stalls or kills the collector holding the arms. Exports check
MemAvailable before every frame and stop while the floor is still free; the
viewer opens the saved file with a memory limit that leaves the same floor.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys

GIB = 1024**3
FLOOR_FRACTION = 0.25
MIN_FLOOR_BYTES = 4 * GIB


def meminfo(path='/proc/meminfo'):
    values = {}
    for line in Path(path).read_text().splitlines():
        name, _, rest = line.partition(':')
        values[name] = int(rest.split()[0]) * 1024
    return values


class MemoryGuard:
    """Raise MemoryError before available RAM drops below the floor."""
    def __init__(self, floor_bytes=None, *, read=None):
        self.read = read = read or (lambda: meminfo())
        if floor_bytes is None:
            gigabytes = os.environ.get('FLASH_ALOHA_MIN_FREE_MEMORY_GB')
            floor_bytes = (float(gigabytes) * GIB if gigabytes else
                           max(MIN_FLOOR_BYTES, FLOOR_FRACTION * read()['MemTotal']))
        self.floor = int(floor_bytes)

    def available(self):
        return self.read()['MemAvailable']

    def check(self, context='Rerun export'):
        available = self.available()
        if available < self.floor:
            raise MemoryError(
                f'{context} stopped: {available / GIB:.1f} GiB RAM available, below the '
                f'{self.floor / GIB:.1f} GiB kept free for robot processes. Close other programs, or set '
                'FLASH_ALOHA_MIN_FREE_MEMORY_GB only if no robot program is running.')
        return available

    def viewer_limit(self):
        """Viewer budget leaving the floor free; Rerun drops the oldest data at this limit."""
        return max(GIB, self.check('Opening Rerun') - self.floor)


def open_rrd(path, guard=None):
    guard = guard or MemoryGuard()
    command = [sys.executable, '-m', 'rerun', str(path), '--memory-limit', f'{guard.viewer_limit()}B']
    print('[Rerun] Opening: ' + shlex.join(command), flush=True)
    return subprocess.Popen(command, start_new_session=True)
