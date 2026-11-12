"""Keep CPUs out of deep idle states while a USB camera streams.

At 100 fps the UVC driver must resubmit isochronous transfers on time. Waking
an idle laptop CPU from C6-C10 can occasionally take too long; the controller
then misses camera packets, the driver loses frame sync and drops a burst of
frames (seen as stale-frame timing failures). A PM QoS request through
/dev/cpu_dma_latency limits the allowed wake-up latency for as long as the file
stays open; the kernel releases it automatically when the process exits.

This is a power hint only: capture, frame selection, clocks and timing
validation are unchanged. Writing the device needs permission, see
scripts/cameras/README.md.
"""
from contextlib import contextmanager
import os
import struct

DEVICE = '/dev/cpu_dma_latency'
DEFAULT_LIMIT_US = 20  # Allows C1E (~2 us); blocks C6 and deeper (>= ~170 us).


@contextmanager
def cpu_latency_limit(microseconds=DEFAULT_LIMIT_US, *, path=DEVICE, notify=print):
    """Hold a CPU wake-up latency limit; yields True when it is active.

    A negative or None limit disables the request. Missing permission is
    reported loudly but does not stop the camera (its timing checks still
    fail closed on stale frames).
    """
    if microseconds is None or microseconds < 0:
        yield False
        return
    try:
        fd = os.open(path, os.O_WRONLY)
    except OSError as exc:
        notify(f'[CPU] WARNING: cannot limit CPU wake-up latency ({exc}). Deep idle states '
               'can drop bursts of USB camera frames; grant access per scripts/cameras/README.md.')
        yield False
        return
    try:
        os.write(fd, struct.pack('i', int(microseconds)))
        notify(f'[CPU] Wake-up latency limited to {int(microseconds)} us while this camera streams')
        yield True
    finally:
        os.close(fd)  # Releases the request.
