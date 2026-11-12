"""Garbage-collection contract for every process that captures or carries live data.

Python's cyclic collector pauses the whole process while it runs. Full
collections are triggered by allocation counts, so in a recorder they recur at
the same rows of every capture and stall the 100 Hz loop for 25-50 ms.

Streaming processes therefore disable automatic collection once streaming
starts and keep it disabled for the rest of the process. Cyclic garbage is
reclaimed only by explicit ``collect_idle`` calls at points where no episode is
being captured (after an episode stops, between rollouts).
"""
import gc
import threading

_lock = threading.Lock()
_latched = False


def pause_for_stream():
    """Disable automatic collection for the rest of this process.

    Idempotent and cheap, so the control loop calls it on every capture. The
    first call freezes existing objects (constant time, no collection: a
    collection here would stall the capture it precedes), so later idle
    collections only scan objects created while streaming. A library that
    re-enables collection is overridden at the next call.
    """
    global _latched
    if _latched and not gc.isenabled():
        return
    with _lock:
        if not _latched:
            gc.freeze()
            _latched = True
        gc.disable()


def collect_idle():
    """Reclaim cyclic garbage at an idle point; automatic collection stays disabled.

    Never call this while an episode is being captured.
    """
    if not _latched:
        return 0
    return gc.collect()


def is_paused():
    return _latched and not gc.isenabled()


def _reset_for_tests():
    """Restore normal collection after a test (test facility only)."""
    global _latched
    with _lock:
        if _latched:
            gc.unfreeze()
        _latched = False
        gc.enable()
