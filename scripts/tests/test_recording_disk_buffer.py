"""Shared asynchronous spool contract, with no cameras or robot hardware."""
import gc
from pathlib import Path
import sys
import threading
from unittest import mock
import weakref

import cv2
from data_processing import episode_store
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from recording import record_episode_single as single
from recording.recording_buffer import BufferLimitReached, DrainingWorker


def sample(value):
    return single.EpisodeSample({'camera': np.full((24, 32, 3), value, np.uint8)},
                                np.full(14, value, dtype=np.float32), np.zeros(14), value / 100,
                                capture_interval=.01)






def test_queue_budget_includes_inflight_and_rejects_without_silent_drops():
    entered, release = threading.Event(), threading.Event()
    written = []
    def write(item):
        entered.set()
        assert release.wait(5)
        written.append(item)
    worker = DrainingWorker(write, max_bytes=20)
    try:
        worker.submit('first', 10, 0)
        assert entered.wait(5)
        worker.submit('second', 10, 1)
        with pytest.raises(BufferLimitReached):
            worker.submit('third', 1, 2)
        assert worker.pending_bytes == 20
        release.set()
        worker.call(lambda: None)
        assert written == ['first', 'second'] and worker.pending_bytes == 0
    finally:
        release.set()
        worker.close()
