"""Bounded capture queues: only the worker encodes or accesses its Zarr file."""
from collections import deque
from collections.abc import MutableMapping
from concurrent.futures import Future
import copy
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import weakref

import numpy as np


class BufferLimitReached(RuntimeError):
    """Capture outpaced disk writing and exhausted the raw-frame budget."""


class DrainingWorker:
    """One ordered writer, including barriers and rollback commands.

    The byte budget includes the in-flight item. Failed writes poison subsequent
    data writes; explicit cleanup commands can still repair the committed prefix.
    """
    def __init__(self, write, max_bytes=32 * 1024**3):
        self.write, self.max_bytes = write, max_bytes
        self.condition = threading.Condition()
        self.queue = deque()
        self.pending_bytes = self.pending_count = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='recording-disk-writer', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'Background recording write failed: {self.error}') from self.error

    def submit(self, item, size, index):
        with self.condition:
            self.check()
            if self.closed:
                raise RuntimeError('Recording writer is closed')
            if self.pending_bytes + size > self.max_bytes:
                raise BufferLimitReached('Recording disk queue reached its RAM limit')
            self.pending_bytes += size
            self.pending_count += 1
            self.queue.append((index, size, item, None))
            self.condition.notify()

    def call(self, function, *, cleanup=False):
        future = Future()
        with self.condition:
            if not cleanup:
                self.check()
            if self.closed:
                raise RuntimeError('Recording writer is closed')
            self.queue.append((None, 0, (function, cleanup), future))
            self.condition.notify()
        return future.result()

    def discard_after(self, count):
        """Cancel queued suffix; an in-flight write finishes before the next call."""
        with self.condition:
            kept = deque()
            for entry in self.queue:
                index, size, _, _ = entry
                if index is not None and index >= count:
                    self.pending_bytes -= size
                    self.pending_count -= 1
                else:
                    kept.append(entry)
            self.queue = kept

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        if threading.current_thread() is not self.thread:
            self.thread.join()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.closed)
                if not self.queue:
                    return
                index, size, item, future = self.queue.popleft()
            try:
                if future is None:
                    self.check()
                    self.write(item)
                else:
                    function, cleanup = item
                    if not cleanup:
                        self.check()
                    future.set_result(function())
            except BaseException as exc:
                if future is not None:
                    future.set_exception(exc)
                else:
                    with self.condition:
                        if self.error is None:
                            self.error = exc
            finally:
                if future is None:
                    with self.condition:
                        self.pending_bytes -= size
                        self.pending_count -= 1
                # Do not retain the last raw frame while idle.
                item = function = None


def _dispose_store(worker, state, directory):
    worker.discard_after(0)
    try:
        if threading.current_thread() is worker.thread:
            if state.writer:
                state.writer.close()
        else:
            worker.call(lambda: state.writer.close() if state.writer else None, cleanup=True)
    finally:
        worker.close()
        shutil.rmtree(directory, ignore_errors=True)
