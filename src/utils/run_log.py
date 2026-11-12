"""Per-run robot log for offline diagnosis; it never feeds back into control.

Each run gets ``<root>/<date>/<HHMMSS>_<name>/`` with:

- ``console.log``: everything the run prints.
- ``run.json``: the run's settings.
- ``steps_<n>.npz``: one row per executed control step (commanded target, measured
  qpos after sending it, and caller fields), saved per segment and on every exit path.
- ``handoffs_<n>.json``: direct inference's chunk handoff numbers.

Direct inference logs to ``logs/inference``; physical replay to ``logs/replay``.
"""
import json
import re
import sys
import time
from datetime import datetime

import numpy as np

import config.paths as paths


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


class RunLog:
    def __init__(self, task_name, root=None):
        root = paths.INFERENCE_LOG_DIR if root is None else root
        now = datetime.now()
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(task_name)).strip('_') or 'run'
        self.directory = root / now.strftime('%Y-%m-%d') / f"{now.strftime('%H%M%S')}_{name}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._console = open(self.directory / 'console.log', 'a', buffering=1)
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._console)
        self.rows, self.handoffs, self.segments, self.closed = [], [], 0, False
        print(f'[Run log] {self.directory}')

    def write_settings(self, **settings):
        (self.directory / 'run.json').write_text(json.dumps(settings, indent=1, default=str))

    def step(self, **fields):
        if not self.closed:
            self.rows.append(dict(fields, time=time.time()))

    def handoff(self, metrics):
        self.handoffs.append(metrics)

    def flush(self, **metadata):
        """Save the steps and handoffs collected since the last flush (one demo or segment)."""
        if self.closed and self._console.closed or (not self.rows and not self.handoffs):
            return
        name = f'{self.segments:03d}'
        if self.rows:
            arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
            np.savez(self.directory / f'steps_{name}.npz', **arrays,
                     **{f'meta_{key}': np.asarray(value) for key, value in metadata.items()})
        (self.directory / f'handoffs_{name}.json').write_text(json.dumps(self.handoffs, indent=1))
        print(f'[Run log] saved {len(self.rows)} steps and {len(self.handoffs)} handoffs as segment {name}')
        self.rows, self.handoffs = [], []
        self.segments += 1

    def close(self, **metadata):
        if self.closed:
            return
        self.closed = True
        try:
            self.flush(**metadata)
        finally:
            if sys.stdout is not self._stdout:
                sys.stdout = self._stdout
            self._console.close()
