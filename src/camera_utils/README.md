# Camera utilities

This package owns the camera server, intrinsic calibration, shared-memory readers,
parallel consumer caches, and clock/settings contracts. Operator tools and their
implementations live in [`scripts/cameras/`](../../scripts/cameras/README.md):
server management, tuning, previews and controls.
`capture.py` contains the camera infrastructure formerly inside `training_utils.py`;
`sources.py` contains the shared task-preview camera source.

Live callers still use `standard_capture.capture_observation`. Its snapshot
provider now imports `camera_utils.capture`; capture order, frame selection,
clock provenance, pixel copies, recording fields, and failure checks are unchanged.
Importing camera readers no longer imports training, Torch, or robot SDK modules.

Example operator entry points, run from the repository root:

```bash
./scripts/cameras/camera_servers_tmux.sh start
.venv/bin/python src/camera_utils/camera_server.py --help
.venv/bin/python scripts/cameras/livestream_camera.py --help
```

Use `from camera_utils.capture import acquire_camera_readers` for reader leases.
Camera state has one owner, `camera_utils.capture`; it is not re-exported through
training utilities. Physical camera settings and calibration artifacts remain at
their existing paths.

`camera_server.py` and `robotmq_camera_cache.py` are importable implementations.
Run the server directly from `src/camera_utils/camera_server.py`; the tmux
manager uses that path. The cache starts its worker directly from
`src/camera_utils/robotmq_camera_cache.py`. The intrinsic calibration
command also retains its thin `scripts/cameras/camera_int.py` launcher.
