# Camera scripts

Camera maintenance and preview implementations live here.
The shared camera server, readers, cache and timing/settings contracts live in
[`src/camera_utils/`](../../src/camera_utils/README.md).

| Script | Purpose |
| --- | --- |
| `camera_servers_tmux.sh` | Start, inspect and stop the per-camera server processes |
| `configure_camera_controls.sh` | Disable dynamic exposure framerate on left, right and overhead cameras |
| `tune_cameras.py` | Tune controls interactively and save camera settings |
| `camera_int.py` | Calibrate camera intrinsics from checkerboard images |
| `livestream_camera.py` | Preview frames from a running camera server |

Run commands from the repository root. For example:

```bash
./scripts/cameras/camera_servers_tmux.sh status
.venv/bin/python scripts/cameras/tune_cameras.py --help
.venv/bin/python src/camera_utils/camera_server.py --help
```

The server manager is distinct from `configure_camera_controls.sh`, which
changes device controls. There is one server implementation and one manager.

The controls script no longer cycles auto white balance or auto exposure.

The shared RobotMQ cache launches its worker directly from `src/camera_utils`.
It has no operator script. Its helper reads cameras in parallel into shared-memory
rings and reports frame IDs and driver/consumer timestamps, keeping camera waits
out of robot loops.

Tuning, calibration and recording destinations keep their existing meanings.
The intrinsic calibration and controls scripts retain their repository-root
working-directory convention. Start camera capture or change device controls
only when you intend to operate the cameras.

## CPU wake-up latency (prevents dropped USB frame bursts)

At 100 fps a laptop that lets its CPUs sleep in deep idle states (C6-C10,
~170-230 us to wake) can occasionally service the UVC camera too late; the
driver then drops a burst of frames and recording fails with a stale-frame
error (`frame age ... exceeds 50 ms`). Each camera server therefore holds a
PM QoS limit of 20 us through `/dev/cpu_dma_latency` while it runs
(`--cpu-latency-us`, negative disables). The kernel releases it when the server
exits. Capture, timestamps and timing validation are unchanged.

One-time permission setup (otherwise the server prints a warning and runs
without the limit):

```bash
sudo groupadd -f cpulatency
sudo usermod -aG cpulatency "$USER"
echo 'KERNEL=="cpu_dma_latency", GROUP="cpulatency", MODE="0660"' | \
  sudo tee /etc/udev/rules.d/99-flash-aloha-cpu-latency.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --name-match=cpu_dma_latency
```

Log out and back in (and restart tmux with `tmux kill-server`) so the new group
applies, then restart the camera servers. They should print
`[CPU] Wake-up latency limited to 20 us while this camera streams`.
