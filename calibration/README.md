# Calibration files (examples)

Everything in this directory was measured on one physical setup and is included only
as an example of the expected files and formats. **None of it is valid for another
robot, camera or mounting: recalibrate before use.**

| File | What it is | How to produce your own |
| --- | --- | --- |
| `camera_params.npz` | Saved camera controls (exposure, gain, ...) per camera | `python scripts/cameras/tune_cameras.py` |
| `start_position.npz` | Shared start pose (SDK joint radians, gripper width in metres) | `uv run scripts/setup/set_up_recording.py` (per-task poses go to `start_positions/<task>.npz`) |
| `masking/` | Masking config, camera intrinsics and extrinsics for projected masks | See [`masking/README.md`](masking/README.md) |

Gravity masses and friction terms are not stored here: the ARX tuners
(`uv run scripts/tune.py gravity L5 can0`, then `friction`) save them in the model's
URDF under `src/hardware/arx5-sdk/models/`; see
[`docs/arm_bringup.md`](../docs/arm_bringup.md#tuning-details).
