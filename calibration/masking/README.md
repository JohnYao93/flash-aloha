# Masking configuration and calibration

`masking.yaml` is the single description of masking for this checkout. It is read by
`src/config/masking_config.py` and applied by `src/utils/masking.py`; nothing about a
mask is hard-coded. Select another file with `FLASH_ALOHA_MASKING_CONFIG=/path/to.yaml`.

## Strategies

A training recipe picks a strategy by name (`data.masking_strategy`). Each strategy is a
list of operations, applied in order:

| Operation | Meaning |
| --- | --- |
| `project: {camera, frame, arm, corners}` | A polygon given in metres, drawn in a camera through forward kinematics and the calibration. `frame` is `base`, `wrist` or `end_effector`: the polygon sits at that link with the axes of the arm base. The operation runs once per arm; `arm: self` uses that arm's link and `arm: other` the other arm's. `camera` is `overhead` or `wrist` (that arm's wrist camera). |
| `region: {camera, x, y}` | A rectangle of an image, as fractions of its width and height. `camera` is `overhead`, `wrist` or a camera name. |
| `blank: CAMERA` | Zero whole images (`wrist`, `overhead` or a camera name). |
| `zero: proprioception` | Zero the qpos; it is then not normalized. |

Corners are written for an unmirrored arm; arms with `mirror_y: true` negate y, so one
definition serves a left and a right arm. To add a strategy, add a named list under
`strategies:` (reuse operations with YAML anchors, as the shipped ones do) and name it in
the recipe. A strategy that only uses `region`, `blank` and `zero` needs no calibration.

Masks are resolution independent. Intrinsics are stored at `calibration.image_size` and
rescaled to the size of the image being masked, and regions are fractions, so the same
strategy works at any policy resolution (including a different aspect ratio from a plain
resize).

## Calibration

The `calibration` section describes one physical setup: the arm model (URDF and the links
masks can follow), the overhead camera, and each arm with its wrist camera. The state is
one block per arm in the listed order (its joints, then its gripper), so a single-arm
setup lists one arm. The shipped files belong to one bimanual ARX setup; projected masks
are only correct there. Replace them for another setup:

| File | Contents | Written by |
| --- | --- | --- |
| `<camera>_intri_imgs/<camera>_intrinsics.npy` | Pickled dict with `camera_matrix` and `dist_coeffs` at `image_size` | `python scripts/cameras/camera_int.py --camera_name NAME --width 320 --height 180` |
| `robot_<camera>_hand/<camera>_ee_to_camera.npy` | 4x4 pose of the wrist camera in its arm's end-effector frame | `python scripts/setup/calibrate_masking.py wrist-camera --arm ARM` |
| `head-to-robot_<camera>_hand/head_to_robot_base.npz` | `average_transform`: 4x4 pose of the arm base in the overhead camera frame | `python scripts/setup/calibrate_masking.py overhead --arm ARM` |

Run them in that order for each arm, with the camera servers stopped. The two
`calibrate_masking.py` steps are operator-run: the arm homes, then stays in
gravity-compensated passive mode while you guide it by hand to view a checkerboard
(`s` saves a view, `q` finishes), and returns home. Ctrl+C or any failure enters damping
until `h` homes the arm (`docs/safe_exit.md`). `--solve-only` recomputes from the saved
views without the arm or cameras.

Conventions: a point in the arm base frame maps to overhead pixels through
`K_overhead @ (T_overhead_base @ p)`, and to wrist-camera pixels through
`K_wrist @ inv(T_base_ee @ T_ee_camera) @ p`, where `T_base_ee` is the forward
kinematics of the arm's joints. Lens distortion is not applied to masks.

## Tuning the polygons (optional)

`python scripts/setup/tune_masking.py STRATEGY --episode EPISODE` shows every camera of a
recorded episode with the strategy's masks and lets you move, resize and reshape its
polygons with the keyboard (`--live` does the same on live cameras with hand-guided ARX
arms, operator-run). Saving writes a complete config beside this one
(`masking.tuned.yaml`): select it with `FLASH_ALOHA_MASKING_CONFIG`, or replace
`masking.yaml` with it. Editing `masking.yaml` by hand is equivalent.

## Checking and consistency

Check a calibration on recorded data before training with it: set `data.masking_strategy`
in a training recipe and run
`python scripts/visualize.py episode --data-dir EPISODE --training-config RECIPE`, which
shows each camera at the policy resolution with every frame's mask applied from that
frame's recorded qpos.

A checkpoint stores a contract of its mask: the strategy's operations and, for projected
masks, a digest of the calibration files and URDF. Direct inference rebuilds the contract
from its own masking config and refuses to run if it differs, so editing a strategy or
recalibrating requires retraining rather than silently changing the policy's inputs.
