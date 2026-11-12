# Task configuration

A task YAML describes one physical setup and dataset: the arms and their CAN
interfaces, the cameras, the control rate and where recordings live. Collection,
training, inference and replay all read the selected task before touching hardware.

Two examples are included:

| File | Setup |
| --- | --- |
| [`single_arm.yaml`](single_arm.yaml) | One ARX arm (left, `can0`), two cameras, 100 Hz |
| [`bimanual.yaml`](bimanual.yaml) | Two ARX arms (`can0`, `can1`), three cameras, 100 Hz |

A task has one or two arms: copy the arm block you need. Tasks are
selected by name (the file's stem) or by path (`--task-config`). To keep your own task
files outside the repository and still select them by name, list their directories in
`FLASH_ALOHA_TASK_DIRS` (separated by `:`).

## Schema

```yaml
schema_version: 1
task:
  name: my_task              # must equal the file name
control_freq: 100            # sample and action rate in Hz, for collection, training and execution
data:
  directory: ../../../data/my_task
robot:
  type: arx                  # must be arx
  arms:
    right:
      can_interface: can1
cameras:
  names: [video_overhead, video_right]
camera_settings: {}          # optional required camera controls/modes
collection:
  max_duration_seconds: 5
training:
  observations: [qpos]
  outputs: [action]
  state_dim: 7
  action_dim: 7
```

- **`control_freq`** is the one nominal rate of a task. Recordings and checkpoints must
  declare the same rate; changing the YAML does not resample them. The ARX SDK's
  500 Hz motor loop and the camera frame rate are separate. `control_freq` may not
  exceed 500 Hz.
- **`data.directory`** is relative to the YAML. Paths under the repository's `data/`
  follow `FLASH_ALOHA_DATA_DIR` (see `src/config/paths.py`).
- **`robot.arms`** has a `left` and/or `right` slot, each with its own CAN interface.
  An arm may name a `model` (`L5`, the default, or `Trossen_L5`) and may set
  `command_timeout_s` and `max_feedback_age_s` (the shared hardware runtime's command
  watchdog and feedback-age limit, in seconds). The shared hardware runtime
  (`--shared-hardware`, `record_hardware.py`) requires `model` to be written out; the
  hand-guided recorder supports `L5` only.
- **`cameras.names`** fixes the camera order; a checkpoint with another order is
  rejected rather than rearranged. An empty list is a camera-less task.
- **`collection.max_duration_seconds`** is the longest take, in seconds.
- **`training`** lists the observation and output names and the state/action
  dimensions (7 per arm: six joints and the gripper). Dimensions are not inferred from
  the arm count.

Defaults come from `../task_defaults.yaml`, then the task file, then explicit overrides.
Mappings merge by key; lists and scalars replace. Unknown keys are rejected.

The predicted chunk length is not a task setting: it is `model.chunk_size` in the
[training config](../training/README.md). Execution windows, blending, RTC and
recording choices belong to [inference configs](../inference/README.md). Image size and
preprocessing belong to the checkpoint ([image preprocessing](../../../docs/image_preprocessing.md)).

## Camera settings

`camera_settings: {}` adds no requirement. A non-empty mapping can require per-camera
saved controls and a `[width, height, camera_fps]` mode. Collection snapshots the saved
camera settings into each episode; training carries them into checkpoints and
inference compares them with its local settings. See
[saved camera settings](../../../docs/camera_settings_contract.md).

## Checking a task offline

```bash
python src/config/check_task_config.py src/config/tasks/single_arm.yaml
python src/config/check_task_config.py src/config/tasks/single_arm.yaml --set control_freq=50
```

The first prints the fully resolved task, including derived step counts; `--set`
previews an override without editing the file. Add `--recording /path/to/episode.zarr`
(repeatable) and/or `--checkpoint-config /path/to/all_configs.json` to check saved
recordings and a checkpoint against the task: rate, camera order and settings,
dimensions and the robot identity must agree. Create a new task from an example with
`python scripts/config/create_config.py task NAME --template single_arm`.
