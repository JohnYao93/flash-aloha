# Flash ALOHA: Learning Fast and Precise Manipulation

Project website: johnyao93.github.io/flash-aloha

Data collection, training and deployment for bimanual and single-arm manipulation
with ARX arms (L5 and Trossen_L5): camera servers, hand-guided recording, ACT
and Flow policy training, direct and remote inference, and visualization.

Two policies are included: decoder-only ACT (the default) and Flow, which uses the
LeRobot U-Net/visual backbone. Both support training-time RTC prefix conditioning
and optional image/proprioceptive masking. The remote client can also execute an
OpenPI server's policy (for example Pi0.5); see `docs/image_preprocessing.md`.

## Quick start

The steps below are each described in their own section. Commands that move a robot
are marked operator-run: read the section before running them.

```bash
# 1. Install and run the offline tests (no robot needed)
uv sync --extra dev && source .venv/bin/activate && source scripts/setup/paths.sh
python -m pytest

# 2. Connect an arm, then tune its gravity and friction compensation
uv run scripts/tune.py gravity L5 can0              # MODEL INTERFACE
uv run scripts/tune.py friction L5 can0

# 3. Practice: move the arm by hand, then record a motion and replay it (no cameras, nothing saved)
uv run scripts/practice.py --arm right
uv run src/recording/replay_practice.py --arms right

# 4. Save a start pose and record demonstrations
uv run scripts/setup/set_up_recording.py --task_name single_arm
python scripts/run.py collect single_arm

# 5. Train, then run the policy
python scripts/run.py train single_arm_act
python scripts/run.py infer single_arm_act
```

`single_arm` (one ARX arm, two cameras) and `bimanual` (two ARX arms, three cameras)
are the example tasks in `src/config/tasks/`; copy one to describe your own setup.

## Setup

Run commands from the repository root. Once per terminal:

```bash
uv sync --extra dev
source .venv/bin/activate
source scripts/setup/paths.sh
python -m pytest                    # offline: synthetic observations, simulated hardware
```

The tests never start a camera or a robot controller. `.python-version` selects
Python 3.10.

Build the ARX SDK's Python extension on the robot computer following
[src/hardware/arx5-sdk/README.md](src/hardware/arx5-sdk/README.md) (its Conda environment
supplies the native dependencies). `paths.sh` expects that environment at
`$HOME/miniconda3/envs/arx-py310`; set `ARX_CONDA_PREFIX` if yours is elsewhere, and
`OROCOS_KDL_PYTHON` if you built PyKDL from source.

The files under `calibration/` were measured on one physical setup and are examples
only: recalibrate for your own arms and cameras before use
([calibration/README.md](calibration/README.md)).

Datasets, checkpoints and logs default to `data/`, `ckpts/` and `logs/` in the
repository root and are not part of the repository. To keep them elsewhere, for
example on a larger disk, set any of:

```bash
export FLASH_ALOHA_DATA_DIR=/path/to/datasets
export FLASH_ALOHA_CKPT_DIR=/path/to/checkpoints
export FLASH_ALOHA_LOG_DIR=/path/to/logs
export WANDB_DIR=/path/to/wandb          # Weights & Biases' own setting
```

Every task, training and inference config follows these settings
(`src/config/paths.py`); the YAML files keep their repository-relative paths.


## Connecting an arm (operator-run)

Connect the arm's power and USB-CAN adapter, with arm controllers stopped. Most
adapters appear as a native CAN interface; bring it up at 1 Mbit/s:

```bash
ip -br link                                    # find the interface, commonly can0
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

Then match `robot.arms.<left|right>.can_interface` in your task YAML to the adapter.
Tasks use `robot.type: arx`; each arm may name its `model` (`L5`, the default, or
`Trossen_L5`). Start from [single_arm.yaml](src/config/tasks/single_arm.yaml) for one
arm or [bimanual.yaml](src/config/tasks/bimanual.yaml) for two. Set
`FLASH_ALOHA_TASK_DIRS` to keep your own task YAMLs outside the repository and still
select them by name.

Check that the built SDK extension imports before moving anything:

```bash
python -c "import arx5_interface; print(arx5_interface.__file__)"
```

[docs/arm_bringup.md](docs/arm_bringup.md) has the full procedure: SLCAN and EtherCAT
adapters, persistent device names and troubleshooting. It also lists the ARX utility
scripts: `arx_setup.sh`, `scripts/setup/arx_arms.sh` and `scripts/setup/activate_arm`
shorten the ARX setup described in the
[ARX SDK README](src/hardware/arx5-sdk/README.md), and `src/hardware/check_gain.py`
prints the SDK's default gains.

## Tuning gravity and friction compensation (operator-run)

This is the next step after setup, once per physical arm and before practice,
recording or inference: hand-guided collection relies on the saved compensation.
`scripts/tune.py` runs the ARX SDK tuners, which take the SDK model (`L5` or
`Trossen_L5`) and the CAN interface. Tune gravity first, then friction.

```bash
uv run scripts/tune.py gravity L5 can0         # link masses: the arm should float where you leave it
uv run scripts/tune.py friction L5 can0        # gripper and joint friction feedforward
```

Repeat with the other interface (`can1`) for a second arm. Every other option passes
to the tuner unchanged (`scripts/tune.py gravity --help`).

Both tuners enable the motors with zero position gain: support the arm before starting
and before exiting. `s` saves the link masses (gravity) or the friction terms into the
model's URDF in the SDK, which the SDK loads for that model. Controls are in
[docs/arm_bringup.md](docs/arm_bringup.md#tuning-details).

## Practice, record/replay and start poses (operator-run)

### Practice without cameras

Run gravity-compensated manual practice with both arms, or just one:

```bash
uv run scripts/practice.py --arm both
uv run scripts/practice.py --arm right
```

Left uses `can0`, right uses `can1`; optional `--can-interface` overrides this
mapping, and `--arx-model` selects `L5` (default) or `Trossen_L5`. Arm and gripper
gains are zero and the SDK provides gravity compensation. The Python loop only reads
the arm state, 200 times per second by default (`--controller-fps`).
Support the arms during startup and shutdown. Ctrl+C (or any failure) enters
damping; press `h` to return home, after which the controllers close (see
[docs/safe_exit.md](docs/safe_exit.md)). A finished `--duration` closes without
homing. This entry point does not require cameras.

### Record and replay a motion without cameras

After practising, check that the arm reproduces a motion you guide it through,
before saving a start pose or recording demonstrations. This tool uses no task
and no cameras, has no time limit and writes nothing to disk. The take stays in
memory until the next recording replaces it.

```bash
uv run src/recording/replay_practice.py --arms right
uv run src/recording/replay_practice.py --arms both
```

The arms go passive where they are so you can move them by hand. The keys are
typed in the terminal or sent by a three-pedal foot switch:

| State | Keys |
| --- | --- |
| Idle (arms passive) | `3` starts recording, `q` quits |
| Recording | `3` stops |
| Review (arms hold) | `2` replays, `3` records again, `1` releases the arms |
| Replay | the arms move to the first recorded pose; `1` plays, `1` again stops into damping and `h` returns the arms to a hold; `h`/`n` cancels before playing |

Record, replay, move the arms somewhere else and repeat as often as you like.
The replay runs at the recorded rate with active joint gains
([tunable](#tuning-the-replay-gains)), so keep clear of the arms while it moves. `--fps` sets the sample and replay rate
(default 100), `--can-left`/`--can-right` override the CAN interfaces,
`--start-position [PATH]` first moves to a saved start pose
(`calibration/start_position.npz` when no path is given) before releasing the
arms, and `--move-duration` sets the seconds taken to reach that pose and the
first replay pose (default 3). Support the arms at startup and when quitting:
`q` homes them and closes the controllers.

### Tuning the replay gains

Replay practice has two numbers for how hard the arm tracks the recorded
motion: one multiplies the default joint kp (stiffness), the other the default
joint kd (damping). Both start at 1.00 and move in steps of 0.05, between 0.10
and 2.00:

| Number | Up | Down |
| --- | --- | --- |
| kp | up arrow or `w` | down arrow or `s` |
| kd | right arrow or `d` | left arrow or `a` |

Every change prints the two numbers and the joint gains they give:

```text
Replay gains: kp x0.80, kd x1.20  (joints 1-3 kp 160 kd 8.4; joints 4-6 kp 80 kd 1.8)
```

Press the keys between replays; they apply from the next replay and are
ignored while recording or replaying. A loop that works well:

1. Record a motion with the fast moves and stops you care about (`3`, `3`).
2. Replay it (`2`, then `1` once the arm is at the first pose) and watch.
3. If the arm lags or sags behind the motion, raise kp. If it buzzes or
   overshoots, lower kp or raise kd. If it feels sluggish or the motors
   whine, lower kd.
4. Replay the same take again (`2`) and repeat, changing one number at a time.

Move in small steps and keep clear of the arm: high kp with low kd can
oscillate. `--kp-scale` and `--kd-scale` set the starting numbers, and the
last ones are printed on exit so you can start from them next time. The
numbers are not saved and only affect this tool: recording and inference keep
the default gains. The replay itself is the collector's review replay, so the
same fractions scale the gripper gains too, and a failure or Ctrl+C enters
damping until `h`.

#### Graphing commanded and achieved position

Add `--graph` to see how closely the arm followed each replay:

```bash
uv run src/recording/replay_practice.py --arms right --graph
```

After every replay that played (completed or stopped with `1`), a window opens
with one panel per joint and the gripper, and one column per arm. Each panel
plots the commanded position and the achieved (measured) position against time,
with the RMS and largest error in its title and the kp and kd numbers at the
top. This is helpful for visualizing whether predicted trajectories are being
followed sufficiently: a replay sends its rows the way policy execution sends
a policy's predicted actions, so the gap between the two lines here is the gap
to expect when the arm runs a policy with these gains. A good result has the
achieved line lying on the commanded one, trailing it only slightly, without
overshoot at stops or ripple. A steady offset or growing lag calls for more
kp; overshoot or ripple calls for less kp or more kd.

The achieved position is read right after each target is sent, so it always
trails the commanded line by about one sample. Windows stay open so you can
compare replays at different gains; close them yourself, or they close when
the tool quits. The graph is drawn by a separate process from the tracking log
the replay already writes under `logs/replay/`, and is also saved there as
`tracking.png`. To draw a graph for an earlier replay log:

```bash
uv run src/visualization/replay_tracking_graph.py logs/replay/<date>/<time>_practice_take
```

### Camera preview and saved start poses

To save a start pose for one task, pass its name; the task YAML supplies the
arm(s), CAN interfaces and cameras:

```bash
uv run scripts/setup/set_up_recording.py --task_name single_arm
```

This saves `calibration/start_positions/single_arm.npz`. Recording
(`scripts/record.py single`) and inference for that task then start from it,
including the saved gripper value; tasks without a saved pose keep using the
shared `calibration/start_position.npz`. Flags that contradict the task YAML
(`--arm` or CAN overrides) are rejected; `--cameras` and `--output` may still be
overridden. With `--gripper-only`, a task without its own pose starts from the
shared pose.

Without a task, select the hardware directly (`--arm left|right|both`). In the
UI, Space locks the measured arm and gripper pose, `y` saves it, `n` returns to
practice and `q` exits. Options are in
[docs/arm_bringup.md](docs/arm_bringup.md#saving-start-poses).

## Configuration and training

Task YAML owns cameras, arms/CAN interfaces, action rate and prediction horizon.
Training YAML owns model/training settings. Inference YAML owns execution
windows, blending, RTC and recording choices. The same shared contracts enforce
consistency between collection, training and deployment.

### How the configs, contracts and AGENTS.md fit together

Each stage has one YAML under `src/config/`, selected by name, and each later
stage names the earlier one instead of repeating its values:

| Stage | Config | Owns | Refers to |
| --- | --- | --- | --- |
| Collection | `tasks/NAME.yaml` | arms and CAN interfaces, camera names and order, `control_freq`, data directory, state/action dimensions | `task_defaults.yaml` for defaults |
| Training | `training/NAME.yaml` | model type, `model.chunk_size`, optimizer and data options, training-time RTC, masking | `task:` names the task |
| Inference | `inference/NAME.yaml` | executed chunk steps, blending, runtime RTC, rollout recording, direct or remote backend | `task:` names the task; `checkpoint:` points at the checkpoint's `all_configs.json` |

What a stage decides travels forward as metadata, and the next stage checks it
instead of trusting its own YAML. Each episode stores the task's rate, robot
identity and camera settings. Training copies those, plus its own model, RTC,
masking and image-preprocessing settings, into the checkpoint's
`all_configs.json`. Inference loads that file and refuses to run if the task
or the profile disagrees with it, so an inference profile cannot override the
rate, cameras, dimensions or physical setup. Check any link offline with
`check_task_config.py --recording ... --checkpoint-config ...`,
`check_training_config.py` and `check_inference_config.py`
([src/config/README.md](src/config/README.md)).

The checks themselves live in `src/contracts/`, shared by every stage so that
collection, training and inference cannot each keep their own copy:

| Contract | Guarantees | Used by |
| --- | --- | --- |
| `standard_capture.py` | one proprio read, then one camera snapshot, with exact frame IDs and timestamps | live recording and inference |
| `capture_gc.py` | no automatic garbage collection while capturing or streaming | recorder, camera servers, inference loop |
| `standard_recording.py` | streams start before the first sample and drain on exit; the captured observation is stored as-is | every live recorder |
| `standard_image_processing.py` | the same named, deterministic image transforms | training and inference |
| `action_contract.py` | absolute action targets only | recording, training statistics, replay, inference |
| `action_rate_contract.py` | one declared action rate per dataset and checkpoint | export, training, checkpoint loading, execution |
| `sequence_contract.py` | no training sequence spans a recording gap | training data loading |
| `robot_contract.py` | the arm models and left/right roles match | task, episodes, checkpoint, replay |

Details and the full rules are in [src/contracts/README.md](src/contracts/README.md)
and [docs/collection_inference_contracts.md](docs/collection_inference_contracts.md).

[AGENTS.md](AGENTS.md) makes these contracts binding on coding agents (and is a
good checklist for human contributors). It does not restate the code; each
section points an agent at the module it must reuse and the test to run:

| AGENTS.md section | Requires | Verified by |
| --- | --- | --- |
| Robot operation | agents hand robot commands to the operator and never start a controller, collector or replay | — |
| Mandatory observation and recording contract | new recorders and inference entry points reuse `standard_capture.capture_observation` and record through `standard_recording` | `scripts/tests/test_standard_capture.py` |
| Mandatory garbage-collection pause during capture | GC control only through `capture_gc` | `scripts/tests/test_standard_capture.py` |
| Mandatory policy image preprocessing | train and deploy share one `standard_image_processing` profile; an OpenPI server must advertise the matching image contract | `scripts/tests/test_standard_image_processing.py` |
| Protected recording and hardware code | no change that alters capture, timestamps, recording, drivers, action execution, RTC, chunks/rates/masking, gripper conversion or homing; storage locations change only through `src/config/paths.py` | the contract tests above |

The action, action-rate, sequence and robot contracts have no section of their
own in AGENTS.md; they fall under the protected-code rule.

```bash
python scripts/config/check_training_config.py --training-config single_arm_act
python scripts/config/check_inference_config.py single_arm_act --schema-only
python scripts/config/check_inference_config.py single_arm_remote --schema-only
python scripts/policy/imitate_episodes_train.py --training-config single_arm_act
```

`single_arm_act` is the single-arm ACT example; `bimanual_act` trains both
arms. The `*_flow` recipes train Flow instead
([src/config/training/README.md](src/config/training/README.md)). Point the
task/training configuration at your recordings. A recipe can also train with an
image or proprioceptive mask (`data.masking_strategy`), which direct inference then
applies too; see [docs/image_preprocessing.md](docs/image_preprocessing.md#masking). For new
configs, use `scripts/config/create_config.py --help`. A checkpoint
directory holds `all_configs.json`, `dataset_stats.pkl` and the weights; copy the
whole directory when moving a checkpoint to another machine.

### Training-time RTC

Both ACT and Flow support training-time real-time chunking (RTC). Set
`training.training_time_rtc: true` in the training recipe (the `*_flow` recipes
already do; the `*_act` recipes leave it off). The prefix length defaults to 10
actions (`--training_time_rtc_max_delay`). The model then learns to continue a committed action prefix: at inference
each new chunk is conditioned on the previous plan's next actions, so chunks join
without a jump while the policy query runs in the background. RTC is off by
default at inference, where chunk blending is used instead; enable it with
`rtc.training_time: true` and `rtc.enabled: true` in the inference YAML. The
checkpoint must have been trained with it. Details are in
[src/config/training/README.md](src/config/training/README.md) and
[src/config/inference/README.md](src/config/inference/README.md).

## One command per workflow

`scripts/run.py` resolves task and training-config names into the full command
for `scripts/record.py`, `scripts/infer.py`, `scripts/visualize.py` or training.
Add `--dry-run` to print that command without starting it; other options pass
through to the entry point ([src/config/README.md](src/config/README.md)).

```bash
python scripts/run.py collect single_arm
python scripts/run.py train single_arm_act
python scripts/run.py infer single_arm_act
python scripts/run.py serve single_arm_flow --device cuda:0
python scripts/run.py infer single_arm_flow --inference-config single_arm_remote --remote_host SERVER_HOST
python scripts/run.py view single_arm 24
python scripts/run.py replay single_arm 24 27
```

## Inference (operator-run)

These commands can operate the robot; run them yourself after setup.
Direct inference loads the ACT or Flow checkpoint locally:

```bash
python scripts/infer.py direct --inference-config single_arm_act
```

Direct and remote rollouts move to the task's saved start pose first, including
its saved gripper value, so they start where the recorded takes started.

For server/client operation, start the model server on the inference machine:

```bash
python scripts/infer.py serve \
  --config ckpts/single_arm_flow/all_configs.json --device cuda:0 --port 8000
```

Run the client on the robot host:

```bash
python scripts/infer.py remote \
  --inference-config single_arm_remote --remote_host SERVER_HOST
```

The transport uses a NumPy/MessagePack WebSocket protocol; no OpenPI checkout is
required for this repository's own server. An offline client run can use
`--dry-run --dry_run_dataset_dir /path/to/raw/recordings` against the model server.
The server consumes native recorded frames and applies the shared image
preprocessing at checkpoint resolution. The same client runs an OpenPI server's
policy (profiles `*_openpi_remote`). This repository does not export data for or
train OpenPI; `docs/image_preprocessing.md` describes the image contract that
server must advertise.

By default, direct inference crossfades each new chunk into the previous one with
sigmoid blending (`blending.method: sigmoid` in every direct profile), and RTC is
off (`rtc.enabled: false`). Turning RTC on keeps the crossfade, applied right after
the pinned prefix.

Direct inference supports linear, exponential and sigmoid chunk blending,
`--mixing_steps`, optional nearest-neighbor chunk matching, and temporal
aggregation. `--action_lookahead` and `--gripper_lookahead` skip the start of
each new chunk; see [tuning the lookahead](#tuning-the-action-and-gripper-lookahead).
Prediction horizons must cover the execution window, offsets and blending tail.

### Tuning the action and gripper lookahead

A chunk's first action describes the moment its images were captured. By the
time the policy has answered and the arm has caught up with its last target,
that moment has passed, so executing the chunk from its first action pulls the
arm back. Direct inference without RTC therefore skips the start of every chunk
after the first:

| Flag | Default | Effect |
| --- | --- | --- |
| `--action_lookahead N` | 10 | at each chunk boundary, the new chunk starts at its action `N` instead of action 0 |
| `--gripper_lookahead M` | 5 | the gripper is read a further `M` actions ahead, at `N + M` |

Both are counted in actions, so at 100 Hz the defaults are 100 ms for the arm and
150 ms for the gripper. They are
command-line options, passed through by `run.py`:

```bash
python scripts/run.py infer single_arm_act --action_lookahead 12 --gripper_lookahead 5
```

How the skip fits the rest of the execution:

- **Chunk boundary.** The policy is queried at each boundary (`execution.chunk_steps`
  after the previous one). The new chunk replaces the old one from its action `N`,
  and the following `chunk_steps` actions are executed.
- **Blending.** With `blending.method` set, the first `blending.steps` executed
  actions, starting at action `N`, are blended with the previous chunk's
  continuation past its own executed window. The lookahead moves where the blend
  starts; it does not shorten it.
- **Running out of actions.** A profile is rejected before the robot moves unless
  `max(chunk_steps, first_chunk_steps) + N + M` fits in the checkpoint's
  `chunk_size` (this check counts `N` as at least 10). With the 100-action example
  checkpoint and 50-step chunks, `N + M` can be at most 50.
- **First chunk.** The first chunk of a rollout, and the first after a reset, has
  no chunk boundary before it, so it always runs from action 0. Only the gripper
  offset `M` applies to it.

You can change both numbers during a rollout, while it is paused (`c`, `p` or
the go pedal freezes the arm in place):

| Number | Up | Down |
| --- | --- | --- |
| action lookahead | up arrow or `w` | down arrow or `s` |
| gripper lookahead | right arrow or `d` | left arrow or `a` |

Each press moves one action and prints both numbers with their time:

```text
[Lookahead] action 12 (120 ms), gripper +5 (170 ms); applies when the rollout continues
```

A value below 0, or one that would run past the end of a chunk, is refused and
the reason is printed. When you continue, the gripper offset applies from the
next step and the action lookahead from the next chunk boundary. The keys do
nothing while the policy is moving the arm. The numbers are not saved: put the
ones you settle on in the command line.

To tune, run a rollout, pause it to change a number, continue, and read the line
printed at every boundary (the numbers here are only an example of the format):

```text
[Direct lookahead] handoff step=150: target jump 0.0041 (previous step 0.0038); new vs old plan
over next 10: max 0.0120, -0.0065 along motion (behind) of 0.0410; arm trailed its target ...
```

`target jump` should be close to `previous step`. If the new plan is repeatedly
`(behind)` and the arm hesitates or steps back at boundaries, raise
`--action_lookahead`; if it is repeatedly `(ahead)` and the arm lurches forward,
lower it. Change it one or two actions at a time. Then set `--gripper_lookahead`
so the gripper closes when the arm arrives: raise it if the gripper closes late,
lower it if it closes early.

The two flags apply to direct inference without RTC only. The asynchronous modes
skip the start of a chunk in their own way and reject these flags or do not have
them: direct RTC executes the pinned prefix while the query runs and takes over
after it, and the remote client starts each chunk at `execution.overlap_steps`.
Neither has a separate gripper offset.

Remote RTC uses an asynchronous fixed-overlap handoff and hard action
prefix conditioning. Set `execution.overlap_steps`, `rtc.prefix_steps` and
`rtc.enabled` in its inference YAML; the checkpoint must declare compatible
training-time RTC. The configured overlap is the new chunk's start offset;
measured delay is checked against the overlap budget. Overruns fail closed.
Remote inference requires `blending.method: none`. Direct inference with RTC is
described in [src/config/inference/README.md](src/config/inference/README.md).

## Capture, recording, diagnostics and visualization

Operator entry points are `scripts/record.py`, `scripts/visualize.py` and those under `scripts/cameras` and
`scripts/setup`. Recording uses `capture_observation`, `DistributedRecorder`
and `RawEpisodeBuffer`; encoding stays in camera-server processes. Sampled
frame IDs, camera/consumer timestamps, capture ordering and retained-frame
references are preserved. Missing or invalid references fail closed.

For hand-guided demonstrations with the `single_arm` task (left/can0,
`video_overhead` and `video_left`, 100 Hz, up to 15 seconds), close the camera
tuner and run:

```bash
bash scripts/cameras/camera_servers_tmux.sh start left overhead
uv run scripts/record.py single --task_name single_arm
```

Start the camera server only if it is not already running. The recorder moves to the
task's start pose, including the saved gripper value, and holds it; `1` starts a take
(the arm and gripper go passive so you can guide them), `1` stops it and the arm holds
the measured pose, then `1` saves or `h`/`n` discards, `r` replays the take before
deciding, and `q` quits. These keys match the two-pedal foot switch. Takes are started
and stopped with the `1` key or pedal only. `--guided-start` starts each take wherever
you place the arm, and `--record-replay` also records a replay of each saved take.
Ctrl+C or any failure enters damping until `h` homes the arm
([docs/safe_exit.md](docs/safe_exit.md)). The full walkthrough is in
[docs/hand_guided_recording.md](docs/hand_guided_recording.md); review replay is in
[docs/recording_review_replay.md](docs/recording_review_replay.md).

```bash
python scripts/record.py single --help
python scripts/diagnostics/write_recording_rate_metadata.py --help
python scripts/visualize.py episode --data-dir /path/to/raw/episode --save-rrd episode.rrd
```

Recording review/replay, generic DAgger collection, raw-episode training,
training-resolution previews and Rerun visualization are available. Robot replay and setup are operator-run too.

See `docs/recording_timestamps.md`, `docs/image_preprocessing.md`,
`docs/camera_settings_contract.md`, `docs/replay_timestamps.md` and
`docs/episode_storage.md` for contract details. See `AGENTS.md` for the rules
coding agents follow in this repository.

## Masking (optional)

A policy can be trained with parts of its inputs hidden: image regions that follow the
arms, fixed image regions, whole cameras, or the proprioception. Set
`data.masking_strategy` in a training recipe; direct inference then applies the same
mask. Strategies are defined as data in `calibration/masking/masking.yaml`, not code,
and work at any image resolution
([docs/image_preprocessing.md](docs/image_preprocessing.md#masking)).

Masks that follow the arms need a calibration of your cameras and arms
([calibration/masking/README.md](calibration/masking/README.md)):

```bash
python scripts/cameras/camera_int.py --camera_name video_overhead     # intrinsics, per camera
python scripts/setup/calibrate_masking.py wrist-camera --arm left      # operator-run
python scripts/setup/calibrate_masking.py overhead --arm left          # operator-run
```

Tuning the mask shapes is optional. `scripts/setup/tune_masking.py` shows the masks on
a recorded episode (or on live cameras with hand-guided ARX arms) while you move, resize
and reshape a strategy's polygons, and saves a tuned config:

```bash
python scripts/setup/tune_masking.py end_effector --episode data/TASK/episode_00000.zarr
python scripts/setup/tune_masking.py end_effector --live               # operator-run
```

## Documentation

| Topic | Where |
| --- | --- |
| Arm bring-up, tuning details, start poses | [docs/arm_bringup.md](docs/arm_bringup.md) |
| Failure handling (damping, homing) | [docs/safe_exit.md](docs/safe_exit.md) |
| Hand-guided recording, review and replay | [docs/hand_guided_recording.md](docs/hand_guided_recording.md), [docs/recording_review_replay.md](docs/recording_review_replay.md) |
| Inference layout and the shared hardware runtime | [docs/inference_layout.md](docs/inference_layout.md) |
| Task, training and inference configs | [src/config/README.md](src/config/README.md) and the READMEs beside each config folder |
| Contracts shared by collection, training and inference | [src/contracts/README.md](src/contracts/README.md), [docs/collection_inference_contracts.md](docs/collection_inference_contracts.md) |
| Image preprocessing and masking | [docs/image_preprocessing.md](docs/image_preprocessing.md) |
| Recording format, timestamps and storage | [docs/recording_timestamps.md](docs/recording_timestamps.md), [docs/episode_storage.md](docs/episode_storage.md), [docs/raw_training.md](docs/raw_training.md) |
| Cameras | [scripts/cameras/README.md](scripts/cameras/README.md), [docs/camera_settings_contract.md](docs/camera_settings_contract.md) |

## Licence and vendored code

This repository is released under the MIT License ([LICENSE](LICENSE)). Two
components are included with local modifications under their own licences, which
continue to apply to those directories. Please cite and refer to the original projects:

| Directory | Original project | Licence |
| --- | --- | --- |
| `src/hardware/arx5-sdk` | [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk) (ARX5 arm SDK; base commit `79240ef`, changes listed in [CHANGES_FROM_UPSTREAM.md](src/hardware/arx5-sdk/CHANGES_FROM_UPSTREAM.md)) | MIT, `src/hardware/arx5-sdk/LICENSE` |
| `src/kdl_parser` | [ros/kdl_parser](https://github.com/ros/kdl_parser) (URDF to KDL tree parser, C++ and Python) | BSD 3-Clause, `src/kdl_parser/LICENSE` |
