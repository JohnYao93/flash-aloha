# Visualization

Generic episode visualization lives here. Camera tools
live under `camera_utils/`.

| File | Responsibility |
| --- | --- |
| [visualize_episode.py](visualize_episode.py) | Episode loading, Rerun display, annotations, video output, inference-chunk plots, and CLI dispatch |
| [visualization_utils/episode_visualization.py](visualization_utils/episode_visualization.py) | Inspection adapter over the shared exact-reference episode reader |
| [visualization_utils/rerun_utils.py](visualization_utils/rerun_utils.py) | Rerun robot models, camera poses, annotation overlays, and logging; also used by inference |
| [visualization_utils/training_preview.py](visualization_utils/training_preview.py) | Offline camera-resolution preview with unchanged recorded trajectories |

Helpers use `visualization.visualization_utils.<module>` with `src` on
`PYTHONPATH`. The previous helper paths remain compatibility imports of the same
module objects. The operator entry point is `scripts/visualize.py` (mode `episode`); the main
episode implementation still supports direct script paths. Examples below
assume the repository working directory for existing calibration-relative paths.

## Episode inspection

From the repository root, the operator can view recorded streams with:

```bash
.venv/bin/python scripts/visualize.py episode \
  --robot arx --data-dir /path/to/recording --rerun
```

Reading uses [episode_data.py](../data_processing/episode_data.py) and
[video_processing.py](../data_processing/video_processing.py), shared with training and inference.
Saved frame references select images exactly; viewing must not repair missing
references by choosing newer or nearest frames. Raw `--data-dir` inspection
retains its existing rejection of physical replay flags.

Without a training config, images retain their recorded resolution. The existing
explicit `--pimode` option still requests its separate OpenPI display transform.

### RAM guard and saved-file viewing

Viewing runs on the robot host, so it must never starve the collector holding
the arms. `--rerun` no longer streams into a live viewer: frames are decoded one
at a time, logged as JPEG (quality 90, display only) to
`~/.cache/flash-aloha/episode-views/<name>.rrd`, and the finished file is then
opened with `python -m rerun FILE --memory-limit ...`. `--save-rrd PATH` writes
the file without opening it; reopen it later with `.venv/bin/python -m rerun PATH`.

[memory_guard.py](visualization_utils/memory_guard.py) checks `MemAvailable`
before every logged frame (episode view and training preview). Below the floor
(25% of RAM, at least 4 GiB) the export stops with `MemoryError` and leaves no
partial file. The viewer's memory limit leaves the same floor free.
`FLASH_ALOHA_MIN_FREE_MEMORY_GB` overrides the floor; lower it only when no
robot program is running. `--save_video` still materializes the whole episode.

## Action replay

`--action_replay` replays one published episode on the arm(s) named by the task:
it homes with `reset_to_home()`, ramps for 2 s to the first row, sends one
target per row at the episode's recorded rate (which must match `--fps` when
given, and the task YAML), and homes again. The SDK interpolates each target in
its 500 Hz loop; rows are never skipped, re-timed or sent in a catch-up burst
([docs/replay_timestamps.md](../../docs/replay_timestamps.md)). `--arm_kp` and
`--arm_kd` override the gains, `--action_source raw` replays `/action_raw`, and
tracking plots are saved unless `--no_replay_graph` is given. `--fill_in` is
rejected.

Ctrl+C/SIGTERM or any catchable error while moving takes the shared failure
exit ([docs/safe_exit.md](../../docs/safe_exit.md)): native damping with the
gripper held until the operator presses `h`, then `reset_to_home()`. Ctrl+C
during damping disables in place. Native crashes, SIGKILL and power loss run no
cleanup.

This is robot operation for the operator to run, with the arm supported:

```bash
.venv/bin/python scripts/visualize.py episode \
  --task_name single_arm --episode_idx 2 --action_replay
```

### Review replay of chosen episodes

`--review_replay EP [EP ...]` runs the collector's review replay (the `r` key in
`record_episode_single.py`) on saved episodes, in the order given. It reuses
the collector's backend, camera-server cache, key watcher, gain prompt,
`replay_take` and exits unchanged; it only selects the episodes. Every
episode is loaded and checked against the task's robot contract and rate
before the camera cache or motors start. The arm then moves to the saved start
pose. For each episode, the operator:

1. enters the kp/kd tracking-gain fractions (Enter keeps the previous values,
   Esc skips this episode);
2. waits while the arm returns to the episode's recorded first row, then aligns
   the scene on the overlay of the recorded first frame over the live camera;
3. presses `1` to replay at the recorded rate (`1` again stops into damping,
   and `h` returns to the start position), `h`/`n` to skip, or `q` to finish.

Afterwards, `r` replays the same episode again, Enter moves to the next one
(cycling back to the first), and `x` finishes. A normal finish homes and
closes the controllers. Ctrl+C or any failure takes the shared failure exit.
The camera server must already be running for tasks with cameras.
`--action_replay`, `--data-dir`, `--training-config`, `--fill_in`, `--arm_kp`
and `--arm_kd` are rejected in this mode.
`scripts/run.py replay TASK EP [EP ...]` runs this mode. See
[docs/recording_review_replay.md](../../docs/recording_review_replay.md).

```bash
bash scripts/cameras/camera_servers_tmux.sh start left overhead
.venv/bin/python scripts/visualize.py episode \
  --task_name single_arm --review_replay 24 27
```

## View at a training configuration's camera resolution

```bash
.venv/bin/python scripts/visualize.py episode \
  --robot arx --data-dir data/bimanual/episode_00000.zarr \
  --training-config bimanual_flow --rerun
```

The training config supplies **only camera names and image resolution**. The
example selects 320×180. Images are resized with the shared resize operation;
there is no masking, augmentation, grayscale, normalization, curriculum or
model-tensor logging. Only one image stream per selected camera is logged.
Recorded qpos, qvel and action trajectories are displayed unchanged, at their
saved timestamps. Training action masks, offsets, target chunks and RTC do not
alter the display. No training dataset or normalization statistics are loaded.
Without a training config, the standard viewer keeps recorded image resolution.
A standard training recipe with both dimensions negative also keeps native size.

Preparation finishes before Rerun opens. Prepare a file explicitly and reopen it
without reprocessing:

```bash
.venv/bin/python scripts/visualize.py episode \
  --robot arx --data-dir data/bimanual/episode_00000.zarr \
  --training-config bimanual_flow \
  --save-rrd /tmp/bimanual_training.rrd

.venv/bin/python -m rerun /tmp/bimanual_training.rrd
```

`--prepare-only` suppresses opening the viewer and uses an automatic path under
`$XDG_CACHE_HOME/flash-aloha/training-previews` (normally `~/.cache/...`).
`--preview-frame 100` prepares just one recorded observation. Every preparation
rebuilds the artifact; open its `.rrd` directly to reuse it. Failed builds leave
previous completed files intact. The former augmentation seed, curriculum step,
and Torch thread options have been removed because training transforms no
longer run in this viewer.

Supported recipes use `training.method: standard`.
This mode rejects physical replay and conflicting display transforms. It does
not change recording, training, inference, or robot execution.

## Replay ownership

Action-replay functions live in [utils/replay_utils.py](../utils/replay_utils.py).
The episode CLI selects and validates the action stream/rate, then calls
`replay_arx_actions`. Its standalone homing, gain, positioning, tracking, and
cleanup behavior are preserved.

[recording/recording_replay.py](../recording/recording_replay.py) owns review
replay (`replay_take`), which `--review_replay` calls with the collector's
backend; replay practice uses the same replay. Review replay never appends replay motion to the original take.

The [shared utility documentation](../utils/README.md) describes these separate
lifecycle policies. Changing their scheduling or reset behavior is outside this
organizational refactor. Live robot commands remain operator-run, as required
by the [owner rules](../../AGENTS.md).

## Verification

Use synthetic recordings, and fake controllers/clocks:

```bash
PYTHONPATH=src:scripts:scripts/tests .venv/bin/python -m pytest \
  scripts/tests/test_visualize_episode_data.py \
  scripts/tests/test_training_preview.py \
  scripts/tests/test_fixed_replay.py \
  scripts/tests/test_episode_replay.py \
  scripts/tests/test_review_replay.py
```

These checks cover exact frames and visualization timestamps, shared replay
dispatch, row ordering and full nominal intervals, stop/hold controls, homing,
and imports without loading viewers or a robot SDK into recording review.
