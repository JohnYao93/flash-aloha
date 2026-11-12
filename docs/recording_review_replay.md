# Review replay (operator-run)

Review replay plays recorded rows back on the arm so the operator can check a
take. It is the same procedure in three places:

| Where | What is replayed |
| --- | --- |
| `r` at the collector's save/discard prompt | the stopped take, before it is saved or discarded |
| `r` at the collector's next-episode prompt | the last episode saved in this session (otherwise the newest `episode_*.zarr`) |
| `scripts/visualize.py episode --review_replay EP [EP ...]`, or `scripts/run.py replay TASK EP [EP ...]` | the listed saved episodes, in order |

```bash
uv run scripts/run.py replay single_arm 24 27
```

Replay never edits data. It records only with `--record-replay`, which saves the
replay as a new episode (see
[hand-guided recording](hand_guided_recording.md#recording-a-replay-of-each-take)).

## Procedure

1. **Gain prompt.** Each replay first asks for two tracking-gain fractions, kp
   then kd, applied to every joint and the gripper for the replayed rows only
   (0 < kp <= 2, 0 <= kd <= 2). Enter keeps the previous values (1 and 1 at
   start-up); Esc cancels the replay before anything moves.
2. **Return to the recorded first pose.** The arm makes the saved-pose move to
   the first recorded row and holds it with the collection tracking gains.
3. **Alignment overlay.** A window shows the recorded first frame blended over
   the live image of the overhead camera if one is recorded, otherwise the
   wrist camera. The window is kept above other windows
   (`src/utils/window_utils.py`) without taking focus from the terminal. A task
   with no cameras (`cameras.names: []`) has no overlay and reads the same keys
   from the terminal. Reset the scene, then press `1` (left pedal) to start,
   `h`/`n` (right pedal) to cancel, or `q` to quit.
4. **Replay.** Rows play at the recorded rate like policy execution: one
   timestamped target per arm and tick, interpolated by the SDK, with no skipped
   rows or catch-up bursts. `1` stops early: the arm enters native damping
   (kp = 0, gripper held) and `h` returns it to the start position. A completed
   replay holds its final pose.

The move to the first pose, recording, damping and homing always use the
unscaled gains, which are restored when the replay completes or stops. Ctrl+C
or an error while moving takes the failure exit instead: damping until `h`,
then home and exit ([safe_exit.md](safe_exit.md)). Keys still queued after a
replay are discarded so they cannot answer the next prompt.

## In the collector

At the save/discard prompt the stopped take is already on disk as its raw
capture: rows are the take's samples and the overlay frame is decoded from the
raw video through the first sample's exact frame reference. Nothing is published
until `1` saves; `h`/`n` still discards. `r` can be repeated. A take whose raw
streams cannot be read back is reported and stays in review. Replay does not
modify the take or change its task labels, camera references, timestamps or
save decision.

A saved episode's robot contract and rate must match the task before it is
replayed from the next-episode prompt.

## Saved episodes from the viewer

`--review_replay EP [EP ...]` reuses the collector's backend, camera cache, key
watcher, gain prompt and exits unchanged; it only selects the episodes. Every
episode is loaded and checked against the task's robot contract and rate before
the camera cache or motors start. The arm then moves to the saved start pose,
and each episode runs the procedure above. Afterwards `r` replays the same
episode again, Enter moves to the next one (cycling back to the first), and `x`
finishes. A normal finish homes with `reset_to_home()`. The camera server must
already be running for tasks with cameras. `--action_replay`, `--data-dir`,
`--fill_in`, `--arm_kp` and `--arm_kd` are rejected in this mode.

```bash
bash scripts/cameras/camera_servers_tmux.sh start left overhead
uv run scripts/visualize.py episode --task_name single_arm --review_replay 24 27
```

`visualize.py episode --action_replay` is a separate, older replay of one
episode with tracking plots; see
[src/visualization/README.md](../src/visualization/README.md#action-replay).

## Timing and gains

Review replay sends every retained row once at the collection FPS. A delayed
host tick slows playback instead of skipping rows or sending a catch-up burst.
Each target, including the final one, receives a full action interval. The SDK
interpolates these targets in its independent 500 Hz background loop; see
[fixed-timestep replay](replay_timestamps.md).

The C++ SDK executes the motor loop. `src/utils/hardware_utils.py` is the
application configuration/command adapter: it sets 0.002-second SDK periods and
linear interpolation, and future-dates streaming targets. Recording FPS controls
observation/action sampling, not the motor loop.

Active collection holds, positioning and replay share the preset in
`src/utils/hardware_utils.py`: Kp `[200, 200, 200, 100, 100, 100, 10]` and
Kd `[7, 7, 7, 1.5, 1.5, 1.5, 1]`, ordered J1–J6 then gripper. Passive
hand-guided collection uses zero Kp/Kd with gravity and friction compensation.
Gain equality is not an episode/checkpoint metadata requirement.

## Replay practice

`src/recording/replay_practice.py` (no task, no cameras, no episode saved) uses
the same review replay for its in-memory take. Without cameras there is no
overlay, and instead of the prompt its kp/kd fractions come from the keys
described in the repository README.

The replay code is validated with fake controllers and clocks; that does not
constitute a physical robot validation run.
