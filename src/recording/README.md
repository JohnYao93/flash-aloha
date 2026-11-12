# Recording

Generic recording entry points and infrastructure live here. Shared capture,
recording-adapter, action, and policy invariants remain in
[`contracts/`](../contracts/README.md).

| File | Responsibility |
| --- | --- |
| [record_episode_single.py](record_episode_single.py) | Hand-guided ARX collection, episode prompts, captured samples, and the shared ARX backend (`ARXBackend`) |
| [record_hardware.py](record_hardware.py) | Recording of externally generated targets through the shared hardware runtime |
| [replay_practice.py](replay_practice.py) | Record/replay practice without a task, cameras or saved data |
| [pedal_keys.py](pedal_keys.py) | Foot-switch keys shared by the collector and review replay |
| [distributed_recording.py](distributed_recording.py) | Camera lifecycle barriers, control-stream writer, coordinator, and episode publication |
| [episode_buffer.py](episode_buffer.py) | Recording scopes, stream-backed sample buffers, lazy recorded images, and proprio logging |
| [recording_buffer.py](recording_buffer.py) | Bounded ordered worker queues, failure propagation, rollback, and draining |
| [recording_alignment.py](recording_alignment.py) | Offline checks of exact frame references, clocks, and continuity |
| [recording_replay.py](recording_replay.py) | Post-capture robot/video review of recorded takes |
| [recording_utils.py](recording_utils.py) | Episode indexing, atomic JSON metadata writes, and recording-rate diagnostics |

The collector's review replay (`recording_replay.replay_take`) sends rows through
`ARXBackend.replay_rows` on the collector's existing controllers; replay
practice uses the same replay for its in-memory take. The
[shared replay helpers](../utils/replay_utils.py) serve `--action_replay` in the
[episode viewer](../visualization/visualize_episode.py). Neither appends replay
motion to the original take.

`recording_utils` combines helpers formerly defined in `record_episode_single`,
`distributed_recording`, and `recording_rate`. It depends only on the standard
library and NumPy. Code needing these helpers should import them directly:

```python
from recording.recording_utils import (
    atomic_write_json,
    get_auto_index,
    read_recording_rate,
    save_recording_rate,
    summarize_recording_rate,
)
from recording.episode_buffer import EpisodeBuffer, recording_scope
from contracts import standard_capture, standard_recording
```

The utilities preserve existing semantics: indexing selects the first unused
episode name; JSON conversion handles NumPy values and paths, and writes flush
and fsync before atomic replacement; rate diagnostics retain nominal versus
measured rates and explicitly identify legacy timestamp estimates. They do not
resample observations, change capture clocks, encode video, or control hardware.

## Entry points

Imports use `recording.<module>` with `src` on `PYTHONPATH`. The operator entry
point is `scripts/record.py` (modes `single`, `publish`); it executes the unchanged source entry points.
Both paths support execution from any working directory. From the
repository root, the operator can run:

```bash
source scripts/setup/paths.sh
uv run scripts/record.py single --task_name TASK
```

Use the task's existing arm, camera, and start-position configuration. These are
live robot commands for the operator to run; importing this package starts no
recording or motion.

The collector's `ARXBackend` drives one or two ARX L5 arms through the SDK's
joint controllers and records at the task YAML's `control_freq` (at most the
SDK's 500 Hz loop rate). Each sample reads position and velocity from one SDK
state per arm. Standard capture selects cameras afterward, and the existing
episode buffer persists its exact frame/timing references. Coordinates are the
SDK's joint radians and the gripper width in metres.

The backend uses **motion mode** for the saved-pose move and the holds
before/after each take: the collection tracking gains hold all six joints and
the gripper. The initial target defaults to the task's
`calibration/start_positions/<task>.npz` when saved, else
`calibration/start_position.npz`, and includes the saved gripper value;
`--no-start-position` holds the current pose instead. Press `1` to enter
**recording mode**: arms and grippers are passive (kp=kd=0 with the SDK's
gravity and friction feedforward). Takes are started and stopped with the `1`
key or pedal only. Stopping or aborting a take restores motion mode at the
measured pose before review.

`--guided-start` lets a take start anywhere. Each take skips the saved-pose
move and goes into **guided mode** instead: the same passive support as
recording mode, with nothing captured. Move the arm by hand, then press `1` to
**freeze** it there. This is the same measured arm and gripper hold used after
a take. Press `1` again to record from the frozen pose, exactly as in a normal
take. While frozen, `h`/`n` (right pedal) releases the arm so you can guide it
again. `x` holds the arm, then quits through the normal home move. Episodes
record `guided_start: true` and `start_position_applied: false`; the first
recorded row is the start pose.

```bash
uv run scripts/run.py collect TASK --guided-start
```

The two-pedal foot switch (`src/recording/pedal_keys.py`) drives the collector:
the left pedal (`1`) starts recording, stops it, saves the take and starts/stops
a replay; the right pedal (`h`) discards a take or cancels a replay, and in
damping returns the arm home. `f` stamps a failure event during recording,
`2` a subgoal switch. Each take reloads the setup file and prints its saved
gripper target. A normal exit (`x`/`q`) homes each arm with the SDK's
`reset_to_home()` and closes the controllers. Ctrl+C, SIGTERM and every
catchable failure instead take the shared failure exit: damping until `h`, then
home, then exit ([docs/safe_exit.md](../../docs/safe_exit.md)). SIGKILL, native
crashes and power loss cannot damp. The operator walkthrough is in
[docs/hand_guided_recording.md](../../docs/hand_guided_recording.md).

### Review replay from the collector

In the review after a take, `r` replays it **before** the save/discard decision;
at the next-episode prompt, `r` replays the last saved episode. Both run
`recording_replay.replay_take` on the collector's own controllers: a kp/kd
fraction prompt, the move to the recorded first pose, the alignment overlay,
then `1` to start and `1` to stop into damping (`h` returns to the start
position). Replay never edits data, and records only with `--record-replay`,
which saves the replay as a new episode marked with `replay_recording: true` and
`replay_source_episode`. The procedure is described in
[docs/recording_review_replay.md](../../docs/recording_review_replay.md).

Offline publication of stopped recordings keeps the same arguments:

```bash
uv run scripts/record.py publish --data-dir RECORDING_ROOT
```

The shared camera recorder automatically audits alignment at normal episode stop
and saves `alignment_report.json`; failures mark the recording incomplete.
Offline code can recheck saved streams with `recording_alignment.audit_episode_data`.
The LeRobot export metadata helper lives in
`data_processing.recording_rate_metadata.write_metadata`. Capture-rate summary
metadata can be generated manually with
`scripts/diagnostics/write_recording_rate_metadata.py`. Capture-rate measurement
helpers exist in `recording_utils`, but the recorder does not currently call
`save_recording_rate` automatically. Dataset reading/publication helpers live in
[data_processing](../data_processing/README.md), shared by collection, training,
inference, and visualization.

## Verification

Use synthetic recordings and simulated hardware. With pytest and the test
dependencies installed, run from the repository root:

```bash
PYTHONPATH=src:scripts:scripts/tests .venv/bin/python -m pytest \
  scripts/tests/test_record_episode_single.py \
  scripts/tests/test_standard_capture.py \
  scripts/tests/test_standard_recording.py \
  scripts/tests/test_distributed_recording.py \
  scripts/tests/test_shared_episode_recording.py \
  scripts/tests/test_recording_disk_buffer.py \
  scripts/tests/test_recording_rate.py \
  scripts/tests/test_episode_replay.py \
  scripts/tests/test_replay_practice.py \
  scripts/tests/test_recording_publication.py
```

This organization preserves capture ordering, exact frame references, stream
storage, action execution, timing, gains, and reset behavior. See the
[owner rules](../../AGENTS.md), [storage format](../../docs/episode_storage.md),
and [recording contract](../contracts/README.md#recording-and-exact-frame-persistence)
before changing those behaviors.
