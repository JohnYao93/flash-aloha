# Inference layout

Training lives in `src/policy`; execution and WebSocket serving live in
`src/inference`. The operator launcher `scripts/infer.py` (modes `direct`, `remote`, `serve`) executes these modules.
Both live entry points reuse `contracts.standard_capture.capture_observation`.
Recording persists that observation's exact fields through the shared recorders.

The direct runtime provides chunk blending and configured action/gripper
lookahead. The remote client uses asynchronous fixed-overlap RTC handoffs.
The configured overlap, not a measured-delay-dependent repair, selects the new
chunk start; the measured delay enforces the fail-closed budget.

Direct and legacy remote inference drive the ARX SDK controllers directly. Before
the first rollout they move to the task's saved start pose
(`calibration/start_positions/<task>.npz` when saved, else
`calibration/start_position.npz`), including its saved gripper value, so a
rollout starts where the pedal-started recordings started.

## Shared hardware runtime

`src/hardware/runtime.py` is a second, model/role-checked path to the arms, used
by remote inference with `--shared-hardware` and by the target-provider recorder
`src/recording/record_hardware.py`. It builds one ARX adapter
(`src/hardware/backends/arx.py`) per arm from the task YAML's `robot` block. On
this path every arm must name its `model` (`L5` or `Trossen_L5`) explicitly; the
model is never inferred from the vector length.

The adapters stay inert until `connect()`. Connecting leaves the arm in SDK
damping; the first target anchors a freshly measured pose and enables the active
gains. Nothing on this path homes the arm at start-up. Coordinates stay native:
SDK joint radians and gripper width in metres. Targets outside the SDK's joint
limits or gripper width are rejected, not clamped.

`control_freq` is the action rate; the SDK interpolates each target in its
500 Hz loop. Position and velocity come from one SDK state per arm. Feedback
ages are SDK publication ages, not physical CAN-frame ages. A state older than
`max_feedback_age_s` (default 0.1 s), or arms whose feedback times differ by
more than 20 ms, fails the read.

Each arm has its own command watchdog (`command_timeout_s`, default 0.5 s),
armed by its first submitted target. The target interval must be shorter than
this timeout. Expiry, and any fault in a runtime read or send, puts the arms
into native damping (gripper held) and rejects targets until the operator
presses `h` to home; see [safe_exit.md](safe_exit.md). Only an arm that cannot
be damped is disabled. The watchdog is host software, not a hardware emergency
stop.

The robot contract (`src/contracts/robot_contract.py`) holds only a version and
the model of each `left`/`right` arm. It is stored in episode metadata,
normalization statistics, checkpoint configs and server metadata, and the
runtime refuses a checkpoint or target provider whose contract differs. Tagged
and untagged episodes cannot be mixed in one training set. The contract cannot
detect a physically swapped CAN cable.

### Remote inference with `--shared-hardware`

Operator command:

```bash
PYTHONPATH=src .venv/bin/python -m inference.inference_remote \
  --task-config /path/to/task.yaml --shared-hardware \
  --remote_host localhost --remote_port 8000
```

After connecting, the client holds the measured pose, ramps to the task's saved
start pose (selected as above, including the saved gripper), warms the policy up
and waits paused. Targets are refreshed every tick, and from a background hold
while the loop blocks (warm-up, recorder transitions, synchronous queries).
`c`/`p`/`1` runs or pauses in place, `r` (or the right pedal `h`) returns to the
start pose, and `s`/`f` finishes an episode and holds; the next episode starts
after `c`/`1`/`r` with the same return to the start pose. These returns open the
gripper on the way back and move it to the saved start value, over 1 s, once
the arm has arrived; the start-up ramp does not open it. `q` (or the last
episode ending) homes with the SDK's `reset_to_home()` before the controllers
close. Failures damp until `h`, then home. This path reuses the existing policy
transport, action chunks and RTC. `--no_start_position`, DAgger correction
recording and `--save_episode` are rejected.

### Recording from a target provider

`record_hardware.py` records one episode of externally generated targets. A
provider is named as `module:factory`; `factory(task)` returns a callable with a
`robot_contract` attribute and an optional `close()`. The callable receives the
exact captured observation and returns a complete native action vector, left
then right when both arms are selected. It owns the input-device mapping.

```bash
PYTHONPATH=src .venv/bin/python -m recording.record_hardware \
  --task-config /path/to/task.yaml --target-provider my_inputs:make_provider
```

The recorder starts distributed camera recording before its first capture,
persists the observation's exact `recording_fields()`, and drains and closes on
exit. Failures damp until `h`, then home.
