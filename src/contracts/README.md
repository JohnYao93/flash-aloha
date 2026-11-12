# Shared capture and policy contracts

These contracts apply to collection, training, statistics, replay, local inference,
and remote policy serving.
Task code may supply labels, goals, model features, and additional sequence
boundaries; it must preserve the observation, clock, action, and image contracts.

Import through `contracts` with `src` on `PYTHONPATH`:

```python
from contracts import standard_capture, standard_image_processing
from contracts import standard_recording
from contracts.action_contract import require_absolute_actions
from contracts.action_rate_contract import resolve_policy_rate
from contracts.sequence_contract import training_runs
```

| Module | Responsibility |
| --- | --- |
| [standard_capture.py](standard_capture.py) | One state read followed by one camera snapshot; observation provenance and validation |
| [capture_gc.py](capture_gc.py) | No automatic garbage collection in capturing/streaming processes; idle-only collection |
| [standard_recording.py](standard_recording.py) | Shared observation-to-recording adapters and rollout recording lifecycle |
| [standard_image_processing.py](standard_image_processing.py) | Shared deterministic image transforms and named model profiles |
| [action_contract.py](action_contract.py) | Absolute action representation and rejection of conflicting declarations |
| [action_rate_contract.py](action_rate_contract.py) | Explicit, consistent action and recording rates |
| [sequence_contract.py](sequence_contract.py) | Physical continuity, retained-row gaps, and optional additional boundaries |

The modules validate their own boundaries. Some requirements below also depend
on callers, camera infrastructure, recording adapters, or model servers. Passing
one validator alone does not establish end-to-end parity. Importing the package
must not start camera readers, a recorder, a policy server, or a robot controller.

## Observation capture and clock provenance

- Every live collection and inference path must reuse
  `standard_capture.capture_observation`. The state callback runs once, followed
  by one snapshot of the shared parallel camera-consumer cache.
- Each physical arm adapter must obtain qpos and qvel from the **same SDK state
  result**. The shared function copies both arrays before the camera snapshot so
  another SDK poll cannot overwrite the selected state. Sequential arm reads
  are not a claim of simultaneous sampling across arms.
- Inference must use the returned observation's images. Recording must persist
  its exact `recording_fields()`: contract version, robot-read start/completion
  times, qpos, qvel, and camera-observation metadata. The control payload contains
  references and numeric state, not image pixels.
- Never read state or images again to populate the other half of an observation.
  Post-capture feature construction must not mutate the recorded state or change
  the selected camera packets.
- Robot-read and snapshot timestamps use the monotonic clock. Camera capture
  timestamps retain their driver's monotonic-clock provenance. Per-camera frame
  IDs and consumer-received timestamps identify what was available at the
  snapshot; recording, publication, and decoding must preserve them.
- Never synthesize timestamps from FPS, substitute wall-clock time, or choose a
  nearest/newer frame during publication or decoding. Reusing a recorded frame
  across observations is valid when that exact reference was selected and passes
  timing checks. Independent cameras need not share a frame ID or exposure time.
- qpos and qvel must be finite one-dimensional arrays of equal shape. Human or
  camera-only capture uses `empty_robot_state()`; a zero-camera observation still
  uses the standard state/timestamp boundary.
- Capture contract versions must match. Camera metadata must name exactly the
  requested camera set and bind to the same robot-read timestamps. Read start,
  completion, and camera snapshot must be finite and ordered. Both the state-read
  duration and completion-to-snapshot delay must satisfy the configured age limit.
- Capture timestamps must be finite, positive, and pass age, future-time, and
  cross-camera-skew checks.
  Frame IDs must be nonnegative integers. Consumer publication must be finite,
  occur no later than the snapshot, and not precede capture beyond the configured
  future-time tolerance. Missing, stale, skewed, or unbound observations fail closed.

[Camera timing](../camera_utils/camera_timing.py) owns `TimingLimits`. Current
defaults are 50 ms maximum age, 50 ms camera skew, and 5 ms future tolerance;
`CAMERA_MAX_AGE_MS`, `CAMERA_MAX_SKEW_MS`, and `CAMERA_MAX_FUTURE_MS` configure them.
These are validation limits, not evidence of hardware synchronization or a
policy's measured delay tolerance. Validate the selected observation at its
snapshot time, rather than aging it against a later recorder-worker clock.

`clock` and `snapshot_provider` injection, and recording with
`require_camera_timing=False`, are **simulated-test facilities only**.

## Garbage collection during capture

- Automatic garbage collection never runs in a process that captures or
  carries live data. `capture_observation` calls `capture_gc.pause_for_stream()`
  before every read, which disables collection for the rest of the process (and
  overrides a library that re-enabled it). The recorder coordinator, camera
  servers and the camera cache call it before streaming.
- The first pause freezes existing objects without collecting: a collection
  there would stall the capture it precedes and age its frames.
- Cyclic garbage is reclaimed only by `capture_gc.collect_idle()` at idle points:
  the recorder after an acknowledged stop (both processes), camera servers after
  a recording stop, direct inference after a rollout ends, the policy server
  before each rollout. Never call it during an episode or rollout, and never call
  the `gc` module directly in these processes.

## Recording and exact-frame persistence

Live adapters must use [DistributedRecorder](../recording/distributed_recording.py),
[EpisodeBuffer](../recording/episode_buffer.py), or [standard_recording](standard_recording.py).
Streams start before the first captured sample and drain/close on normal exit,
cancellation, and errors. Camera-server processes own video encoding.

`standard_recording` keeps these adapter guarantees:

- `append_observation` persists the captured state, read timestamps, contract
  version, and exact camera references. It copies the supplied action and adds
  optional task labels without reading hardware again or queuing image pixels.
- `ensure_rollout_started` creates the buffer before capture and exposes lazy
  `RecordedImages` sequences. Appending observations must not decode or encode
  images; later image access resolves recorded references and serializes decoder
  access through a lock.
- `append_rollout` keeps diagnostic state, action, and timestamp lists tied to
  that same observation. It uses each selected frame's capture timestamp.
- `finish_rollout` removes and finishes the active buffer. The shared
  `recording_scope` from `episode_buffer` provides cleanup across exit paths;
  callers must establish that scope or equivalent guaranteed finalization.

Native video and control streams remain authoritative. Task annotations, accepted
episode views, and DAgger labels may select recorded rows and add explicit labels;
they must preserve original row identities, frame references, timestamps, and
continuity. A missing referenced video frame or failed alignment check invalidates
the result. Retain source data for diagnosis; do not repair pairing silently.

See [recording timestamps](../../docs/recording_timestamps.md),
[episode storage](../../docs/episode_storage.md), and
[recording_alignment.py](../recording/recording_alignment.py).

## Deterministic image processing

- Training and deployment must reuse this module with the same named profile:
  source channel order, crop, resize dimensions/interpolation, letterbox,
  grayscale, scaling, layout, and camera ordering. Matching shapes alone does
  not establish matching inputs.
- Custom profiles must compose the shared operations, be selected on both
  consumers, and have train/inference parity tests. Do not fork the implementation
  into a task, trainer, or server.
- `prepare_image` accepts HWC uint8 three-channel images, checks crop bounds,
  applies crop before resize, and converts declared RGB/BGR order without mutating
  input pixels. Width and height must be supplied together and be positive.
- Native ACT/Flow uses checkpoint-owned dimensions, INTER_AREA resizing, CHW
  float32 divided by 255, and optional mean-channel grayscale repeated across
  three channels. RGB is standard; a checkpoint's explicit ALOHA BGR convention
  remains binding. Conflicting top-level and `policy_config` dimensions fail.
- OpenPI physical-camera requests are letterboxed by the client with the
  `openpi_letterbox_448` profile (`openpi_request_images`) and sent as HWC RGB
  uint8 at the model resolution; the server checks the advertised contract and
  never resizes.
- A rendered goal or route canvas is a separately declared task input. It must
  not supply OpenPI physical-camera fields.
- Mask selection and goal rendering must match the checkpoint and training
  path. A runtime mask argument may confirm the saved mask, not replace it.
  Servers must reject transforms they cannot implement faithfully.
- Backbone normalization belongs inside the shared model. Stochastic training
  augmentation follows the shared deterministic stage and is disabled for parity
  comparisons. Recordings retain their original native pixels.
- Native policies use one current image per physical camera. Nonzero
  `image_history_len`, historical virtual cameras (`_t-`), and an inference image
  history buffer are rejected. Proprioception history and separately declared
  task-goal inputs are different features.
- Custom OpenPI checkpoints must carry a matching preprocessing contract.
  Do not silently migrate pre-resized datasets/checkpoints or upscale them to
  imitate native inputs. External trainers/servers must integrate and verify the
  shared profile before claiming parity.

Parity is defined for identical decoded inputs and configuration. Lossy codecs
and different decoders can change the input pixels before this module sees them.
See [image preprocessing](../../docs/image_preprocessing.md) for profiles,
checkpoint acceptance, and historical-data caveats.

## Action representation

- Supported actions are **absolute targets**. Every explicit
  `action_command_style` must be `absolute`; an explicit `relative_action_base`
  is rejected. Check all declarations, including nested `policy_config`,
  `task_config`, and `metadata`, rather than hiding a conflict with an override.
- Missing declarations preserve the existing interpretation of legacy absolute
  data. This compatibility does not prove the provenance of unlabeled values.
  Relative datasets/checkpoints must not be silently interpreted or converted
  into absolute targets.
- Absolute representation does not determine the label source. Measured poses
  and commanded policy actions can both be absolute.
  Task adapters must declare same-row versus future-state labels. DAgger's
  next-measured-state targets must respect retained-row continuity.
- This contract does not change joint order, dimensions, units, normalization,
  gripper conversion, gains, command masking, or homing. Existing robot/model
  adapters remain responsible for those conventions and their compatibility.

## Action and recording rates

- Rates must be finite and strictly positive. Generic declarations use
  `action_rate_hz`, `fps`, and `nominal_recording_hz`, including nested policy/task
  configuration. Present declarations must agree (`rel_tol=1e-6`).
- An explicitly requested runtime rate must agree with the authoritative rate.
  Missing metadata requires an explicit `legacy_rate`; a conflicting legacy
  declaration fails. Empty datasets and mixed-rate training datasets fail.
- `resolve_policy_rate` prioritizes explicit trained `action_rate_hz` declarations
  over observation FPS, while rejecting conflicts between those action-rate
  declarations. An intentionally strided action stream can therefore differ from
  its observation dataset. Raw-action dataset checks use `raw_action_rate_hz`.
- Nominal action rate, achieved recording rate, camera FPS, query frequency,
  and SDK/controller frequency are distinct. Do not derive one from another or
  use a changed controller rate to disguise an action-rate mismatch. Measured
  recording statistics belong in [recording_utils.py](../recording/recording_utils.py).
- Trained horizon, execution horizon, RTC prefix and chunk-overlap rules remain
  model/runner contracts. Correct rate metadata alone does not make execution
  schedules interchangeable; this package must not change their behavior.

## Sequence continuity and task boundaries

- Original retained row indices must be a one-dimensional, strictly increasing,
  nonnegative integer vector. Every omitted row and every change in a physical
  continuity label starts a new run; concatenation must not erase gaps.
- Observation continuity comes from `observations/capture_continuity_index`,
  `recording_timing/continuity_id`, and/or `continuity_id`. All present physical
  labels contribute boundaries. Raw future-action streams use
  `action_raw_capture_continuity_index`; missing raw continuity is rejected when
  observation labels contain gaps.
- Boundary labels must be integer/boolean vectors covering the selected rows;
  training boundary arrays contain one entry per corresponding sample.
  `contiguous_runs` returns nondecreasing run IDs. `run_bounds` requires those
  ordered run IDs and returns the containing half-open interval.
- Target chunks, prediction offsets, proprioception history, future-state
  labels, and action-difference statistics must remain within their permitted
  runs. Removed observations must never become implicit neighbors.
- Generic code does not infer task boundaries from subgoal names, intervention
  labels, or goal changes. An explicitly selected task boundary provider may
  add boundaries, but cannot remove physical ones. Use the same provider for
  training, validation, and statistics.

Task-specific boundary semantics stay with the task rather than being embedded
in these shared modules.

## Verification and changes

Changes require the applicable contract and integration tests using synthetic
data and simulated hardware. With pytest and the test dependencies installed:

```bash
PYTHONPATH=src:scripts:scripts/tests .venv/bin/python -m pytest \
  scripts/tests/test_standard_capture.py \
  scripts/tests/test_standard_recording.py \
  scripts/tests/test_standard_image_processing.py \
  scripts/tests/test_action_contract.py \
  scripts/tests/test_sequence_contract.py \
  scripts/tests/test_recording_publication.py \
  scripts/tests/test_raw_training.py
```

New image profiles need deterministic train/inference parity coverage; capture
changes need ordering, exact-reference, clock and failure-path coverage. Also
run the affected collection, inference, serving, replay and recording
tests. Tests must not start live camera recording or robot controllers.

The owner's [repository rules](../../AGENTS.md) still protect capture, image
processing, recording, hardware execution, rates, masking, gripper conversion,
and homing/reset behavior. This package relocation does not authorize changing
those semantics or deploying changes to a robot.
