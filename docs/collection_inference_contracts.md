# Shared collection, training, replay, and inference contracts

The shared modules and their required invariants live in
[`src/contracts/`](../src/contracts/README.md).

These requirements are independent of task. A task can provide labels, goals,
episode eligibility, and additional sequence boundaries; it cannot replace the
observation, clock, action representation, or preprocessing contracts.

| Contract | Shared owner | Applies to |
| --- | --- | --- |
| One proprio read followed by one camera snapshot; exact selected-frame references | `src/contracts/standard_capture.py` | Live collection and inference, including recorded inference |
| Driver clock provenance, frame age/skew, exact-frame export | `camera_utils/camera_timing.py`, `recording/recording_alignment.py` | Capture, raw validation, and decoding for training/review |
| Start streams before capture; numeric control records; server-owned video encoding; drain on exit | `recording/distributed_recording.py`, `recording/episode_buffer.py`, `contracts/standard_recording.py` | All live recording adapters |
| Absolute action targets only; reject contradictory declarations | `src/contracts/action_contract.py` | Training/statistics, policy configuration and metadata, recording, replay loaders |
| No sequence across a recording gap or removed raw span | `src/contracts/sequence_contract.py` | Generic retained-row selection; native ACT/Flow target chunks and observation history |
| Explicit deterministic transforms and camera roles | `src/contracts/standard_image_processing.py` | Training and inference; task profiles compose these operations |
| Declare native action frequency separately from capture/controller frequency | `contracts/action_rate_contract.py`, diagnostics in `recording/recording_utils.py`, and checkpoint/runtime rate metadata | Export, training, checkpoint loading and execution configuration |
| Matching normalization, action dimensions, model horizon, RTC protocol and preprocessing | Model-family loading/request adapters | Training and serving for that model family |

Absolute refers to representation, not label source: hand-guided measured poses
and selected policy commands can both be absolute. Tasks
still declare whether a label is same-row or a future measured state. Gripper
coordinates retain their existing robot-specific convention. No relative-action
training, statistics conversion, inference conversion, or dataset converter is
supported. Explicit old relative declarations fail, even if another config field
or CLI override says `absolute`. Legacy files without a declaration retain their
existing absolute interpretation; unlabeled numerical data cannot reveal its
provenance automatically.

**Physical continuity and task boundaries are separate.** `training_runs` reads
saved physical continuity (`observations/capture_continuity_index` or
`recording_timing/continuity_id` for Zarr episode observations,
`action_raw_capture_continuity_index` for raw action streams).
Native ACT/Flow target chunks, prediction offsets, image/proprio history, and
action-difference statistics respect those boundaries. Review selection also
splits at omitted raw rows.

**Subtasks are optional task behavior.** A generic task uses the generic training
and inference paths without subtask management. A task with subtasks needs its
task-specific boundary, goal/progression, and inference handling. Definitions
and transitions belong to task files; this distinction does not require copying
subtask settings through recording and checkpoint metadata or requiring an
extra subtask-ID field on every generic policy request.

**Generic training ignores subtask metadata by default**, including declared
task run attributes. Python callers can pass a `task_boundary_provider` to
`load_data`, `EpisodicDataset`, and `get_norm_stats`. Use the same provider for datasets and statistics. Other tasks
can supply their own provider without adding task semantics to the shared loader.

With this optional adapter, a task may advertise an Zarr episode `training_run_id_path` attribute
pointing to an integer vector with one label per observation. For raw future
targets it can advertise `raw_training_run_id_path`, with one label per raw
action. When opted in, a change in either declared vector ends a training sequence. The generic
loader does not infer boundaries from subgoal names, goal generations, or
intervention labels. Those decisions belong to task code. Task exporters must
carry both the physical continuity and their additional run labels into their
output; packing samples into a new file does not erase a gap.
Raw-action mode rejects missing raw continuity when observations contain gaps,
and, only when task boundaries are enabled, requires the raw task-boundary array
when an aligned task array is declared.

Model-specific settings are explicit parameters, not global task defaults:
OpenPI owns physical-camera resizing, while native ACT/Flow uses the saved
image size/profile. Query frequency, trained horizon, execution horizon, RTC
prefix, and low-level controller rate are distinct. Their agreement must be
verified for the selected runner/checkpoint; shared camera capture does not
prove interchangeable execution schedules.

All physical replay paths now use fixed-FPS action steps in retained row order.
Saved timestamp spacing is used for inspection, not playback scheduling. Inference
retains its existing late-RTC handoff behavior; fixed
replay cadence does not reproduce those handoffs. See
[fixed-timestep replay](replay_timestamps.md) for the current behavior.
