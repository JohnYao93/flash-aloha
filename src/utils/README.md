# Shared utilities

Deterministic policy image transforms remain in
[`contracts/standard_image_processing.py`](../contracts/standard_image_processing.py).

`masking.py` applies the image/proprioceptive masks (`Masker`) that the masking config
defines, shared by training, direct inference and the training preview, and
`masking_calibration.py` holds the extrinsic calibration math and `masking_tuner.py` the
optional interactive polygon tuner; see [image preprocessing](../../docs/image_preprocessing.md#masking).

`random_utils.py` owns `set_seed`, shared by training and inference, with the
existing NumPy and Torch seeding behavior.

Training-only datasets, samplers, augmentation, history normalization, and metric
helpers remain in [`training_utils.py`](../policy/training_utils.py). The inference-only
`fill_in` helper lives in [`inference_utils.py`](../inference/inference_utils.py); its
callers retain their existing action-rate guards. Episode readers and publication
live in [`data_processing/`](../data_processing/README.md).

`window_utils.py` asks the X11 window manager to keep this process's OpenCV
window above others without moving focus (OpenCV's Qt backend ignores
`WND_PROP_TOPMOST`); it is a no-op without a display.

`hardware_utils.py` contains the shared ARX helpers formerly split across
`arx_active_gains.py`, `arx_collection_gains.py`, `arx_eval_gains.py`,
`arx_control_timing.py`, and `arx_model_selection.py`.

It owns the common active gain preset, controller timing configuration,
streaming-target timestamps, fixed replay interval validation, and SDK model
name normalization. Collection/evaluation gain names remain aliases of the same
preset. Function bodies and constant values are unchanged by consolidation.

Callers import directly from `utils.hardware_utils`. Importing the module does
not load the robot SDK, open CAN ports, or construct controllers. Recording,
inference, and replay entry points supply their existing SDK objects.

`replay_utils.py` owns shared replay functions extracted from the episode viewer
and the older recording replay: action-stream selection/preparation, command dispatch,
gain/state/torque helpers, tracking diagnostics and standalone ARX replay. It imports no SDK or visualization framework eagerly;
plotting loads Matplotlib only when requested by the tracking helper.

`visualization/visualize_episode.py --action_replay` calls
`replay_utils.replay_arx_actions`. The collector's review replay
(`recording_replay.replay_take`, also used by `recording/replay_practice.py`)
does not use this module; it sends rows through the collector backend's `replay_rows`
([docs/recording_review_replay.md](../../docs/recording_review_replay.md)).

The dispatch helper preserves immediate holds (`interval_s=None`) and scheduled
targets (`send_joint_target` with the original interval). It does not alter the
prepared command's joint/gripper fields. Command construction remains specific
to each caller so zero-velocity/torque fields and gripper conversion keep their
existing semantics.

Standalone replay constructs its controllers, homes before playback, ramps to
the first pose, and homes during cleanup. Optional tracking is unchanged.

Neither path substitutes recorded timestamps for nominal replay intervals or
skips targets to catch up after a delay. Camera-only review and capture-draining
remain in `recording/recording_replay.py`. The [owner rules](../../AGENTS.md)
still apply; tests use fake controllers and clocks.
