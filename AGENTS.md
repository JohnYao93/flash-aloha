# Robot operation

Give the user commands to run for robot operation. Never start a robot
controller, collector, replay, or robot motion yourself. An explicit request to
cancel an already-running robot session authorizes its normal graceful shutdown.
Read-only checks and tests with simulated hardware are allowed.

# Mandatory observation and recording contract

Every new or customized live recorder and policy-inference entry point MUST
reuse `src/contracts/standard_capture.py: capture_observation`. Do not implement local
image/proprio pairing. Read qpos and qvel from one SDK state result per arm,
then snapshot the shared parallel camera-consumer cache. Inference must use
the images from that returned observation; recording must persist its exact
`recording_fields()` (including frame IDs and driver/consumer timestamps).
Never reread images or proprio to populate the other half of an observation,
derive timestamps from FPS, or select nearest/newer frames during export.

Record through `DistributedRecorder`, `RawEpisodeBuffer`, or the adapters in
`src/contracts/standard_recording.py`; start streams before the first captured sample and
drain/close on all exit paths. Encoding remains in camera-server processes.
Task customization may add labels, actions, edits and post-capture image
transforms, but must not override capture ordering, packet selection, clock
provenance, or timing validation. `clock`/`snapshot_provider` injection and
`require_camera_timing=False` are simulated-test facilities only.

Missing/stale/skewed frames fail closed. Missing referenced frames in the raw
video invalidate export; retain raw data for diagnosis, never silently repair
the pairing. See `docs/recording_timestamps.md` and run
`scripts/tests/test_standard_capture.py` when changing either side.

# Mandatory garbage-collection pause during capture

Python's automatic garbage collection MUST NOT run in any process while it
captures or carries live data: the recording/inference control loop, the
recorder coordinator, camera servers and the camera cache. Use only
`src/contracts/capture_gc.py`. `capture_observation` disables automatic
collection from the first capture for the rest of the process; other streaming
processes call `capture_gc.pause_for_stream()` before streaming. Reclaim cyclic
garbage only with `capture_gc.collect_idle()` at idle points (after an episode
stop is acknowledged, between rollouts), never during an episode or rollout.
Do not call `gc.enable()`, `gc.disable()`, `gc.freeze()` or `gc.collect()`
directly in these processes. See `docs/recording_timestamps.md` and run
`scripts/tests/test_standard_capture.py` when changing either side.

# Mandatory policy image preprocessing

Training and inference MUST reuse `src/contracts/standard_image_processing.py` for deterministic
image transforms, with the same policy profile/configuration: source channel
order, crop, resize interpolation, letterbox, grayscale, float scaling and layout.
Do not implement separate train/deploy copies.
Custom processing MUST build on `standard_image_processing` by
composing its operations and defining a named policy profile; do not fork or
copy the standard implementation. Add the same profile to both consumers.
Resize ownership is model-specific. OpenPI through the remote
`inference_remote` client: the client letterboxes each native camera frame with
the `openpi_letterbox_448` profile (`openpi_request_images`) and sends RGB
uint8 at the model resolution; the server never resizes and rejects other shapes.
OpenPI training data keeps native pixels and OpenPI training letterboxes them
with `resize_with_pad`, which the profile matches exactly. The image contract
(`openpi_image_contract`) is the single declaration of that profile, its pixel
fingerprint and the client-side policy metadata; the trainer refuses to start
unless its own resize reproduces the fingerprint, every checkpoint stores the
contract, the server advertises it, and the client refuses any contract it does
not reproduce. This repository holds the client side only. Native
ACT/Flow uses the standard processor at its checkpoint-configured custom size.
Conflicting top-level/model dimensions must fail. Rendered task-goal canvases
are separately declared inputs, not permission to resize physical streams.
OpenPI checkpoints must carry a matching image-preprocessing contract; do not
silently migrate previously resized datasets/checkpoints or upscale them to
simulate native input. Keep augmentation separate from resize ownership.
Keep native recorded streams unchanged. Apply training-only stochastic
augmentations separately, after the
shared deterministic stage. Reuse the same task-specific masking/goal renderer
on both sides; a server unable to apply a checkpoint's transforms must reject it,
not silently omit them. Backbone normalization remains inside the shared model.
External trainers/servers must integrate the shared profile before claiming
parity; matching input shapes alone is insufficient. Add train/inference parity
tests to `scripts/tests/test_standard_image_processing.py` for every new profile.
See `docs/image_preprocessing.md` for supported profiles and migration caveats.

# Protected recording and hardware code

Do not modify the recording/input pipeline or any hardware-control code or
configuration in a way that breaks the contracts that must hold at collection,
training and inference time, and do not change their behavior through wrappers or
modified copies. Preserve camera/proprio capture and alignment, timestamps and frame
IDs, image preprocessing, recording/storage, robot drivers, action execution, RTC,
action chunks/rates/masking, gripper conversion, and homing/reset behavior. If a
requested change needs one of these to behave differently, say so and stop at that
boundary instead of making the change.

Data, checkpoint and log locations may be changed, but only through
`src/config/paths.py`, so they stay consistent across the repository.
