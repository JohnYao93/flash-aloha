# Recording clocks and independent streams

The hand-guided collector and the inference recorders
use `DistributedRecorder`. Camera-less tasks use the same format without video streams.
Each camera server owns its independent video stream. Robot samples contain
numeric state and task metadata; optional Zarr episode exports do not define the raw
recording timeline. ARX reads position and velocity from one `get_joint_state()`
result per arm. This is not a
simultaneous hardware measurement of both arms; host read timing is preserved.

Collection and inference share a RobotMQ camera-consumer cache. A lightweight
helper process runs one reader thread per camera, independent of the collector's
Python GIL. A slow/missing camera does not block the other readers. Full native
pixels stay in bounded shared-memory rings; the helper publishes coherent bundles
of frame IDs and real timestamps to a native C++ receiver in the consumer process.
The robot thread snapshots that local cache without a network request. Inference
copies pixels from the selected slots, while recording copies only numeric metadata.
`consumer_received_timestamp` records publication of the copied pixels in the
helper's cache; RobotMQ's relative transport clock is never used for alignment.
Eight slots per configured camera bound image memory (about 70 MiB of rings for
three 1280×800 BGR cameras). Readers share one helper per consuming process and
release it when the final lease closes. Camera additions do not restart it.
Acquisition and video encoding remain in the camera-server processes; their
bounded queues drain continuously, not only after the episode.

All live robot observation paths use `standard_capture.capture_observation`:
the collectors, remote inference and direct inference. Rollout and DAgger exports use
`standard_recording` over the same raw writer. Preview-only readers and offline compatibility exporters are not live
observation capture. Evaluation videos/NumPy plots remain diagnostics; use the
accompanying raw streams for aligned data.

## Required API for new scripts

`record_episode_single.py` is the standard standalone entry point.
`record_episode_robologger.py` has been removed. Task collectors and
inference continue to use the same independent per-camera RoboLogger backend.
For example, use `uv run scripts/record.py single --task_name TASK`.
Review `--help` before migrating custom commands; old arguments are not silently
forwarded. `--check-proprio-jitter` remains available, but the old CLI-only
`--camera-endpoint-base` option is not supported by this entry point.
New raw streams live under `<dataset>/data/capture_*`, with accepted Zarr episode references alongside them. The legacy `zip_robologger_episode.py` converter and
`generic_stream_compat.py` fallback have been removed; recorded data is retained.
Training and inspection use the raw readers with saved frame IDs.

Use `standard_capture.capture_observation(read_state, camera_names,
include_images=needs_policy_images)` once per observation. The state callback
returns `(qpos, qvel)` or a dict with both; auxiliary state is permitted. Only
after this call may a task resize, mask, annotate or construct policy features.
Send `observation.images` to preprocessing and `observation.recording_fields()`
to the shared recorder. Do not call a separate image getter or reread joints.
Inference receives owned image copies that remain valid while a policy query runs.
Slots are pinned during copying, and their generation and frame metadata must
match the selected references. If a slot was overwritten before it could be
pinned, capture fails instead of selecting a newer image. `include_images=False`
avoids copying pixels during collection without changing frame selection.

`EpisodeCapture`/`capture_episode_sample` provide this API for task collectors.
`EpisodeBuffer` and `standard_recording` provide task/rollout adapters over
`DistributedRecorder`. Start them before the first observation; use
`recording_scope` or an explicit `finally` to drain every exit path. The live
writer rejects missing/unbound contract-v1 payloads both before enqueueing and
inside the coordinator. Contract version is persisted in the control store and
manifest. Clock and camera-provider injection is for simulated tests only.

The contract is also in root `AGENTS.md` and covered by
`scripts/tests/test_standard_capture.py`. Direct camera mode is rejected. Use
server inference for simultaneous rollout + DAgger; the legacy direct runner
supports either recording mode independently, not two competing camera writers.

## Why older rows differed by 1.12 seconds

A row means a robot sample plus the left/overhead/right image assigned to that
sample. It does not mean those cameras exposed simultaneously.

The older camera server initialized an independent timestamp origin and then
added exactly 10 ms per delivered frame, even outside recording. All 100 recordings of one older dataset used this
format. Across the first and last episodes (ordered by file modification time),
the clock origins remained consistent while the cross-camera spread grew from
about 90 ms to 1.12 seconds.

In the first row of `55e00ccad4824bf78d68f0b414afa081`, the overhead counter was
6,840,473 and the left counter 6,840,247: 226 increments, or 2.26 synthetic
seconds. Their inferred origins differed by -1.129748443 seconds, producing a
1.130251558-second timestamp difference. Most rows have a 225-frame counter
difference and therefore a 1.120251558-second timestamp difference.

Between the first rows of `499928cea6364a95b99ed9bb3a79a38a` and that episode,
wall time advanced 66,807.813613 seconds while the left synthetic clock advanced
67,414.348493 seconds. Delivered-frame counts therefore advanced around 100.9 Hz
while timestamps assumed 100 Hz. Unequal accumulated frame progress across
cameras produced the additional relative drift. The old files cannot determine
whether that unequal progress came from sensor-rate differences, dropped frames
or capture stalls, nor recover physical exposure skew. Do not shift old images
by 1.12 seconds on the assumption that this was a measured image delay.

The newer `record_data/red_and_blue_v2` recordings inspected at the same time
were still schema 3 (10 files, no timing sidecars), with approximately 1.22–1.23
seconds of synthetic timestamp spread in the five completed recordings.

## Current contract

- `main_timestamps` and shared-memory timestamps are unmodified V4L2 driver
  timestamps in `CLOCK_MONOTONIC` seconds. Missing timestamps or unknown clock
  flags are rejected, not replaced with dequeue time. Nominal video FPS affects
  playback metadata only.
  An inode-bound `/dev/shm/flash_aloha_cam_<name>.clock.json` declaration lets
  inference reject old/unverified producers even if their synthetic timestamp
  happens to look fresh. Restarting a server recreates both the segment and its
  declaration. A declaration from an older segment cannot authorize a new one.
- `<camera>.timing.jsonl` maps every encoded video frame to capture time,
  frame-ready time, writer start/end time, driver flags/sequence and continuity.
  The flags distinguish end-of-frame from start-of-exposure timestamps. Neither
  this metadata nor matching timestamps provides hardware exposure triggering.
- `control.zarr/timestamps` records monotonic robot read completion;
  `robot_read_started_at` records the start of the read interval. Both arms use
  the same host clock as the cameras. `elapsed_seconds` is an edited task clock
  and must not be substituted for absolute timestamps during alignment.
- New control streams also store `camera_observation/snapshot_timestamp` and,
  under `camera_observation/<camera>/`, `frame_number`, `capture_timestamp`, and
  `consumer_received_timestamp`. Proprio is read first, then the consumer cache
  is snapshotted. Frame IDs identify the exact image paired with that reading;
  repeated IDs are valid when the robot samples faster than a camera.
  Inference recording preserves the snapshot actually used by the query, not
  a newer snapshot taken after the policy returns.
- Encoding queues preserve original capture times and report dropped frames.
  Shared-memory readers verify publication before and after copying pixels, so
  inference uses image/timestamp pairs from the same published frame.
- Conversion of new recordings follows the saved frame IDs, checks their
  capture timestamps and continuity, and fails if a referenced frame was dropped
  or is missing. It does not substitute a timestamp neighbor. Legacy recordings
  without references retain capture-time matching and cannot prove the exact
  consumer observation. Neither mode shifts clocks to force overlap.

### No garbage collection during capture

Python's full garbage collections are triggered by allocation counts, so in
an earlier hand-guided recorder they recurred at the same rows of different
captures (for example rows 456–457 of three captures of one task) and stalled
the 100 Hz loop for 25–50 ms. Every capturing or streaming process therefore
runs with automatic collection disabled (`src/contracts/capture_gc.py`) and
collects only at idle points between episodes or rollouts; see the
[contracts README](../src/contracts/README.md#garbage-collection-during-capture).

A one-hour-equivalent soak of the direct-inference step path (360,000 steps,
7,200 Flow queries with direct RTC, per-step capture and run logging, no
collection at all) grew the process by 2.4 MB and left about 21,000
unreachable objects, which one collection cleared in 1.8 ms. Disabled collection
therefore cannot exhaust memory within a session; idle collections only bound it.

## Alignability checks

Before recording becomes ready, after resume and every 250 ms during capture,
the coordinator reads current camera headers together. It checks each frame's
age, camera-to-camera spread and future timestamps. A failed check marks the
episode incomplete and preserves its raw streams. Inference checks the actual
image/timestamp bundle before supplying a new camera observation to the policy.

After writers drain, `alignment_report.json` checks monotonicity, driver clock
provenance, coverage, retained robot indices and rollback continuity without
decoding video. With references, its authoritative `consumer` results check the
selected frames' existence, age/skew, consumer publication before the snapshot,
and proprio-to-snapshot delay. Capture-time and camera-ready-time matching remain
separate diagnostics. Camera-ready time precedes shared-memory/consumer
publication and is not itself proof of what inference could see.
Failed checks prevent accepted episode publication.

Default limits are 50 ms frame age, 50 ms cross-camera spread and 5 ms into the
future. Configure them for the policy using `CAMERA_MAX_AGE_MS`,
`CAMERA_MAX_SKEW_MS`, and `CAMERA_MAX_FUTURE_MS` in the collector/inference
environment. These are engineering limits, not a measured policy tolerance.

Raw data from the shared collector buffer lives in `<dataset>/data/capture_*`.
Accepted episodes reference the
original streams; retain the whole dataset tree when moving recorded data.

## Operation (run these yourself)

RobotMQ is the default consumer cache. Restart the collector/inference process
to load this change; existing camera servers using the current clock contract
do not need a restart. Startup prints `[Camera cache] RobotMQ reader process
started`. `FLASH_ALOHA_CAMERA_CACHE=python` selects the previous Python cache for
diagnostic comparison; choose the backend before starting readers. There is no
automatic fallback on a native cache error. The existing freshness/skew checks
still fail closed. Keeping the native cache fresh does not eliminate pauses in
the Python robot loop itself.

If the camera servers also need upgrading, stop any active recording before
restarting them:

```bash
./scripts/cameras/camera_servers_tmux.sh start --replace
```

At normal episode stop, the shared recorder automatically writes
`alignment_report.json` and updates `manifest.json`; a failed audit marks the
recording incomplete. Inspect these files for a saved episode. Offline code can
rerun the read-only `recording.recording_alignment.audit_episode_data` function
with `src` on `PYTHONPATH`.

Robot replay reads retained `commanded_action` samples and edited
`elapsed_seconds` directly from finalized, aligned streams. It honors rollback
indices and excludes rollback waits; no image export is needed. The dataset
viewer reads the same source streams and exact camera references. Retain video,
control arrays, task labels, and timing sidecars together.

`video_extra`/`video_extra_overhead` cameras must also run this repository's
updated camera server (ports base+5/base+6, `flash_aloha_cam_*` shared memory),
with their calibrated dimensions/FPS. Other camera servers do not implement the
verified timestamp contract and are rejected.
Legacy direct-camera mode is not supported by the unified raw collectors.

## Direct raw training and viewing

The standard trainer and offline Rerun viewer share `data_processing.episode_data.Episode`
and exact referenced-frame decoding. Training can consume these independent
streams without an Zarr episode export. See [raw training](raw_training.md) for action
labels, optional task boundaries, cache controls and viewing commands.
