# Fixed-timestep replay

Physical replay has one timing mode: send every retained action once, in row
order, at the episode's declared action FPS. This matches the fixed per-action
interval used by inference. Saved timestamp spacing does not drive playback.

At 100 FPS, actions are issued at nominal 10 ms intervals. For example, three
retained actions saved at 0, 20 and 53 ms are replayed in three 10 ms slots. The
nominal playback duration is action count / FPS (30 ms in this example), plus
initial positioning and any application delays. A recording that achieved only
50 samples/second while declaring 100 FPS will play faster than its recorded
wall-clock duration. Its measured rate does not replace its declared FPS.

Each ARX target is passed to the SDK with a fixed deadline one action interval
from the current SDK time. The SDK continues linear interpolation at 500 Hz
(2 ms control cycles). Replay does not interpolate targets using saved times,
look ahead to timestamped waypoints, or reconstruct the recorded trajectory's
jitter. A late application tick sends the next unsent action; it does not skip
rows or issue a burst to recover the saved timeline. Delays can lengthen replay.
The final action receives its full action interval before replay completes.

This is the only physical timing mode in
`src/visualization/visualize_episode.py --action_replay`. Playback uses the saved
FPS. Standalone replay requires an explicit legacy FPS for recordings without
rate metadata, and supplied FPS must agree with existing declarations. A
selected task YAML must agree with the episode's rate.

Saved timestamps and camera frame references remain untouched. They still
support recording validation, camera/proprio alignment, diagnostics, and
visualization. The loaders still validate
saved timestamp shape, finiteness and ordering as recording-integrity checks,
but those values are not a playback schedule.

Start positioning, gains, immediate stop/hold, and homing/reset behavior retain
their existing task-specific behavior. Fixed action cadence does not reproduce
inference compute latency, chunk replacement, RTC, or model-generated targets.

The owner explicitly requested fixed-timestep-only replay after reviewing the
saved-timestamp implementation. This supersedes older timestamp-based replay
descriptions in the historical contract audit.
