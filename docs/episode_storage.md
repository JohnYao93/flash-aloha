# Episode storage and data processing

Collection, rollouts and DAgger share one recording format:

- `control.zarr`: robot state, commanded actions, continuity, and the camera frame references observed at each sample.
- `<camera>.zarr/main.mp4` and `<camera>.timing.jsonl`: independent encoded camera streams and driver timing.
- `manifest.json`, `alignment_report.json`, and optional edits/review/task sidecars.

Camera-server processes own encoding. The recording loop appends numeric values
and exact camera references through the standard observation contract.

Saving an accepted episode publishes an `episode_XXXXX.zarr` directory. Its
`episode.json` identifies source stream directories and original control-row
indices. Its Zarr arrays contain task annotations, events, and inference
query/chunk diagnostics. Source video is never copied or re-encoded. A grouped
route episode or checkpoint can reference several recorded segments.

Keep the source directories and published episodes together when copying a
dataset. References are relative, so moving the entire dataset tree preserves
them. Copying only an `episode_XXXXX.zarr` directory is insufficient. Existing
stream recordings remain readable; existing exported files are not read or
converted by this code.

`data_processing.episode_data.Episode` and `open_episode` resolve those selections for training
and inspection. They fetch the exact recorded frame IDs, including repeated
frames, through `data_processing.video_processing.ExactVideo`. Video readers use indexed seeks,
bounded frame caches and per-camera parallel decoding. Decoder state is not
inherited by training workers. There is no nearest-timestamp reconstruction.
Missing references or failed alignment checks reject the episode.

`episode_buffer.EpisodeBuffer` owns live recording lifecycle. `data_processing.episode_store`
publishes selections and Zarr labels. `data_processing.episode_inputs` supplies offline
validation/statistics helpers; `episode_visualization` shares the same reader.

Rollouts preserve commanded actions. DAgger publishes expert next-measured-pose
targets only where both rows belong to the same uninterrupted retained segment.
The policy targets, intervention masks, original action stream, and continuity
labels remain explicit. Discarding an episode marks it excluded without deleting
its source streams.

A stopped distributed recording with pending decisions can be published after
acceptance using `src/recording/distributed_recording.py --data-dir <recording-root>`.
This is offline metadata publication; it never starts a robot or camera service.
