# Training and viewing raw episodes

The standard ACT/Flow trainer reads independent raw recordings directly. Set
`data.directory` in the task YAML to the recording root, or to one raw stream
directory. Use `scripts/policy/imitate_episodes_train.py` for native training.
No zipping, Zarr episode creation, or decoded-image dataset is required. Task FPS and
chunk durations retain the existing shared collection/training/inference contract.
The specialized route recipes and external OpenPI trainers retain their own
input formats; this change adds raw support to the standard trainer.

The reader expects finalized `manifest.json`, `alignment_report.json`,
`control.zarr`, camera timing sidecars and each camera's `*.zarr/main.mp4`.
It discovers raw stream directories recursively. If a dataset root contains raw
recordings, its Zarr episode exports are ignored to avoid training on duplicate episodes.
Use separate roots to deliberately mix raw recordings and legacy-only Zarr episode data.
Incomplete/discarded episodes are excluded from discovery; invalid alignment or
missing references fail when opened. Pointing directly at an invalid episode
also fails. Raw datasets lacking exact consumer frame references require a
separate, explicit migration; nearest-timestamp matching is not a fallback.

## What a training row means

The shared reader, `src/data_processing/episode_data.py`, obtains each row's qpos, qvel,
commanded action and camera frame IDs from the raw control store. It resolves
those IDs against each camera's timing sidecar using the existing
`camera_timing.selected_frame_indices` contract. It verifies capture timestamp
and continuity provenance and decodes the corresponding encoded frame ordinal.
It never picks an image from nominal FPS or a nearby timestamp. Consecutive
robot rows can legitimately reference the same camera image.

`action` is the same row's recorded `commanded_action`. No next-row shift is
introduced by opening raw data. Existing explicit prediction-offset/future-action
options still control target sampling. In inspected v3 kinesthetic recordings,
commanded action equals measured qpos. Guided DAgger recordings may contain
policy commands; loading them does not manufacture expert action labels.
Use the task's expert-labeling workflow when that distinction matters.

Rollback edits and reviewed exclusions select original raw rows. Gaps and
physical continuity boundaries always prevent history/action chunks from
crossing removed or disconnected samples. Subtask boundaries remain optional and
task-owned; generic training does not interpret them.
Native training consumes one current image per physical camera. Multi-frame
image history is rejected explicitly, including in old checkpoint metadata.
Proprioception history remains a separate option.

The loader supplies native decoded BGR images to the existing shared policy
preprocessor and task masking logic. It introduces no new resize/color profile.
Avoid comparing raw pixels with a lossy JPEG Zarr episode re-encoding as if they were
identical; parity tests use the same decoded source frames.

## Rerun

From the repository root:

```bash
.venv/bin/python scripts/visualize.py episode \
  --robot arx --data-dir /path/to/raw/stream-directory --rerun
```

`--rerun` writes the episode to an `.rrd` file under the RAM guard, then opens
that file (see `src/visualization/README.md`). To write the file without
opening it:

```bash
.venv/bin/python scripts/visualize.py episode \
  --robot arx --data-dir /path/to/raw/stream-directory \
  --save-rrd /tmp/episode.rrd
```

A parent directory is accepted if it contains exactly one raw stream directory.
The viewer uses the same frame lookup and decoder as training. Each retained
robot row is logged with its referenced camera frames; `episode_time` uses
saved elapsed timestamps relative to the first retained row. Repeated frame IDs
remain repeated images. This offline viewing timeline does not change physical
replay's fixed-FPS scheduling. An RRD contains viewing data and therefore costs
additional disk space only when explicitly saved.

## Loading performance

- Cameras decode concurrently; training decodes only the requested cameras.
- Sequential viewing advances existing decoders instead of reopening videos.
- Random reads seek to keyframes using a verified presentation-timestamp index.
  Video timestamps only locate an already-selected frame; they never decide
  observation pairing. The first distant/backward access scans that video once.
- Small persistent index files are shared across workers with atomic creation
  and a file lock. They store roughly eight bytes per encoded frame, not images.
  A changed video file signature produces a new index.
- Persistent spawned PyTorch workers keep independent decoders and bounded
  episode/frame caches. The normal `--num_dataloader_workers` flag controls them;
  zero workers also works. Statistics are reused when training/statistics roots
  resolve to the same episodes.

Environment controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLASH_ALOHA_VIDEO_INDEX_CACHE` | `~/.cache/flash-aloha/video-index` | Small disposable video indexes; use local SSD storage on training hosts. |
| `FLASH_ALOHA_EPISODE_CACHE_SIZE` | `4` | Positive maximum number of open raw episodes per process. |
| `FLASH_ALOHA_FRAME_CACHE_MB` | `64` | Decoded-image cache budget per episode, divided across its cameras; zero disables it. |

Frame-cache memory scales with workers × cached episodes × per-episode budget,
plus decoder buffers, numeric arrays, and prefetched batches. Random access to
long video GOPs still requires decoding intermediate frames. These optimizations
do not guarantee GPU saturation; tune workers/cache sizes for available CPU,
RAM and storage bandwidth. A real 5,217-row v3 smoke check returned a three-camera
training sample successfully; this is input validation, not a complete model
training run or an end-to-end throughput benchmark.

The offline readers do not delete existing Zarr episode files or change collector output
behavior. Changing automatic
exports in other collectors crosses the protected recording/storage boundary in
`AGENTS.md` and is outside this offline change.
