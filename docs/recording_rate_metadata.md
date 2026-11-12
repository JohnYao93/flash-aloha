# Recording and checkpoint rates

The shared raw recorder saves the configured `fps` in the manifest. The
`EpisodeCapture`/`EpisodeBuffer` path also saves `capture_interval` in
`control.zarr`; episode readers expose retained values as
`recording_timing/interval_seconds`. Initial, resumed, and forced samples use NaN,
so operator waits, encoding pauses and forced pedal samples do not distort the
measured rate. Retained row selections exclude rolled-back samples.

`recording.recording_utils.read_recording_rate` computes timing statistics from
these intervals. Measured Hz is valid interval count divided by their total
duration; reports also include nominal rate, median interval rate, counts, and
signed error. These measure robot sample cadence, not camera FPS.

The current recorder does **not** automatically call `save_recording_rate` at
finalization. That helper can save `measured_recording_hz` and
`recording_rate_json` attributes, but its presence does not mean a summary was
saved. Alignment auditing is already automatic at normal camera-recording stop,
via `DistributedRecorder`, which writes `alignment_report.json`.

## Dataset export

For manual diagnostics after export, run:

```bash
uv run scripts/diagnostics/write_recording_rate_metadata.py \
  --dataset-root /path/to/lerobot_dataset \
  /path/to/source/episode_00000.zarr /path/to/source/episode_00001.zarr
```

This command prints the rate summary and writes `meta/recording_rate.json` in
the exported dataset. It does not run automatically when recording stops.

The box-data exporter is not present in this checkout. Its finalization step
should call the reusable helper after creating the LeRobot dataset, with
**exactly the source episodes included in that export**:

```python
from data_processing.recording_rate_metadata import write_metadata

write_metadata(dataset_root, source_episode_paths)
```

The helper lives in `src/data_processing/recording_rate_metadata.py`; use
`src` on `PYTHONPATH`. It recomputes source rates
from saved intervals and writes `meta/recording_rate.json`. Carry this file with
the dataset to the training machine. Regenerate it when the source selection or
export rate changes. The helper only writes timing metadata; it does not resample
data or convert images.

This dataset-level file requires the export's `meta/info.json` and exact source
selection, so it cannot be produced just by stopping a raw recording. No exporter
in this checkout currently calls the helper automatically.

Source capture frequency and training action frequency are distinct: a 100 Hz
recording exported with actions resampled to 30 Hz retains nominal capture rate
100 Hz and `action_sample_hz=30`. The latter comes from `meta/info.json` and is the
checkpoint's native action playback rate. The exporter remains responsible for
actually resampling, rather than merely relabeling fps.

## Training and serving

An OpenPI LeRobot loader that integrates this manifest reads it at initialization, before model training. It prints nominal and measured
rates and their signed percentage difference for each source. Differences over
4% in either direction produce a warning; smaller differences are still logged.
Missing timing is explicitly unavailable, never assumed to match nominal fps.
Older recordings with only synthetic timestamps provide a labeled estimate;
recordings without timestamps cannot provide an actual measurement.

Each new checkpoint saves `assets/recording_rate.json`, including source-rate
reports and `action_sample_hz`. The policy loader publishes that report plus the
native action rate in server metadata. A configured rate that conflicts with the
checkpoint is rejected. A faster low-level controller may interpolate actions;
it must not reinterpret the native action sequence at that faster rate.

These hooks cover the local OpenPI LeRobot training/checkpoint path. Other remote
training checkouts need the same hooks. Existing recordings/checkpoints are not
rewritten, and no robot or training process is launched by these changes.
