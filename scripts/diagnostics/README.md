# Offline diagnostics

`write_recording_rate_metadata.py` summarizes nominal and measured capture rates
for the source episodes of a LeRobot export. It writes the summary to the export's
`meta/recording_rate.json`, using `meta/info.json` for the exported action rate.

```bash
uv run scripts/diagnostics/write_recording_rate_metadata.py \
  --dataset-root /path/to/lerobot_dataset \
  /path/to/source/episode_00000.zarr /path/to/source/episode_00001.zarr
```

Pass exactly the source episodes included in the export. The command runs manually
and leaves source recordings unchanged. Its reusable implementation is
[`src/data_processing/recording_rate_metadata.py`](../../src/data_processing/recording_rate_metadata.py).
See [recording-rate metadata](../../docs/recording_rate_metadata.md) for details.
