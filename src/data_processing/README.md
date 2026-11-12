# Shared episode data processing

This package provides offline episode access for training, inference validation,
visualization, and annotation. Imports use `data_processing.<module>` with
`src` on `PYTHONPATH`; importing the package itself loads no data or models.

| Module | Responsibility |
| --- | --- |
| [episode_data.py](episode_data.py) | Episode discovery, raw/published readers, exact camera references, and bounded episode caching |
| [episode_store.py](episode_store.py) | Native episode publication, source selections, and Zarr annotations |
| [episode_inputs.py](episode_inputs.py) | Offline policy input validation, statistics, row metadata, and causal goal selection |
| [dagger_review.py](dagger_review.py) | Reversible review selections, segment boundaries, stale-review checks, and retained training runs |
| [video_processing.py](video_processing.py) | Exact ordinal video decoding, indexed seeks, and bounded frame caching |
| [recording_rate_metadata.py](recording_rate_metadata.py) | Carry source capture rates into an exported LeRobot dataset's metadata |

Saved control rows and frame references remain authoritative. Publication keeps
the original source streams; readers must fail on missing referenced frames.
Video timestamps help seek an already selected frame, never choose a different
observation. Metadata and historical goals stay within their saved row and
continuity boundaries. See [episode storage](../../docs/episode_storage.md)
and the [shared contracts](../contracts/README.md).

Training datasets, batching, augmentation, and training statistics remain in
[training_utils.py](../policy/training_utils.py). Live recording lifecycle belongs to
[recording](../recording/README.md), and deterministic policy image processing
belongs to [contracts](../contracts/README.md).
