"""Carry Zarr capture timing into an exported LeRobot dataset (no image conversion)."""
import json
import math
from pathlib import Path

from data_processing import episode_store

from recording.recording_utils import read_recording_rate


def write_metadata(dataset_root, recordings):
    dataset_root = Path(dataset_root)
    info = json.loads((dataset_root / 'meta/info.json').read_text())
    action_hz = float(info['fps'])
    if not math.isfinite(action_hz) or action_hz <= 0:
        raise ValueError('Dataset fps must be finite and positive')
    sources = []
    for path in sorted({Path(p).resolve() for p in recordings}):
        with episode_store.Store(path, 'r') as root:
            report = read_recording_rate(root)
        report['source'] = str(path)
        sources.append(report)
        nominal, actual = report['nominal_recording_hz'], report['measured_recording_hz']
        error = report['relative_error']
        print(f'{path.name}: nominal={nominal} Hz, measured={actual} Hz'
              + (f', difference={error:+.2%}' if error is not None else ', comparison unavailable'))
    if not sources:
        raise ValueError('Provide the Zarr files actually included in the export')
    result = dict(schema_version=1, action_sample_hz=action_hz, sources=sources)
    target = dataset_root / 'meta/recording_rate.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(f'Training action rate: {action_hz:g} Hz; saved {target}')
    return result
