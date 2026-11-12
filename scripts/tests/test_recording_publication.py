"""Native metadata segments retain exact recording references on publication."""
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from data_processing.episode_data import Episode
from recording.distributed_recording import publish_recording
from test_raw_training import make_raw


@pytest.mark.parametrize('kind', ['rollout', 'dagger'])
def test_metadata_segments_publish(tmp_path, kind):
    raw = make_raw(tmp_path / 'raw')
    control = zarr.open_group(str(raw / 'control.zarr'), mode='a')
    count = len(control['timestamps'])
    control.create_dataset('inference_query', data=np.zeros(count, dtype='u1'))
    control.create_dataset('segment_id', data=np.zeros(count, dtype='i8'))
    control.create_dataset('policy_action', data=np.asarray(control['commanded_action']))
    manifest_path = raw / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(is_double_arm=False, **{f'{kind}_dir': str(tmp_path / kind)},
        outputs={kind: dict(requested=True, decision='accept')},
        metadata_segments=[dict(task_name='task',
                                intervention_active=False, openpi_prompt='')])
    manifest_path.write_text(json.dumps(manifest))
    outputs = publish_recording(manifest_path)
    assert len(outputs) == 1
    with Episode(raw) as source, Episode(outputs[0]) as published:
        expected = count - 1 if kind == 'dagger' else count
        assert len(published.qpos) == expected
        np.testing.assert_array_equal(published.qpos, source.qpos[:expected])
        for name in source.images:
            np.testing.assert_array_equal(published.images[name][0], source.images[name][0])
    assert (raw / 'control.zarr').exists()
