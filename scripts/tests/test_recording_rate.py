"""Rate metadata follows retained samples, not pauses or discarded attempts."""
from pathlib import Path
import sys

from data_processing import episode_store
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from recording.recording_utils import read_recording_rate, summarize_recording_rate
from test_shared_episode_recording import Backend, Clock
from recording.record_episode_single import EpisodeCapture


def test_saved_rate_excludes_pause_forced_samples_and_rollback(tmp_path):
    clock = Clock()
    capture = EpisodeCapture(Backend(), [], fps=100, clock=clock.monotonic)
    retained = [capture.sample()]
    for _ in range(3):
        clock.sleep(.02)  # Recorder only achieved 50 Hz despite a 100 Hz target.
        retained.append(capture.sample())
    capture.pause()
    clock.sleep(120)
    capture.resume(retained[-1].timestamp)
    retained.append(capture.sample())
    clock.sleep(.001)
    retained.append(capture.sample(force=True))
    path = tmp_path / 'episode.zarr'
    with episode_store.Store(path, 'w') as root:
        root.attrs['fps'] = 100
        root.create_dataset('observations/qpos', data=[sample.qpos for sample in retained])
        root.create_dataset('recording_timing/timestamps', data=[sample.timestamp for sample in retained])
        root.create_dataset('recording_timing/interval_seconds', data=[np.nan if sample.capture_interval is None else sample.capture_interval for sample in retained])
        from recording.recording_utils import save_recording_rate
        save_recording_rate(root)
    with episode_store.Store(path) as root:
        report = read_recording_rate(root)
        assert root.attrs['nominal_recording_hz'] == 100
        assert root.attrs['measured_recording_hz'] == pytest.approx(50)
        assert report['measured_interval_count'] == 3
        assert report['sample_count'] == 6
        assert report['relative_error'] == pytest.approx(-.5)
        assert len(root['recording_timing/timestamps']) == 6
        assert np.isnan(root['recording_timing/interval_seconds'][0])
        assert np.isnan(root['recording_timing/interval_seconds'][-2:]).all()


def test_short_recording_does_not_invent_measured_frequency():
    report = summarize_recording_rate(100, [np.nan], sample_count=1)
    assert report['nominal_recording_hz'] == 100
    assert report['measured_recording_hz'] is None
    assert report['relative_error'] is None


@pytest.mark.parametrize('hz', [0, -1, float('nan'), float('inf')])
def test_invalid_nominal_frequency(hz):
    with pytest.raises(ValueError, match='finite and positive'):
        summarize_recording_rate(hz, [.01], sample_count=2)


def test_export_preserves_capture_rate_separately_from_dataset_rate(tmp_path, capsys):
    import json
    from data_processing.recording_rate_metadata import write_metadata
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'fps': 30}))
    source = tmp_path / 'raw.zarr'
    with episode_store.Store(source, 'w') as f:
        f.attrs['nominal_recording_hz'] = 100
        f.create_dataset('observations/qpos', data=np.zeros((3, 14)))
        f.create_dataset('recording_timing/interval_seconds', data=[np.nan, .02, .02])
    report = write_metadata(root, [source])
    assert report['action_sample_hz'] == 30
    assert report['sources'][0]['nominal_recording_hz'] == 100
    assert report['sources'][0]['measured_recording_hz'] == 50
    assert json.loads((root / 'meta/recording_rate.json').read_text()) == report
    assert 'difference=-50.00%' in capsys.readouterr().out
