"""Synthetic independent streams for offline task-publication tests only."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import zarr


directories = []


def recorded_samples(samples, fps=100):
    directory = Path(tempfile.mkdtemp(prefix='episode-test-stream-'))
    directories.append(directory)
    names = list(samples[0].images)
    count = len(samples)
    stamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    control = zarr.open_group(str(directory / 'control.zarr'), mode='w')
    control.attrs['action_command_style'] = 'absolute'
    fields = dict(qpos=np.stack([s.qpos for s in samples]),
                  qvel=np.stack([s.qvel for s in samples]),
                  commanded_action=np.stack([s.qpos for s in samples]),
                  timestamps=stamps, elapsed_seconds=stamps,
                  continuity_id=np.zeros(count, dtype='i8'),
                  metadata_id=np.zeros(count, dtype='i8'),
                  intervention_active=np.zeros(count, dtype='u1'))
    fields['camera_observation/snapshot_timestamp'] = stamps
    for name in names:
        frames = [sample.images[name] for sample in samples]
        height, width = frames[0].shape[:2]
        video = directory / f'{name}.zarr/main.mp4'
        video.parent.mkdir()
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        assert writer.isOpened()
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        timing = [dict(video_frame=i, frame_number=1000+i,
                       capture_timestamp=float(stamp-.001), continuity_id=0)
                  for i, stamp in enumerate(stamps)]
        (directory / f'{name}.timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in timing))
        for key, value in dict(frame_number=1000+np.arange(count),
                               capture_timestamp=stamps-.001,
                               consumer_received_timestamp=stamps).items():
            fields[f'camera_observation/{name}/{key}'] = value
    for name, value in fields.items():
        control.create_dataset(name, data=value)
    (directory / 'manifest.json').write_text(json.dumps(dict(
        complete=True, retained=True, alignable=True, sample_count=count,
        fps=fps, camera_names=names, metadata={'action_command_style': 'absolute'})))
    (directory / 'alignment_report.json').write_text(json.dumps(dict(alignable=True, errors=[])))
    for index, sample in enumerate(samples):
        sample.episode_directory = str(directory)
        sample.robot_index = index
    return samples


def attach_streams(path, fps=100):
    """Give a synthetic annotation fixture authoritative source references."""
    import os
    from data_processing import episode_store
    with episode_store.Store(path) as root:
        poses = np.asarray(root['observations/qpos'])
        velocities = np.asarray(root['observations/qvel']) if 'observations/qvel' in root else np.zeros_like(poses)
        names = list(root['observations/images']) if 'observations/images' in root else []
        samples = recorded_samples([SimpleNamespace(qpos=q, qvel=v, timestamp=i/fps,
            images={name: np.zeros((16, 16, 3), np.uint8) for name in names})
            for i, (q, v) in enumerate(zip(poses, velocities))], fps=fps)
    source = Path(samples[0].episode_directory)
    manifest = json.loads((source/'manifest.json').read_text())
    (Path(path)/'manifest.json').write_text(json.dumps(manifest))
    (Path(path)/'episode.json').write_text(json.dumps(dict(schema_version=1,
        sources=[dict(directory=os.path.relpath(source, path), indices=list(range(len(samples))))])))


def retained_labels(directory):
    directory = Path(directory)
    records = {row['raw_index']: row['label'] for row in
               map(json.loads, (directory/'collection_labels.jsonl').read_text().splitlines())}
    keep = json.loads((directory/'edits.json').read_text())['retained_robot_indices']
    return [records[index] for index in keep]


def directory_bytes(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in Path(directory).rglob('*') if p.is_file()}
