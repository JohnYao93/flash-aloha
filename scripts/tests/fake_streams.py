import json
from pathlib import Path
import cv2
import numpy as np
import zarr
from recording.distributed_recording import ControlWriter

class FakeStreams:
    """Use real disk formats with synthetic cameras and a synchronous robot sink."""
    instances = []
    def __init__(self, **kwargs):
        self.instances.append(self)
        assert kwargs['require_camera_timing']
        self.names = kwargs['camera_names']
        self.enqueued_sample_count = self.committed_sample_count = 0
        self.continuity_id = 0
        self.camera_statuses = {}
        self.commands = []
        self.videos = {}
        self.timing = {name: [] for name in self.names}

    def start_episode(self, request, timeout):
        self.request = request
        self.directory = Path(request.data_dir) / request.session_id
        self.directory.mkdir(parents=True)
        self.writer = ControlWriter(self.directory / 'control.zarr')
        self.labels = (self.directory / 'collection_labels.jsonl').open('w')
        for name in self.names:
            group = zarr.open_group(str(self.directory / f'{name}.zarr'), mode='w')
            video = cv2.VideoWriter(str(self.directory / f'{name}.zarr' / 'main.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), 100., (64, 48))
            assert video.isOpened()
            self.videos[name] = (group, video)

    def check_health(self):
        pass

    def append_sample(self, sample):
        sample = dict(sample, continuity_id=self.continuity_id)
        self.labels.write(json.dumps(dict(raw_index=self.enqueued_sample_count, label=sample['collection_label'])) + '\n')
        # Two camera frames per control tick proves streams have independent counts.
        for name in self.names:
            for offset in (-.00002, -.00001):
                rows = self.timing[name]
                timestamp = sample['timestamp'] + offset
                if sample.get('capture_contract_version'):
                    timestamp = sample['camera_observation']['cameras'][name]['capture_timestamp'] + .001 + offset + .00001
                index = len(rows)
                rows.append(dict(video_frame=index, frame_number=1000 + index,
                    timestamp=timestamp + 1.2, capture_timestamp=timestamp - .001,
                    frame_ready_timestamp=timestamp, write_started_timestamp=timestamp + .002,
                    write_finished_timestamp=timestamp + .003, continuity_id=self.continuity_id,
                    buffer_flags=0x2000, driver_sequence=index))
                self.videos[name][1].write(np.full((48, 64, 3), (10 + index * 3) % 256, np.uint8))
        if sample.get('capture_contract_version'):
            from contracts.standard_capture import validate_capture_payload
            validate_capture_payload(sample, self.names)
        else:
            sample['camera_observation'] = dict(snapshot_timestamp=sample['timestamp'], cameras={
            name: dict(frame_number=rows[-1]['frame_number'], capture_timestamp=rows[-1]['capture_timestamp'],
                       consumer_received_timestamp=rows[-1]['frame_ready_timestamp'] + .000001)
            for name, rows in self.timing.items()})
        self.writer.append(sample)
        self.enqueued_sample_count += 1
        self.committed_sample_count += 1
        return sample['camera_observation']

    def pause(self, timeout):
        self.commands.append('pause')

    def resume(self, timeout):
        self.commands.append('resume')
        self.continuity_id += 1

    def stop_episode(self, **kwargs):
        self.labels.close()
        for name, (group, video) in self.videos.items():
            video.release()
            group.create_dataset('main_timestamps', data=np.array([r['timestamp'] for r in self.timing[name]]))
            (self.directory / f'{name}.timing.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.timing[name]))
        from dataclasses import asdict
        (self.directory / 'manifest.json').write_text(json.dumps(dict(**asdict(self.request), complete=not kwargs['incomplete'],
            state='incomplete' if kwargs['incomplete'] else 'complete', sample_count=self.writer.count,
            camera_video_encoding={'codec': 'test_mp4v'}, camera_stop_status=self.camera_statuses)))
        from recording.recording_alignment import audit_episode_data
        report = audit_episode_data(self.directory)
        (self.directory / 'alignment_report.json').write_text(json.dumps(report))
        manifest_path = self.directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['alignable'] = report['alignable']
        manifest_path.write_text(json.dumps(manifest))

    def close(self):
        if hasattr(self, 'labels'):
            self.labels.close()
        for _, video in self.videos.values():
            video.release()
