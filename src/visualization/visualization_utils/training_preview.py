"""Offline camera-resolution preview with unchanged recorded trajectories.

A training recipe supplies camera names, image dimensions and its masking strategy
(``data.masking_strategy``), which is applied to each frame with that frame's recorded
qpos exactly as training applies it. No training samples, statistics, augmentation,
normalization or targets are built.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys
import tempfile

import numpy as np

from config.training_args import build_parser
from config.training_config import load_training_config, resolve_training_task, training_summary
from contracts.standard_image_processing import resize_image
from visualization.visualization_utils.episode_visualization import VisualizationEpisode, resolve_episode_directory
from visualization.visualization_utils.memory_guard import MemoryGuard, open_rrd


def resolve_recipe(reference):
    recipe = load_training_config(reference)
    initial = recipe.arguments(build_parser())
    _, settings = resolve_training_task(initial)
    parser = build_parser()
    args = vars(parser.parse_args(['--training-config', str(recipe.source)]))
    if args.get('training_method', 'standard') != 'standard':
        raise ValueError('Resolution preview supports training.method: standard. '
                         'Specialized route methods own their image dimensions separately.')
    return training_summary(args)


def preview_output_path(args, directory):
    if args.get('save_rrd'):
        return Path(args['save_rrd']).expanduser().resolve()
    source = str(resolve_episode_directory(directory))
    recipe = str(load_training_config(args['training_config']).source)
    identity = json.dumps(['resolution-only-v1', source, recipe, args.get('preview_frame')])
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')).expanduser()
    return cache / 'flash-aloha' / 'training-previews' / f'{Path(source).stem}-{suffix}.rrd'


def open_prepared_preview(path):
    return open_rrd(path)


class TrainingPreview:
    def __init__(self, directory, reference):
        self.config = resolve_recipe(reference)
        self.directory = resolve_episode_directory(directory)
        self.camera_names = self.config['task_config']['camera_names']
        width, height = self.config['width'], self.config['height']
        if width < 0 and height < 0:
            width = height = None
        elif width <= 0 or height <= 0:
            raise ValueError('Image width and height must both be positive, or both negative for native resolution')
        self.width, self.height = width, height
        from utils.masking import Masker
        self.masker = Masker(self.config['task_config'].get('masking_strategy'))

    def transform(self, image):
        return resize_image(image, self.width, self.height)


def log_frame(rr, episode, preview, frame):
    rr.set_time('frame', sequence=frame)
    rr.set_time('episode_time', duration=float(episode.timestamps[frame]))
    images = episode.frame_bundle(frame, preview.camera_names)
    images, _ = preview.masker.mask({camera: images[camera] for camera in preview.camera_names}, episode.qpos[frame])
    for camera in preview.camera_names:
        # Shared readers return BGR uint8; Rerun displays RGB. This is only the
        # display channel convention, not policy color augmentation.
        rr.log(f'cameras/{camera}', rr.Image(np.ascontiguousarray(images[camera][..., ::-1])))
    for name, values in [('qpos', episode.qpos), ('qvel', episode.qvel), ('action', episode.actions)]:
        if values is not None:
            for index, value in enumerate(values[frame]):
                rr.log(f'recorded/{name}/dim_{index}', rr.Scalars(float(value)))


def visualize_training_episode(args):
    conflicts = [key for key in ('action_replay', 'save_video', 'plot_inference_chunks', 'pimode',
                                 'masking_strategy', 'aloha', 'single_arm', 'fps') if args.get(key)]
    if conflicts:
        raise ValueError(f'--training-config is an offline resolution-only preview; conflicting overrides: {conflicts}')
    if args.get('preview_frame') is not None and args['preview_frame'] < 0:
        raise ValueError('--preview-frame must be nonnegative')
    directory = args.get('data_dir')
    if directory is None:
        from visualization.visualize_episode import resolve_dataset_location
        root, name = resolve_dataset_location(args)
        directory = Path(root) / f'{name}.zarr'
    output = preview_output_path(args, directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prepare_training_episode(args, directory, output)
    print(f'[Preview] Ready: {output}', flush=True)
    print('[Preview] Reopen: ' + shlex.join([sys.executable, '-m', 'rerun', str(output)]), flush=True)
    if not args.get('prepare_only') and not args.get('save_rrd'):
        open_prepared_preview(output)
    return output


def _prepare_training_episode(args, directory, output):
    temporary_path = None
    recording = None
    guard = MemoryGuard()
    guard.check('Preview export')
    try:
        preview = TrainingPreview(directory, args['training_config'])
        with VisualizationEpisode(preview.directory, transform=preview.transform) as episode:
            missing = set(preview.camera_names) - set(episode.images)
            if missing:
                raise ValueError(f'Recording lacks configured cameras: {sorted(missing)}')
            count = len(episode.qpos)
            frames = range(count)
            if args.get('preview_frame') is not None:
                frame = args['preview_frame']
                if frame >= count:
                    raise ValueError('Requested frame is outside the episode')
                frames = [frame]
            import rerun as rr
            import rerun.blueprint as rrb
            from tqdm import tqdm
            rr.init('episode_resolution_preview')
            recording = rr.get_global_data_recording()
            with tempfile.NamedTemporaryFile(prefix=f'.{output.stem}-', suffix='.rrd',
                                             dir=output.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            rr.save(str(temporary_path))
            rr.send_blueprint(rrb.Blueprint(rrb.TimePanel(timeline='episode_time'), auto_views=True))
            rr.log('preview/settings', rr.TextDocument(json.dumps(dict(
                training_config=preview.config['training_config'], width=preview.width, height=preview.height,
                camera_names=preview.camera_names, masking_strategy=preview.masker.strategy,
                description='Camera resize and the recipe\'s mask only. Original recorded qpos, qvel and actions; saved timestamps.'
            ), indent=2)), static=True)
            print(f'[Preview] Preparing {len(frames)} frames; camera size '
                  f'{preview.width}×{preview.height}' if preview.width else
                  f'[Preview] Preparing {len(frames)} frames at recorded resolution', flush=True)
            for frame in tqdm(frames, desc='Preparing camera preview', unit='frame'):
                guard.check(f'Preview export at frame {frame}')
                log_frame(rr, episode, preview, frame)
            recording.flush()
            rr.disconnect(recording=recording)
            recording = None
            os.replace(temporary_path, output)
    finally:
        if recording is not None:
            recording.flush()
            rr.disconnect(recording=recording)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
