"""Post-capture review of a take, without editing its samples.

Robot replay uses every retained action at the take's nominal FPS. The caller
owns input/UI and the existing robot connection; this module opens neither.
"""
import time
from pathlib import Path

import numpy as np

from recording.pedal_keys import GO_KEY, REJECT_KEYS


def finish_take(samples):
    """Drain stopped capture before review; never append replay observations."""
    buffers = {}
    if hasattr(samples, 'finish'):
        buffers[id(samples)] = samples
    for sample in samples:
        buffer = getattr(getattr(sample, 'images', None), 'buffer', None)
        if buffer is not None:
            buffers[id(buffer)] = buffer
    for buffer in buffers.values():
        buffer.finish()


OVERLAY_ALPHA = 0.6  # Weight of the recorded first frame over the live camera image.
KEEP_ABOVE_ATTEMPTS = 200  # Overlay frames to wait for the window to map.


def overlay_camera(camera_names):
    """Overhead if recorded; otherwise the (single) wrist camera."""
    names = list(camera_names)
    if not names:
        raise ValueError('Replay alignment needs at least one recorded camera')
    return next((name for name in names if 'overhead' in name), names[0])


def blend_reference(live, reference, alpha=OVERLAY_ALPHA):
    """Recorded reference over the live frame; both BGR uint8."""
    import cv2
    live = np.asarray(live)
    reference = np.asarray(reference)
    if reference.shape[:2] != live.shape[:2]:
        reference = cv2.resize(reference, (live.shape[1], live.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(live, 1.0 - alpha, reference, alpha, 0.0)


def latest_saved_episode(dataset_dir):
    """Highest-index published episode in a dataset directory, or None."""
    import re
    from pathlib import Path
    episodes = [(int(match.group(1)), path) for path in Path(dataset_dir).glob('episode_*.zarr')
                if (match := re.fullmatch(r'episode_(\d+)\.zarr', path.name))]
    return str(max(episodes)[1]) if episodes else None


def load_replay_episode(path, *, camera_names, robot_contract, fps):
    """Recorded rows and the first frame of the overlay camera, with contract checks."""
    from contracts.robot_contract import require_robot_match
    from data_processing.episode_data import Episode
    with Episode(path, inspection=True) as episode:
        require_robot_match(robot_contract, episode.attrs, context=f'replay episode {path}')
        if abs(float(episode.fps) - float(fps)) > 1e-9:
            raise ValueError(f'Episode rate {episode.fps:g} Hz differs from the task rate {fps:g} Hz')
        actions = np.array(episode.actions, dtype=np.float64)
        camera = reference = None  # Camera-less task: no alignment overlay.
        if camera_names:
            camera = overlay_camera([name for name in camera_names if name in episode.images])
            reference = np.array(episode.images[camera][0])
    if actions.ndim != 2 or len(actions) < 2:
        raise ValueError('Replay requires at least two recorded actions')
    return actions, camera, reference


def take_replay_source(samples, camera_names):
    """Rows and first frame of an unsaved take, read back from its raw capture.

    The stopped take's streams are already on disk; frames come from the raw
    video through each sample's exact frame reference. Nothing is published.
    """
    finish_take(samples)
    actions = np.array([getattr(sample, 'action', sample.qpos) for sample in samples], dtype=np.float64)
    if actions.ndim != 2 or len(actions) < 2:
        raise ValueError('Replay requires at least two recorded samples')
    if not camera_names:
        return actions, None, None  # Camera-less task: no alignment overlay.
    camera = overlay_camera(camera_names)
    reference = np.array(samples[0].images[camera])
    return actions, camera, reference


def replay_episode(backend, path, camera_names, **kwargs):
    """Replay a saved episode: load its rows and first frame, then replay_take."""
    ctx = backend.ctx
    actions, camera, reference = load_replay_episode(
        path, camera_names=camera_names, robot_contract=ctx.robot_contract, fps=ctx.fps)
    return replay_take(backend, actions, camera, reference, label=str(path), **kwargs)


def replay_take(backend, actions, camera, reference, *, poll_keys, live_frame, label='take',
                   gain_scales=(1.0, 1.0), notify=print, show=None, pump=None, close_window=None,
                   sleep=time.sleep, record=None):
    """Return to the recorded start pose, align on an overlay, then replay on '1'.

    With no camera (``camera`` None) there is no overlay window; the same keys
    are read from the terminal.

    Returns 'completed', 'stopped', 'cancelled' or 'quit'. The '1' stop key damps
    until 'h' returns to the start position. Errors and Ctrl+C propagate to the
    collector's failure exit (damping until 'h', then home and exit). Keys still
    queued afterwards are discarded so a repeated '1' cannot save.
    ``gain_scales`` are the (kp, kd) fractions applied to the replayed rows only.
    ``record`` (--record-replay) records the replay: ``start()`` opens its streams
    after '1' and before the first row, ``sample(index, row)`` captures each tick,
    and ``stop()`` drains them once the rows end. Errors leave draining to the
    caller's recording scope.
    """
    try:
        return _replay_take(backend, actions, camera, reference, poll_keys=poll_keys,
                               live_frame=live_frame, label=label, gain_scales=gain_scales,
                               notify=notify, show=show,
                               pump=pump, close_window=close_window, sleep=sleep, record=record)
    finally:
        for _ in range(1000):
            if not poll_keys():
                break


def _replay_take(backend, actions, camera, reference, *, poll_keys, live_frame, label,
                    gain_scales, notify, show, pump, close_window, sleep, record=None):
    import cv2
    window = 'Replay alignment'
    if camera is None:
        live_frame, pump, close_window = (lambda name: None), (lambda: []), (lambda: None)
    if show is None:
        from utils.window_utils import keep_window_above
        attempts = [KEEP_ABOVE_ATTEMPTS]

        def show(image):
            cv2.imshow(window, image)
            # Each replay reopens the window; keep it above the focused terminal.
            if attempts[0] > 0:
                attempts[0] = 0 if keep_window_above(window) else attempts[0] - 1
    close_window = close_window or (lambda: cv2.destroyWindow(window))

    def window_keys():
        key = cv2.waitKey(1) & 0xff  # Also draws the window.
        return [] if key == 255 else [chr(key)]
    pump = pump or window_keys
    ctx = backend.ctx
    notify(f'[Replay] {label}: {len(actions)} rows at {ctx.fps:g} Hz; '
           + (f'overlay camera {camera}' if camera else 'no camera overlay'))

    def operator_stop():
        # A deliberate stop, not a failure: damp, then 'h' returns to the start
        # position for the review. Errors and Ctrl+C propagate to the collector's
        # failure exit instead (damping until 'h', then home and exit).
        notify(f"[Replay] Stopped by the '{GO_KEY}' key; entering damping")
        backend.enter_safe_stop()
        notify("[Replay] DAMPING: press 'h' to return to the start position")
        while 'h' not in [str(key).lower() for key in poll_keys() if key]:
            sleep(0.01)
        backend.setup_for_recording()

    notify('[Replay] Returning to the recorded start pose...')
    backend.move_to_pose(actions[0])

    reject = '/'.join(REJECT_KEYS)
    notify(("[Replay] Overlay: recorded first frame over the live camera. " if camera else "[Replay] ")
           + f"{GO_KEY} starts the replay; {reject} cancels; q quits.")
    decision = None
    try:
        while decision is None:
            frame = live_frame(camera)
            if frame is not None:
                image = blend_reference(frame, reference)
                cv2.putText(image, f'{GO_KEY}: replay   {reject}: cancel   q: quit', (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                show(image)
            for key in [*poll_keys(), *pump()]:
                if key == GO_KEY:
                    decision = 'go'
                elif key in REJECT_KEYS:
                    decision = 'cancelled'
                elif key in ('q', 'x', '\x1b'):
                    decision = 'quit'
            sleep(0.01)
    finally:
        close_window()
    if decision != 'go':
        notify('[Replay] Not replayed; holding the start pose')
        return decision

    notify(f'[Replay] Replaying at the recorded rate; {GO_KEY} stops')
    stop_keys = lambda: GO_KEY in poll_keys()
    kp_scale, kd_scale = gain_scales
    # Commanded rows and measured positions per tick (logs/replay/), for tracking-delay diagnosis.
    import config.paths
    from utils.run_log import RunLog
    source = Path(str(label))
    run_log = RunLog(f'{source.parent.name}_{source.stem}' if source.suffix else source.name,
                     root=config.paths.REPLAY_LOG_DIR)
    run_log.write_settings(label=str(label), fps=ctx.fps, rows=len(actions), kp_scale=kp_scale, kd_scale=kd_scale)
    recording = {}
    if record is not None:
        record.start()  # Streams start before the first replayed row is captured.
        recording = dict(on_tick=record.sample)
    try:
        completed = backend.replay_rows(actions, fps=ctx.fps, should_stop=stop_keys,
                                        kp_scale=kp_scale, kd_scale=kd_scale, log=run_log, **recording)
    finally:
        run_log.close()
    if record is not None:
        record.stop()
    if not completed:
        operator_stop()
        return 'stopped'
    backend.enter_motion_mode()
    notify('[Replay] Completed; holding the final pose')
    return 'completed'
