"""Model/role-checked rollout execution through the common arm interface.

Uses the existing remote policy/chunk/RTC implementation and standard capture.
Arm handling follows direct inference (inference_direct) and the legacy remote
rollouts: connect and hold the measured pose, ramp to the task's saved start pose,
warm the policy up, refresh targets every tick (also while the loop blocks),
and home before disabling on every normal exit. Failures damp until 'h'.
"""
from contextlib import ExitStack
import os
from queue import Queue, Empty
import threading
import sys
import uuid

import numpy as np

from hardware.arm_factory import build_arms
from hardware.runtime import HardwareRuntime, RuntimeDamping
from hardware.safe_exit import damping_exit

KEYS_HELP = ('c/p/1: run or pause; r (pedal h): return to the start pose; s/f: finish an episode; '
             'q: home and quit')


def resolve_start_action(args, task_name, robot_type, hardware):
    """The saved start pose, selected and loaded exactly as direct inference does."""
    from config.task_config import default_start_position_path
    from inference import inference_direct as direct
    path = args.start_position_path
    if path is None:
        raise ValueError('Shared hardware always starts from the saved start pose; remove --no_start_position')
    if path == direct.DEFAULT_START_POSITION_PATH:  # same per-task/shared selection as the recorder
        path = default_start_position_path(task_name, robot_type)
    if not os.path.exists(os.path.expanduser(path)):
        raise FileNotFoundError(f'Saved start position not found: {path}')
    roles = list(hardware.arms)
    # Use the saved gripper, as the recorder does: takes start from the saved pose.
    # Gripper width is metres, so it keeps the default (small) gripper ramp step.
    start = direct.load_start_action_from_npz(path, is_double_arm=len(roles) == 2,
        action_dim=hardware.action_dim, closed_gripper_pos=direct.GRIPPER_CLOSED_POS,
        single_arm_side=roles[0] if len(roles) == 1 else None, use_saved_gripper_positions=True)
    return start, {}


def validate_start_action(hardware, start):
    """Check the start pose against every arm (native limits once connected) before moving."""
    offset = 0
    for arm in hardware.arms.values():
        end = offset + arm.describe().action_dim
        arm.validate_target(start[0, offset:end], interval_s=hardware.interval_s)
        offset = end
    if offset != start.shape[1]:
        raise ValueError('Start pose does not describe every selected arm')


def run_shared_rollouts(args, *, key_reader=None):
    # Reject unsupported workflows before constructing an SDK or connecting CAN.
    if args.record_dagger_corrections or args.dagger_footpedal_device or args.save_episode:
        raise ValueError('Shared hardware supports policy rollouts; manual DAgger/video workflows use their existing backend entry points')
    from inference.inference_remote import PolicyRuntime, PolicySession
    from inference import inference_direct as direct
    from inference.policy_chunks import validate_overlap_budget
    from inference.policy_transport import WebsocketOpenPIClient, timing_now, wait_until_loop_deadline
    from inference.policy_warmup import warm_up
    from camera_utils.capture import capture_image_from_shared_memory
    from contracts import capture_gc
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    from recording.record_episode_single import wait_for_cameras
    from getkey import getkey

    inputs = Queue()
    input_stop = threading.Event()

    def home_requested():
        # During the failure exit, 'h' arrives through the same key reader.
        while True:
            try:
                if inputs.get_nowait() == 'h':
                    return True
            except Empty:
                return False
    camera_stop = threading.Event()
    def read_keys():
        while not input_stop.is_set():
            try:
                inputs.put((key_reader or getkey)().lower())
            except (EOFError, KeyboardInterrupt):
                inputs.put('q')
                return

    def next_key():
        try:
            key = inputs.get_nowait()
        except Empty:
            return None
        # Two-pedal foot switch: right pedal returns to the start pose; left pedal is c.
        key = {'h': 'r', '1': 'c'}.get(key, key)
        if key == '\x03':
            raise KeyboardInterrupt  # Ctrl+C read as a key: damping, as for SIGINT.
        return key

    with ExitStack() as cleanup:
        cleanup.callback(input_stop.set)
        client = WebsocketOpenPIClient(args.remote_host, args.remote_port, args.api_key)
        cleanup.callback(client._ws.close)
        policy = PolicyRuntime(args, client)
        robot = args._yaml_task.as_dict()['robot']
        hardware = HardwareRuntime(build_arms(robot))
        hardware.require_policy(client.server_metadata)
        if hardware.action_dim != policy.action_dim:
            raise ValueError('Policy dimensions differ from configured hardware')
        start, start_options = resolve_start_action(args, policy.task['task_name'], robot['type'], hardware)
        validate_start_action(hardware, start)
        session = PolicySession(policy)
        cleanup.callback(session.close)
        camera_thread = threading.Thread(target=capture_image_from_shared_memory,
            args=(camera_stop,), kwargs={'camera_names': policy.cameras}, daemon=True)
        camera_thread.start()
        def close_cameras():
            camera_stop.set()
            camera_thread.join(timeout=5)
        cleanup.callback(close_cameras)
        wait_for_cameras(policy.cameras, timeout_sec=15)
        recorder = None
        if args.record_rollout or args.eval:
            recorder = DistributedRecorder(camera_names=policy.cameras, data_dir=args.recording_data_dir)
            cleanup.callback(recorder.close)
        # Runs after the normal-exit home move below (or the failure exit's home).
        cleanup.callback(hardware.close)
        arms = direct.RuntimeArms(hardware)
        dt = hardware.interval_s
        dual = len(hardware.arms) == 2

        def capture():
            return hardware.capture(policy.cameras, include_images=True)

        def move_to_start():
            # Same ramp as direct inference; it submits one target per tick itself.
            direct.move_to_start_action(arms, {}, start, execution_dt=dt, is_double_arm=dual, **start_options)
            return capture().qpos.copy()

        def return_to_start():
            # Reset from a hold: grippers open on the way back, saved gripper once arrived.
            direct.return_to_start_with_open_grippers(arms, {}, start, execution_dt=dt, is_double_arm=dual,
                                                      **start_options)
            return capture().qpos.copy()

        def hold_until(hold, accepted):
            """Refresh the hold every tick until one of ``accepted`` keys arrives."""
            while True:
                begin = timing_now()
                key = next_key()
                if key in accepted:
                    return key
                arms.submit(hold)
                wait_until_loop_deadline(begin, dt)

        try:
            # Any failure, including Ctrl+C: damping until 'h', then home.
            with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None,
                              home_requested=home_requested):
                # Connect, then hold the measured pose before anything else can block.
                arms.connect_and_hold()
                validate_start_action(hardware, start)  # native joint limits are known now
                hold = move_to_start()
                with direct.idle_hold(arms):
                    warm = capture()
                    warm_up(lambda index: policy.query(client, warm.images, warm.qpos, args.prompt,
                                                       query_index=index, rtc_metadata=policy.rtc_metadata())[1],
                            args.live_startup_warmup_steps,
                            check=(lambda timing: validate_overlap_budget(args.fps, args.async_prefetch_steps,
                                                                          timing['query_total'] * 1000))
                            if args.async_inference else None)
                threading.Thread(target=read_keys, daemon=True, name='shared-policy-controls').start()
                print(f'Shared hardware ready at the start pose. {KEYS_HELP}')
                for rollout_index in range(args.num_rollouts):
                    if rollout_index:
                        # The arm holds where the last episode ended until the operator confirms.
                        print(f'Episode {rollout_index} done. Press c/1 or r to return to the start pose '
                              'for the next episode, q to home and quit')
                        if hold_until(hold, ('c', 'r', 'q', 'x')) in ('q', 'x'):
                            break
                        hold = return_to_start()
                    with direct.idle_hold(arms):
                        capture_gc.collect_idle()  # paused between rollouts, before any capture
                        session.reset()
                        if recorder:
                            from camera_utils.camera_settings_contract import snapshot_camera_settings
                            metadata = {**hardware.recording_metadata(args.fps),
                                        'camera_settings': snapshot_camera_settings(policy.cameras),
                                        'task_name': policy.task['task_name'],
                                        'openpi_prompt': args.prompt}
                            recorder.start_episode(EpisodeRequest(session_id=uuid.uuid4().hex,
                                data_dir=args.recording_data_dir, camera_names=policy.cameras,
                                record_rollout=True, record_dagger=False, rollout_dir=args.rollout_dir,
                                rollout_task_name=policy.task['task_name'], dagger_dir='', dagger_task_name='',
                                dagger_camera_names=[], is_double_arm=dual,
                                fps=args.fps, eval_mode=bool(args.eval), metadata=metadata), timeout=15)
                            recorder.pause(2)
                    print(f'Episode {rollout_index + 1}: paused at the start pose. {KEYS_HELP}')
                    paused, executed, success, quit_requested = True, 0, False, False
                    while executed < args.max_timesteps:
                        begin = timing_now()
                        key = next_key()
                        if key in ('q', 'x'):
                            quit_requested = True
                            break
                        if key in ('s', 'f'):
                            success = key == 's'
                            break
                        if key in ('c', 'p', 'r'):
                            session.reset()
                            hold = capture().qpos.copy()
                            arms.submit(hold)
                            paused = (not paused) if key in ('c', 'p') else True
                            if key == 'r':
                                hold = return_to_start()
                            if recorder:
                                with direct.idle_hold(arms):
                                    (recorder.pause if paused else recorder.resume)(2)
                        if paused:
                            arms.submit(hold)
                            wait_until_loop_deadline(begin, dt)
                            continue
                        observation = capture()
                        if session.worker is None:
                            # A synchronous query blocks the loop; keep the hold fresh meanwhile.
                            with direct.idle_hold(arms):
                                action = session.tick(lambda: observation, args.prompt)
                        else:
                            action = session.tick(lambda: observation, args.prompt)
                        if not inputs.empty():
                            action = None
                        if action is not None:
                            action = np.asarray(action)
                            if action.shape != (1, hardware.action_dim):
                                raise ValueError('Policy must return one complete action row per execution tick')
                            hold = action[0].copy()
                            executed += 1
                        arms.submit(hold)
                        if action is not None:
                            session.committed()
                        if recorder:
                            recorder.append_sample({**observation.recording_fields(),
                                'commanded_action': hold.copy(), 'policy_action': hold.copy(),
                                'intervention_active': False})
                        wait_until_loop_deadline(begin, dt)
                    if recorder and recorder.active_session_id:
                        with direct.idle_hold(arms):
                            recorder.stop_episode(retained=True, outputs={'rollout': {
                                'requested': True, 'decision': 'success' if success else 'failure'}})
                    if quit_requested:
                        break
                # Every normal exit homes before the motors are disabled.
                direct.reset_controllers_to_home(arms)
        except BaseException:
            # The failure exit has already damped and homed (or disabled in place).
            if recorder and recorder.active_session_id:
                recorder.stop_episode(retained=True, outputs={'rollout': {
                    'requested': True, 'decision': 'failure'}}, incomplete=True)
            raise
