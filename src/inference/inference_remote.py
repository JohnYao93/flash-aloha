"""Native remote Flow inference with shared capture and execution contracts."""
from __future__ import annotations

# Support direct execution as well as package imports.
if __package__ in (None, ''):
    import sys as _path_sys
    from pathlib import Path as _Path
    _path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from queue import Queue, Empty
import threading
import time
import uuid
import sys
import numpy as np

from contracts.action_contract import require_absolute_actions
from utils.hardware_utils import ACTIVE_KP, ACTIVE_KD
from contracts.action_rate_contract import resolve_policy_rate
from camera_utils.camera_settings_contract import check_inference_camera_settings
from contracts import capture_gc, standard_capture
from contracts import standard_image_processing
from hardware.safe_exit import ARXDamping, damping_exit
from inference import policy_chunks
from inference.policy_chunks import validate_overlap_budget
from inference.policy_warmup import DEFAULT_WARMUP_STEPS, warm_up
from inference.policy_transport import WebsocketOpenPIClient, TimingStats, timing_now, wait_until_loop_deadline
from data_processing.episode_data import open_episode
from data_processing.episode_inputs import discover_episodes, read_robot_array, row_metadata
from config.paths import DAGGER_DIR, INFERENCE_RAW_DIR, ROLLOUT_DIR
from config.task_config import default_start_position_path, select_task_config, require_task_value

ACTION_DIM = STATE_DIM = 14
CAMERA_NAMES = ['video_left', 'video_overhead', 'video_right']
_RUNTIME = None
# Longest committed action prefix the OpenPI training-time RTC recipe conditions on.
OPENPI_MAX_RTC_PREFIX_STEPS = 20


def validate_policy_metadata(metadata):
    require_absolute_actions(metadata)
    standard_image_processing.require_single_frame(metadata)
    check_inference_camera_settings(metadata)
    if metadata.get('model_family') == 'openpi':
        # OpenPI server; PolicyRuntime checks its image contract against the task cameras.
        if metadata.get('request_schema') != 'openpi':
            raise ValueError('OpenPI servers must use the OpenPI request schema')
        return
    from config.supported_policy import require_supported_policy, require_unmasked_for_serving
    scoped_metadata = dict(metadata)
    if 'policy_class' not in scoped_metadata and metadata.get('policy_family') == 'flash_aloha':
        scoped_metadata['policy_class'] = 'Flow'
    require_supported_policy(scoped_metadata)
    require_unmasked_for_serving(scoped_metadata)


def configure_runtime_shape(args, metadata):
    global ACTION_DIM, STATE_DIM, CAMERA_NAMES
    name = args.task_name or metadata.get('task_name')
    task = select_task_config(name, args.task_config)
    if task is None:
        raise ValueError('Choose --task_name or --task-config for native inference')
    config = task.legacy()
    args.task_name = task.name
    args._yaml_task = task
    args._yaml_runtime_task_config = config
    ACTION_DIM = 7 if config['single_arm'] else 14
    robot_dim = metadata.get('robot_action_dim', metadata.get('action_dim', ACTION_DIM))
    if robot_dim != ACTION_DIM:
        raise ValueError('Checkpoint robot action dimension differs from the task arm configuration')
    STATE_DIM = int(metadata.get('state_dim', ACTION_DIM))
    CAMERA_NAMES = list(config['camera_names'])
    if STATE_DIM != ACTION_DIM:
        raise ValueError('Native remote inference requires a qpos-only policy; use direct inference for other state profiles')
    if args.query_frequency is None:
        from config.inference_config import default_execution_steps
        args.query_frequency = default_execution_steps(task)


def configure_yaml_policy_timing(args, metadata):
    # Bind the task to the served checkpoint's chunk size (its training config).
    task = args._yaml_task.require_checkpoint(metadata)
    args.fps = require_task_value(args.fps, task.fps, 'fps')
    from config.inference_config import configure_execution
    args._yaml_task = configure_execution(vars(args), task, metadata, backend='remote')


class PolicyRuntime:
    """Stateless request contract plus the shared transport/chunk implementation."""
    def __init__(self, args, client):
        self.args, self.client = args, client
        for name in ('num_rollouts', 'max_timesteps', 'dry_run_episodes', 'async_dry_run_steps', 'timing_interval'):
            if getattr(args, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if args.live_startup_warmup_steps < 0:
            raise ValueError('live_startup_warmup_steps must be nonnegative')
        metadata = client.server_metadata
        validate_policy_metadata(metadata)
        args.fps = resolve_policy_rate(metadata, args.fps, legacy_rate=args.legacy_policy_fps)
        configure_runtime_shape(args, metadata)
        configure_yaml_policy_timing(args, metadata)
        self.task = args._yaml_runtime_task_config
        self.cameras, self.action_dim = list(CAMERA_NAMES), ACTION_DIM
        self.horizon = args._yaml_task.prediction_steps
        self.schema = metadata.get('request_schema', 'openpi')
        self.native = self.schema == 'native' or metadata.get('policy_family') == 'flash_aloha'
        self.openpi = metadata.get('model_family') == 'openpi'
        self.model_action_dim = int(metadata.get('model_action_dim', self.action_dim))
        if not self.native:
            # The client letterboxes; the checkpoint's contract must match this process's pixels.
            self.image_contract = standard_image_processing.require_openpi_image_contract(
                metadata.get('image_preprocessing'), self.cameras)
        if args.async_inference:
            validate_overlap_budget(args.fps, args.async_prefetch_steps, args.max_inference_delay_ms)
            if args.query_frequency + args.async_prefetch_steps > self.horizon:
                raise ValueError('Execution window plus overlap exceeds the policy horizon')
        if args.rtc_prefix_steps < 0 or args.rtc_prefix_steps > self.horizon:
            raise ValueError('Invalid RTC prefix length')
        if self.openpi and args.rtc_prefix_steps > OPENPI_MAX_RTC_PREFIX_STEPS:
            raise ValueError(f'OpenPI RTC prefix exceeds its trained {OPENPI_MAX_RTC_PREFIX_STEPS}-step delay range')
        self.timing = TimingStats(enabled=args.timing, interval=args.timing_interval)

    def rtc_metadata(self, prefix=None):
        if not self.args.rtc:
            return {}
        if prefix is None:
            if self.openpi:
                return {}
            prefix = np.empty((0,self.action_dim),np.float32)
        prefix = prefix[:self.args.rtc_prefix_steps]
        delay = min(self.args.async_prefetch_steps, len(prefix))
        if self.openpi:
            padded = np.zeros((self.horizon, self.action_dim), np.float32)
            padded[:len(prefix)] = prefix
            return {'action_prefix': dict(actions=padded, prefix_length=len(prefix), delay=delay)}
        return {'rtc/enabled': True, 'rtc/inference_delay_steps': delay, 'rtc/prev_chunk_left_over': prefix}

    def query(self, client, images, qpos, prompt, *, query_index, rtc_metadata=None):
        started = timing_now()
        qpos = np.asarray(qpos, np.float32)
        if qpos.shape != (self.action_dim,) or not np.isfinite(qpos).all():
            raise ValueError('Policy state shape/value mismatch')
        request = {'observation/state': qpos, 'client/arm_count': self.action_dim//7,
                   'client/robot_action_dim': self.action_dim}
        if self.native:
            request['client/image_color_space'] = 'BGR'
            request.update({f'observation/images/{name}': images[name] for name in self.cameras})
        else:
            request.update(standard_image_processing.openpi_request_images(images, self.image_contract))
            request['client/image_preprocessing_sha256'] = self.image_contract['fingerprint']['sha256']
            request['prompt'] = prompt
        request.update(rtc_metadata or {})
        response, timing = client.infer_with_timing(request)
        actions = np.asarray(response['actions'], np.float32)
        raw_shape = actions.shape
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        supported = {(self.horizon, self.action_dim), (self.horizon, self.model_action_dim)}
        if actions.shape not in supported or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite action chunk with shape in {supported}, got {actions.shape}')
        timing['query_total'] = timing_now()-started
        timing['server_roundtrip'] = timing['query_total']
        return actions[None,:,:self.action_dim], timing, raw_shape


class PolicySession:
    """Shared async scheduling."""
    def __init__(self, runtime):
        self.runtime, self.args = runtime, runtime.args
        self.worker = None
        if self.args.async_inference:
            self.worker = policy_chunks.AsyncOpenPIQueryWorker(policy_client=runtime.client,
                timing_stats=runtime.timing, query_actions=runtime.query,
                max_delay_s=self.args.async_prefetch_steps / self.args.fps)
            self.worker.start()
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        if self.worker:
            self.worker.reset()
        self.actions = self.pending = None
        self.step = self.chunk_step = self.chunk_start = self.update_step = self.queries = 0

    def close(self):
        if self.worker:
            self.worker.stop()

    def tick(self, observe, prompt):
        if self.worker:
            self.pending = policy_chunks.poll_async_result(self.worker, self.pending,
                current_chunk_step=self.chunk_step, log_prefix='[Native]', expected_goal_generation=self.generation)
            if self.actions is None and self.pending is not None:
                self.actions, self.chunk_start = policy_chunks.merge_sync_openpi_chunk(
                    new_action_chunk=self.pending['action_chunk'],current_chunk_size=self.args.query_frequency)
                self.pending = None
                self.update_step = self.step
            elif self.actions is not None:
                (self.pending,self.actions,self.chunk_start,self.update_step,self.chunk_step,*_) = policy_chunks.maybe_merge_async_result(
                    self.args,pending_async_result=self.pending,all_actions=self.actions,t=self.step,
                    current_chunk_step=self.chunk_step,current_chunk_update_t=self.update_step,
                    current_chunk_start_index=self.chunk_start,async_prefetch_steps=self.args.async_prefetch_steps,
                    timing_stats=self.runtime.timing,log_prefix='[Native]')
            query = policy_chunks.should_submit_async_query(self.worker,all_actions=self.actions,
                pending_async_result=self.pending,current_chunk_step=self.chunk_step,current_chunk_start_index=self.chunk_start,
                async_prefetch_steps=self.args.async_prefetch_steps,force_submit=self.actions is None)
        else:
            query = self.actions is None or self.step % self.args.query_frequency == 0
        if query:
            observation = observe()
            prefix = policy_chunks.get_unexecuted_chunk_suffix(self.actions,
                current_chunk_step=self.chunk_step,current_chunk_start_index=self.chunk_start)
            rtc = self.runtime.rtc_metadata(prefix)
            if self.worker:
                self.worker.submit(image_dict=observation.images,qpos=observation.qpos,prompt=prompt,
                    query_index=self.queries,submit_chunk_step=self.chunk_step,submit_rollout_index=self.step,
                    rtc_metadata=rtc,goal_generation=self.generation)
            else:
                actions,_,_ = self.runtime.query(self.runtime.client,observation.images,observation.qpos,prompt,
                    query_index=self.queries,rtc_metadata=rtc)
                self.actions,self.chunk_start = policy_chunks.merge_sync_openpi_chunk(new_action_chunk=actions,current_chunk_size=self.args.query_frequency)
                self.update_step,self.chunk_step = self.step,0
            self.queries += 1
        if self.actions is None:
            return None
        return policy_chunks.select_action_from_openpi_chunk(all_actions=self.actions,
            elapsed_from_chunk_update=self.step-self.update_step,chunk_start_index=self.chunk_start)

    def committed(self):
        self.step += 1
        self.chunk_step += 1


def run_dry_run(args):
    """Exercise raw observations through the same scheduling used by live inference."""
    from types import SimpleNamespace
    client = WebsocketOpenPIClient(args.remote_host,args.remote_port,args.api_key)
    session = None
    try:
        runtime = PolicyRuntime(args,client)
        paths = discover_episodes(args.dry_run_dataset_dir or runtime.task['dataset_dir'],args.fps)
        session = PolicySession(runtime)
        for path in paths[:args.dry_run_episodes]:
            session.reset()
            with open_episode(path) as data:
                rows = read_robot_array(data,'observations/qpos',runtime.action_dim)
                for row in range(min(len(rows),args.async_dry_run_steps)):
                    def observe():
                        return SimpleNamespace(qpos=rows[row],images=data.frame_bundle(row,runtime.cameras))
                    action = session.tick(observe,str(row_metadata(data,row).get('openpi_prompt',args.prompt)))
                    if action is not None:
                        if not np.isfinite(action).all():
                            raise ValueError('Dry-run action is non-finite')
                        session.committed()
                    if args.async_inference:
                        time.sleep(1/args.fps)
                if session.step == 0:
                    raise RuntimeError('Dry run produced no executable actions')
                print(f'[Dry Run] {path}: {session.step} actions, {session.queries} queries; no hardware started')
    finally:
        if session:
            session.close()
        client._ws.close()


run_async_dry_run = run_dry_run


def run_rollouts(args, *, key_reader=None):
    # Select shared hardware before importing any ARX-specific execution code.
    if getattr(args, 'shared_hardware', False):
        from inference.shared_hardware import run_shared_rollouts
        return run_shared_rollouts(args, key_reader=key_reader)
    from contracts.robot_contract import metadata_robot_contract
    selected_task = select_task_config(args.task_name, args.task_config)
    if selected_task is not None and selected_task.as_dict()['robot']['type'] != 'arx':
        raise ValueError('This runner requires robot.type arx')
    # Hardware imports are deliberately confined to the live entry point.
    from inference import inference_direct as arm
    from getkey import getkey
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    client = WebsocketOpenPIClient(args.remote_host,args.remote_port,args.api_key)
    controllers,commands = {},{}
    session = recorder = camera_stop = pedal = None
    dual = False
    recording = manual = False
    gains = {}
    inputs = Queue()
    input_stop = threading.Event()
    def read_keys():
        while not input_stop.is_set():
            try:
                inputs.put((key_reader or getkey)().lower())
            except (EOFError,KeyboardInterrupt):
                inputs.put('q')
                return
    failed = False

    def home_requested():
        # During the failure exit, 'h' arrives through the same key reader.
        while True:
            try:
                if inputs.get_nowait() == 'h':
                    return True
            except Empty:
                return False

    try:
        # Any failure, including Ctrl+C: native ARX damping until 'h', then home.
        with damping_exit(lambda: ARXDamping(controllers.values()) if controllers else None,
                          home_requested=home_requested):
            runtime = PolicyRuntime(args,client)
            task = runtime.task
            if metadata_robot_contract(client.server_metadata) is not None:
                from contracts.robot_contract import require_robot_match
                legacy_robot = {'robot_contract_version': 1, 'arms': {role: {'model': 'L5'}
                    for role in ([task.get('single_arm_side') or 'left'] if task['single_arm'] else ['left', 'right'])}}
                require_robot_match(legacy_robot, client.server_metadata, context='legacy ARX execution')
            dual = not task['single_arm']
            arm.EXECUTION_DT = dt = 1/args.fps
            camera_stop = arm.setup_cameras_and_robot(is_double_arm=dual,camera_names=runtime.cameras)
            config,left,right = arm.init_arm(is_double_arm=dual,kp=args.arm_kp,kd=args.arm_kd,
                single_arm_side=task.get('single_arm_side'),single_arm_can=task.get('single_arm_can'),
                can_interface_left=task['can_interface_left'],can_interface_right=task.get('can_interface_right'))
            controllers = {'left':left,**({'right':right} if dual else {})}
            commands = {name:arm.arx5.JointState(config.joint_dof) for name in controllers}
            gains = {name:arm.snapshot_controller_gain(controller) for name,controller in controllers.items()}
            start_path = args.start_position_path
            if start_path == arm.DEFAULT_START_POSITION_PATH:  # same per-task/shared selection as the recorder
                start_path = default_start_position_path(task['task_name'],'arx')
            start = arm.load_start_action_from_npz(start_path,is_double_arm=dual,
                action_dim=runtime.action_dim,closed_gripper_pos=arm.GRIPPER_CLOSED_POS,single_arm_side=task.get('single_arm_side'),
                use_saved_gripper_positions=True)  # takes start from the saved pose, gripper included
            arm.move_to_start_action(controllers,commands,start,execution_dt=dt,is_double_arm=dual)
            def capture():
                return standard_capture.capture_observation(lambda:arm.get_observations(controllers),runtime.cameras,include_images=True)
            warm = capture()
            warm_up(lambda index: runtime.query(client,warm.images,warm.qpos,args.prompt,query_index=index,
                                                rtc_metadata=runtime.rtc_metadata())[1],
                    args.live_startup_warmup_steps,
                    check=(lambda timing: validate_overlap_budget(args.fps,args.async_prefetch_steps,
                                                                  timing['query_total']*1000))
                    if args.async_inference else None)
            session = PolicySession(runtime)
            if args.record_dagger_corrections and args.dagger_footpedal_device:
                from inference.dagger_controls import DaggerFootpedalReader
                pedal = DaggerFootpedalReader(args.dagger_footpedal_device, inputs.put)
                pedal.start()
            if args.record_rollout or args.record_dagger_corrections or args.eval:
                recorder = DistributedRecorder(camera_names=runtime.cameras,data_dir=args.recording_data_dir)
            threading.Thread(target=read_keys,daemon=True,name='native-policy-controls').start()
            print('c/p: run or pause; r: ready reset; q: home and exit; pedal 1: c, pedal h: ready reset; '
                  'DAgger 1 (while running): freeze, 2: intervention, 3: policy')
            for rollout in range(args.num_rollouts):
                capture_gc.collect_idle()  # paused between rollouts, before any capture
                session.reset()
                paused,finished = True,False
                executed = 0
                key = None
                video = {name: [] for name in runtime.cameras}
                hold = capture().qpos[None]
                if recorder:
                    recorder.start_episode(EpisodeRequest(session_id=uuid.uuid4().hex,data_dir=args.recording_data_dir,
                        camera_names=runtime.cameras,record_rollout=bool(args.record_rollout or args.eval),
                        record_dagger=args.record_dagger_corrections,rollout_dir=args.rollout_dir,rollout_task_name=task['task_name'],
                        dagger_dir=args.dagger_dir,dagger_task_name=task['task_name'],dagger_camera_names=runtime.cameras,
                        is_double_arm=dual,fps=args.fps,eval_mode=bool(args.eval),metadata={'task_name':task['task_name'],'openpi_prompt':args.prompt}),timeout=15)
                    recorder.pause(2)
                while not finished and executed < args.max_timesteps:
                    begin = timing_now()
                    if pedal is not None and pedal.error is not None:
                        raise RuntimeError('DAgger pedal input failed') from pedal.error
                    try:
                        key = inputs.get_nowait()
                    except Empty:
                        key = None
                    # Two-pedal foot switch: right pedal resets to ready; left pedal is
                    # c, except that DAgger keeps it as freeze while running.
                    if key == 'h':
                        key = 'r'
                    elif key == '1' and (paused or not args.record_dagger_corrections):
                        key = 'c'
                    if key in ('q','\x03'):
                        return
                    if key in ('1','2','3') and not args.record_dagger_corrections:
                        key = None
                    if key in ('c','p','r','1','2','3'):
                        session.reset()
                        hold = capture().qpos[None]
                        arm.set_action(hold,controllers,commands,dual,interpolate=False)
                        if manual:
                            arm.apply_gain_mode(controllers,gains,dual,zero_gains=False)
                        manual = key=='2' and args.record_dagger_corrections
                        paused = (not paused) if key in ('c','p') else key in ('r','1')
                        if manual:
                            arm.apply_gain_mode(controllers,gains,dual,zero_gains=True)
                        if key=='r':
                            arm.return_to_start_with_open_grippers(controllers,commands,start,execution_dt=dt,is_double_arm=dual)
                            hold = capture().qpos[None]
                    if key in ('s','f'):
                        finished=True
                        break
                    if recorder and (recording != (not paused) or key in ('c','p','r','1','2','3')):
                        (recorder.resume if not paused else recorder.pause)(2)
                        recording = not paused
                        recorder.set_metadata({'task_name':task['task_name'],'intervention_active':manual,'openpi_prompt':args.prompt},timeout=2)
                    if paused:
                        arm.set_action(hold,controllers,commands,dual,interpolate=False)
                    else:
                        observation = capture() if recorder or manual or args.save_episode else None
                        action = observation.qpos[None] if manual else session.tick(lambda: observation if observation is not None else capture(),args.prompt)
                        if not inputs.empty():
                            continue
                        if action is not None:
                            if not manual:
                                arm.set_action(action,controllers,commands,dual)
                            if recorder:
                                recorder.append_sample({**observation.recording_fields(),'commanded_action':action[0],
                                    'policy_action':action[0],'intervention_active':manual})
                            if args.save_episode:
                                for name in runtime.cameras:
                                    video[name].append(observation.images[name].copy())
                            session.committed()
                            executed += 1
                            runtime.timing.record('loop_total', timing_now() - begin)
                            runtime.timing.report_if_due(step=executed-1, query_count=session.queries)
                    wait_until_loop_deadline(begin,dt)
                if manual:
                    hold = capture().qpos[None]
                    arm.set_action(hold,controllers,commands,dual,interpolate=False)
                    arm.apply_gain_mode(controllers,gains,dual,zero_gains=False)
                    manual = False
                if args.save_episode and any(video.values()):
                    directory = Path(args.rollout_dir).expanduser() / task['task_name']
                    directory.mkdir(parents=True, exist_ok=True)
                    arm.save_videos(video,dt,video_path=str(directory/f'{uuid.uuid4().hex}.mp4'))
                if recorder:
                    recorder.stop_episode(retained=True,outputs={'rollout':{'requested':bool(args.record_rollout or args.eval),'decision':'success' if key=='s' else 'failure'},
                        'dagger':{'requested':args.record_dagger_corrections,'decision':'keep'}})
                    recording=False
    except BaseException:
        failed = True  # The failure exit already homed (or disabled in place).
        raise
    finally:
        input_stop.set()
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        def cleanup(label, callback):
            try:
                callback()
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(f'[Cleanup] {label}: {exc}', file=sys.stderr)
        if controllers and not failed:
            cleanup('hold', lambda: arm.set_action(
                np.asarray(arm.get_observations(controllers)['qpos'])[None],
                controllers,commands,dual,interpolate=False))
            if manual:
                cleanup('restore gains', lambda: arm.apply_gain_mode(controllers,gains,dual,zero_gains=False))
        if session:
            cleanup('worker', session.close)
        if pedal:
            cleanup('pedals', pedal.close)
        if recorder:
            if recorder.active_session_id:
                cleanup('drain recording', lambda: recorder.stop_episode(retained=True,outputs={},incomplete=True))
            cleanup('recorder', recorder.close)
        if camera_stop:
            cleanup('cameras', camera_stop.set)
        if controllers and not failed:
            cleanup('home', lambda: arm.reset_controllers_to_home(controllers,dual))
        cleanup('policy connection', client._ws.close)
        if cleanup_errors and not active_error:
            raise RuntimeError('Runtime cleanup failed') from cleanup_errors[0]


run_openpi_rollouts = run_rollouts


def build_parser():
    from config.inference_config import InferenceArgumentParser
    p=InferenceArgumentParser(backend='remote', description=__doc__)
    p.add_argument('--shared-hardware', action='store_true', help='Use the model/role-checked ARX adapters')
    p.add_argument('--task_name');p.add_argument('--task-config',dest='task_config')
    p.add_argument('--remote_host',default='localhost');p.add_argument('--remote_port',type=int,default=8000)
    p.add_argument('--api_key');p.add_argument('--prompt',default='')
    p.add_argument('--control-freq','--fps',dest='fps',type=float);p.add_argument('--legacy-policy-fps',type=float)
    p.add_argument('--query_frequency','-q',type=int)
    p.add_argument('--async_inference','--async-inference',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--no_async_inference',dest='async_inference',action='store_false')
    p.add_argument('--async_prefetch_steps',type=int,default=20)
    p.add_argument('--max-inference-delay-ms',type=float)
    p.add_argument('--rtc',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--rtc_prefix_steps',type=int,default=20)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--dry_run_dataset_dir')
    p.add_argument('--dry_run_episodes',type=int,default=2);p.add_argument('--async_dry_run_steps',type=int,default=200)
    p.add_argument('--live_startup_warmup_steps',type=int,default=DEFAULT_WARMUP_STEPS)
    p.add_argument('--record_rollout',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--record_dagger_corrections',action='store_true')
    p.add_argument('--dagger-footpedal-device',help='Optional exclusive Linux pedal device for generic DAgger 1/2/3 controls')
    p.add_argument('--eval',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--save_episode',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--recording_data_dir',default=str(INFERENCE_RAW_DIR))
    p.add_argument('--rollout_dir',default=str(ROLLOUT_DIR));p.add_argument('--dagger_dir',default=str(DAGGER_DIR))
    p.add_argument('--num_rollouts',type=int,default=10);p.add_argument('--max_timesteps',type=int,default=100000)
    p.add_argument('--start_position',dest='start_position_path',default='calibration/start_position.npz')
    p.add_argument('--no_start_position',dest='start_position_path',action='store_const',const=None)
    p.add_argument('--arm_kp','--kp',type=float,nargs=7,default=list(ACTIVE_KP))
    p.add_argument('--arm_kd','--kd',type=float,nargs=7,default=list(ACTIVE_KD))
    p.add_argument('--timing',action='store_true');p.add_argument('--timing_interval',type=int,default=500)
    return p


def main():
    args=build_parser().parse_args()
    (run_dry_run if args.dry_run else run_rollouts)(args)


if __name__=='__main__':
    main()
