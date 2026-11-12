"""Offline runtime contracts with fake policies, synthetic videos and no hardware."""
from types import SimpleNamespace
import ast
from pathlib import Path
from data_processing import episode_store
import numpy as np
import pytest
import yaml

from test_raw_training import make_raw
from data_processing.episode_inputs import discover_episodes
from inference.policy_chunks import maybe_merge_async_result, validate_overlap_budget, InferenceDelayExceeded


@pytest.mark.parametrize('elapsed', [20,25,40])
def test_all_handoffs_use_configured_overlap(elapsed):
    args=SimpleNamespace(fps=100,query_frequency=50)
    chunk=np.repeat(np.arange(100,dtype='f4')[None,:,None],7,axis=2)
    result=dict(action_chunk=chunk,query_index=1,submit_chunk_step=0,query_timing={'query_total':.1})
    merged=maybe_merge_async_result(args,pending_async_result=result,all_actions=chunk,t=elapsed,
        current_chunk_step=elapsed,current_chunk_update_t=0,current_chunk_start_index=0,async_prefetch_steps=20)
    assert merged[1][0,0,0]==20
    assert merged[-1]['start_index']==20


def test_overlap_must_exceed_delay_and_overruns_fail():
    validate_overlap_budget(100,21,200)
    with pytest.raises(ValueError,match='greater'):
        validate_overlap_budget(100,20,200)
    chunk=np.zeros((1,100,7),np.float32)
    result=dict(action_chunk=chunk,query_index=1,submit_chunk_step=0,latency_s=.21)
    with pytest.raises(InferenceDelayExceeded):
        maybe_merge_async_result(SimpleNamespace(fps=100,query_frequency=50),pending_async_result=result,
            all_actions=chunk,t=25,current_chunk_step=25,current_chunk_update_t=0,
            current_chunk_start_index=0,async_prefetch_steps=20)


def test_raw_discovery_rejects_wrong_fps_and_episode_only(tmp_path):
    raw=make_raw(tmp_path/'raw')
    assert discover_episodes(raw,100)==[str(raw)]
    with pytest.raises(ValueError,match='FPS'):
        discover_episodes(raw,50)
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'episode_00000.zarr').touch()
    with pytest.raises(FileNotFoundError,match='raw'):
        discover_episodes(legacy)


@pytest.mark.parametrize('module_name', ['inference.inference_remote'])
def test_native_dry_run_uses_raw_frames_without_episode(tmp_path,monkeypatch,module_name):
    import importlib
    native = importlib.import_module(module_name)
    for name in ('ACTION_DIM','STATE_DIM','CAMERA_NAMES','IS_DOUBLE_ARM','REQUEST_SCHEMA',
                 'OPENPI_MODEL_FAMILY'):
        if hasattr(native,name):
            monkeypatch.setattr(native,name,getattr(native,name))
    from data_processing.episode_data import clear_episode_cache, Episode
    raw=make_raw(tmp_path/'raw')
    monkeypatch.setenv('FLASH_ALOHA_VIDEO_INDEX_CACHE',str(tmp_path/'index'))
    task=dict(schema_version=1,task=dict(name='test_native'),fps=100,
        data=dict(directory=str(raw)),robot=dict(type='arx',arms=dict(left=dict(can_interface='can0'))),
        cameras=dict(names=['video_left','video_overhead']),collection=dict(max_duration_seconds=10),
        training=dict(observations=['qpos'],outputs=['action'],state_dim=7,action_dim=7))
    taskpath=tmp_path/'task.yaml';taskpath.write_text(yaml.safe_dump(task))
    class Client:
        server_metadata=dict(policy_family='flash_aloha',request_schema='native',state_dim=7,
            action_dim=7,action_rate_hz=100,prediction_horizon=100,action_command_style='absolute')
        def __init__(self,*args):
            self._ws=SimpleNamespace(close=lambda:None)
        def infer_with_timing(self,request):
            calls.append(request)
            return {'actions':np.zeros((100,7),np.float32)},{}
    calls=[]
    monkeypatch.setattr(native,'WebsocketOpenPIClient',Client)
    monkeypatch.setattr(native,'check_inference_camera_settings',lambda _:None,raising=False)
    import camera_utils.camera_settings_contract as camera_settings_contract
    monkeypatch.setattr(camera_settings_contract,'check_inference_camera_settings',lambda _:None)
    args=native.build_parser().parse_args(['--task-config',str(taskpath),'--dry-run','--no_async_inference'])
    try:
        native.run_dry_run(args)
        indices = [0] if module_name=='inference.inference_remote' else [0,4,8]
        assert len(calls)==len(indices)
        with Episode(raw) as episode:
            for index, request in zip(indices,calls):
                np.testing.assert_array_equal(request['observation/images/video_left'],episode.frame_bundle(index)['video_left'])
                np.testing.assert_array_equal(request['observation/state'],episode.qpos[index])
    finally:
        clear_episode_cache()


@pytest.mark.parametrize('tampered', [False, True])
def test_openpi_dry_run_sends_contract_letterboxed_images(tmp_path,monkeypatch,tampered):
    from inference import inference_remote as native
    from contracts import standard_image_processing as images
    from data_processing.episode_data import clear_episode_cache, Episode
    for name in ('ACTION_DIM','STATE_DIM','CAMERA_NAMES'):
        monkeypatch.setattr(native,name,getattr(native,name))
    raw=make_raw(tmp_path/'raw')
    monkeypatch.setenv('FLASH_ALOHA_VIDEO_INDEX_CACHE',str(tmp_path/'index'))
    task=dict(schema_version=1,task=dict(name='test_native'),fps=100,
        data=dict(directory=str(raw)),robot=dict(type='arx',arms=dict(left=dict(can_interface='can0'))),
        cameras=dict(names=['video_left','video_overhead']),collection=dict(max_duration_seconds=10),
        training=dict(observations=['qpos'],outputs=['action'],state_dim=7,action_dim=7))
    taskpath=tmp_path/'task.yaml';taskpath.write_text(yaml.safe_dump(task))
    contract=images.openpi_image_contract({'video_left':dict(request_key='observation/wrist_image',source_shape=[48,64,3]),
                                           'video_overhead':dict(request_key='observation/image',source_shape=[48,64,3])})
    if tampered:
        contract['fingerprint']['sha256']='0'*64
    calls=[]
    class Client:
        server_metadata=dict(model_family='openpi',request_schema='openpi',state_dim=7,action_dim=7,
            robot_action_dim=7,model_action_dim=32,action_rate_hz=100,action_horizon=100,
            action_command_style='absolute',image_preprocessing=contract)
        def __init__(self,*args):
            self._ws=SimpleNamespace(close=lambda:None)
        def infer_with_timing(self,request):
            calls.append(request)
            return {'actions':np.zeros((100,32),np.float32)},{}
    monkeypatch.setattr(native,'WebsocketOpenPIClient',Client)
    import camera_utils.camera_settings_contract as camera_settings_contract
    monkeypatch.setattr(camera_settings_contract,'check_inference_camera_settings',lambda _:None)
    monkeypatch.setattr(native,'check_inference_camera_settings',lambda _:None)
    args=native.build_parser().parse_args(['--task-config',str(taskpath),'--dry-run','--no_async_inference',
                                           '--dry_run_episodes','1','--async_dry_run_steps','1'])
    try:
        if tampered:
            with pytest.raises(ValueError,match='contract mismatch'):
                native.run_dry_run(args)
            assert not calls
            return
        native.run_dry_run(args)
        (request,)=calls
        assert request['client/image_preprocessing_sha256']==contract['fingerprint']['sha256']
        assert not any(key.startswith('observation/images/') for key in request)
        with Episode(raw) as episode:
            frames=episode.frame_bundle(0)
        for name,key in (('video_left','observation/wrist_image'),('video_overhead','observation/image')):
            assert request[key].shape==(448,448,3) and request[key].dtype==np.uint8
            np.testing.assert_array_equal(request[key],images.letterbox_image(frames[name][...,::-1],448,448))
    finally:
        clear_episode_cache()


@pytest.mark.parametrize('config',[{'image_history_len':1},{'policy_config':{'image_history_len':2}},
    {'task_config':{'camera_names':['video_left','video_left_t-1']}}])
def test_old_image_history_checkpoints_fail_explicitly(config):
    from contracts.standard_image_processing import require_single_frame
    with pytest.raises(ValueError,match='history|Historical'):
        require_single_frame(config)










def test_async_request_without_response_fails_at_overlap(monkeypatch):
    import threading
    from inference import policy_chunks as chunks
    from inference.policy_transport import TimingStats
    entered, release = threading.Event(), threading.Event()
    now = [0.0]
    monkeypatch.setattr(chunks, 'timing_now', lambda: now[0])
    def query(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return np.zeros((1,100,7)), {}, (100,7)
    worker = chunks.AsyncOpenPIQueryWorker(policy_client=object(), query_actions=query,
        timing_stats=TimingStats(enabled=False, interval=1), max_delay_s=.2)
    worker.start()
    try:
        assert worker.submit(image_dict={}, qpos=np.zeros(7), prompt='', query_index=0, submit_chunk_step=0)
        assert entered.wait(1)
        now[0] = .2
        with pytest.raises(InferenceDelayExceeded):
            worker.raise_if_failed()
        worker.reset()
        worker.raise_if_failed()  # Reset invalidates the old request's deadline.
    finally:
        release.set()
        worker.stop()


def test_native_async_executes_contiguous_fixed_windows(monkeypatch):
    from inference import inference_remote as native
    from inference.policy_transport import TimingStats
    requests = []
    class Worker:
        def __init__(self, **kwargs): self.request = None
        def start(self): pass
        def stop(self): pass
        def reset(self): self.request = None
        def raise_if_failed(self): pass
        def has_work_pending(self): return self.request is not None
        def submit(self, **request):
            self.request = request
            requests.append(request['submit_rollout_index'])
            return True
        def poll_result(self):
            if self.request is None: return None
            request = self.request
            if request['query_index'] and session.step-request['submit_rollout_index'] < 10:
                return None
            self.request = None
            return dict(request, query_timing={}, latency_s=.1,
                action_chunk=np.repeat((request['qpos'][0]+np.arange(100))[None,:,None],7,axis=2))
    monkeypatch.setattr(native.policy_chunks,'AsyncOpenPIQueryWorker',Worker)
    args=native.build_parser().parse_args(['--fps','100','--query_frequency','50'])
    runtime=SimpleNamespace(args=args,client=object(),timing=TimingStats(enabled=False,interval=1),
        query=lambda:None,rtc_metadata=lambda prefix:{})
    session=native.PolicySession(runtime)
    output=[]
    for tick in range(151):
        action=session.tick(lambda:SimpleNamespace(images={},qpos=np.repeat(session.step,7)), '')
        if action is not None:
            output.append(action[0,0]);session.committed()
    np.testing.assert_array_equal(output,np.arange(150))
    assert requests == [0,30,80,130]
    session.reset()
    assert session.actions is None and session.step == 0
    session.close()


@pytest.mark.parametrize('failure', ['none','query','recording'])
def test_native_live_loop_simulated_cleanup_and_exact_recording(monkeypatch, failure):
    from inference import inference_remote as native
    from inference import inference_direct as arm
    from recording import distributed_recording
    import queue
    import threading
    events, samples = [], []
    args=native.build_parser().parse_args(['--no-async_inference','--no_start_position',
        '--max_timesteps','2','--num_rollouts','1','--live_startup_warmup_steps','0','--record_rollout'])
    args.fps=100;args.save_episode=False
    runtime=SimpleNamespace(task=dict(single_arm=True,single_arm_side='left',single_arm_can='can0',
        can_interface_left='can0',task_name='test'),cameras=['video_left'],action_dim=7,
        timing=SimpleNamespace(record=lambda *a:None,report_if_due=lambda **kw:None))
    monkeypatch.setattr(native,'PolicyRuntime',lambda *a:runtime)
    # A legacy checkpoint: no robot_contract in the server metadata.
    monkeypatch.setattr(native,'WebsocketOpenPIClient',lambda *a:SimpleNamespace(server_metadata={},_ws=SimpleNamespace(close=lambda:events.append('client'))))
    class Session:
        step=queries=0
        def __init__(self,*a): pass
        def reset(self): self.step=0
        def close(self): events.append('worker')
        def tick(self,observe,prompt):
            obs=observe()
            if failure=='query': raise ValueError('query failed')
            return obs.qpos[None]
        def committed(self): self.step+=1
    monkeypatch.setattr(native,'PolicySession',Session)
    keys=queue.Queue();keys.put('c')
    monkeypatch.setattr(native,'Queue',lambda:keys)
    monkeypatch.setattr(native,'threading',SimpleNamespace(Event=threading.Event,
        Thread=lambda **kw:SimpleNamespace(start=lambda:None)))
    monkeypatch.setattr(native,'wait_until_loop_deadline',lambda *a:None)
    monkeypatch.setattr(arm,'setup_cameras_and_robot',lambda **kw:SimpleNamespace(set=lambda:events.append('cameras')))
    class FakeARX:
        # Failure exit (docs/safe_exit.md): native damping, the operator presses 'h', reset_to_home.
        def set_to_damping(self): events.append('damping'); keys.put('h')
        def get_controller_config(self): return SimpleNamespace(default_gripper_kp=1.0,default_gripper_kd=0.1)
        def get_gain(self): return SimpleNamespace(gripper_kp=0.0,gripper_kd=0.0)
        def set_gain(self,gain): pass
        def get_joint_state(self): return None
        def reset_to_home(self): events.append('home')
    monkeypatch.setattr(arm,'init_arm',lambda **kw:(SimpleNamespace(joint_dof=6),FakeARX(),None))
    monkeypatch.setattr(arm,'snapshot_controller_gain',lambda *a:object())
    monkeypatch.setattr(arm,'move_to_start_action',lambda *a,**kw:None)
    monkeypatch.setattr(arm,'set_action',lambda *a,**kw:events.append('command'))
    monkeypatch.setattr(arm,'get_observations',lambda *a:{'qpos':np.zeros(7),'qvel':np.zeros(7)})
    monkeypatch.setattr(arm,'reset_controllers_to_home',lambda *a:events.append('home'))
    observations=[]
    def capture(*a,**kw):
        stamp=len(observations)+1
        fields=dict(timestamp=stamp,qpos=np.full(7,stamp),qvel=np.zeros(7),camera_observation={'exact_frame_id':stamp})
        obs=SimpleNamespace(qpos=fields['qpos'],images={'video_left':np.zeros((2,2,3),np.uint8)},
            recording_fields=lambda:fields)
        observations.append(fields);return obs
    monkeypatch.setattr(native.standard_capture,'capture_observation',capture)
    class Recorder:
        active_session_id=None
        def __init__(self,**kw): pass
        def start_episode(self,*a,**kw): self.active_session_id='test';events.append('record_start')
        def pause(self,*a): pass
        def resume(self,*a): pass
        def set_metadata(self,*a,**kw): pass
        def append_sample(self,sample):
            samples.append(sample)
            if failure=='recording': raise ValueError('recording failed')
        def stop_episode(self,**kw): self.active_session_id=None;events.append('record_stop')
        def close(self): events.append('recorder')
    monkeypatch.setattr(distributed_recording,'DistributedRecorder',Recorder)
    if failure=='none': native.run_rollouts(args)
    else:
        with pytest.raises(ValueError,match=failure+' failed'):
            native.run_rollouts(args)
    assert all(event in events for event in ('worker','record_stop','recorder','cameras','home','client'))
    assert events.count('home')==1  # The failure exit's home replaces the normal one.
    if failure=='none':
        assert 'damping' not in events
        assert events.index('record_stop')<events.index('cameras')<events.index('home')<events.index('client')
    else:
        # The arm is made safe first (damping until 'h', home), then recording and cameras close.
        assert events.index('damping')<events.index('home')<events.index('record_stop')<events.index('cameras')<events.index('client')
    for sample in samples:
        original=observations[sample['timestamp']-1]
        assert sample['camera_observation'] is original['camera_observation']
        np.testing.assert_array_equal(sample['commanded_action'],original['qpos'])


def test_raw_metadata_and_goal_history_respect_saved_ids_and_gaps():
    from data_processing.episode_inputs import row_metadata, causal_goal_row
    root=SimpleNamespace(attrs={'openpi_prompt':'default'},
        control={'metadata_id':np.array([0,0,1,1,1,1])},
        retained_indices=np.array([0,1,2,4,5]), runs=np.array([0,0,0,1,1]),
        manifest={'metadata_segments':[{'openpi_prompt':'first'},{'openpi_prompt':'second'}]})
    assert row_metadata(root,0)['openpi_prompt']=='first'
    assert row_metadata(root,2)['openpi_prompt']=='second'
    rng=np.random.default_rng(1)
    assert causal_goal_row(root,2,rng)==2  # New primitive.
    assert causal_goal_row(root,3,rng)==3  # Removed row prevents crossing the gap.
    assert {causal_goal_row(root,4,rng) for _ in range(40)}=={3,4}
    root.manifest['metadata_segments']=[]
    with pytest.raises(ValueError,match='Missing metadata segment'):
        row_metadata(root,2)


