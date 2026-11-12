"""Record native actions from a model/role-declared target provider.

Providers own their input device and coordinate mapping. The recorder owns
capture, execution timing and recording lifecycle through shared contracts.
"""
import argparse
from contextlib import ExitStack
import importlib
import time
import sys
import uuid

from config.task_config import load_task_config
from contracts.robot_contract import require_robot_match
from hardware.arm_factory import build_arms
from hardware.runtime import HardwareRuntime, RuntimeDamping
from hardware.safe_exit import damping_exit


def record_targets(task, provider, *, recorder_factory=None, camera_target=None):
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    from camera_utils.capture import capture_image_from_shared_memory
    from recording.record_episode_single import wait_for_cameras
    import threading
    with ExitStack() as cleanup:
        if callable(getattr(provider, 'close', None)):
            cleanup.callback(provider.close)
        doc = task.as_dict()
        hardware = HardwareRuntime(build_arms(doc['robot']))
        cleanup.callback(hardware.close)
        # Providers explicitly declare their target model and arm roles.
        require_robot_match(hardware.contract, {'robot_contract': provider.robot_contract}, context='target provider')
        dt = task.dt
        for arm in hardware.arms.values():
            arm.validate_target([0] * arm.describe().action_dim, interval_s=dt)
        names = doc['cameras']['names']
        from camera_utils.camera_settings_contract import check_inference_camera_settings, snapshot_camera_settings
        check_inference_camera_settings(task.legacy(), context='collection')
        stop = threading.Event()
        camera_thread = threading.Thread(target=camera_target or capture_image_from_shared_memory,
            args=(stop,), kwargs={'camera_names': names}, daemon=True)
        camera_thread.start()
        def close_cameras():
            stop.set()
            camera_thread.join(timeout=5)
        cleanup.callback(close_cameras)
        wait_for_cameras(names, timeout_sec=15)
        recorder = (recorder_factory or DistributedRecorder)(camera_names=names, data_dir=doc['data']['directory'])
        cleanup.callback(recorder.close)
        request = EpisodeRequest(session_id=uuid.uuid4().hex, data_dir=doc['data']['directory'],
            camera_names=names, record_rollout=True, record_dagger=False,
            rollout_dir=doc['data']['directory'], rollout_task_name=task.name,
            dagger_dir='', dagger_task_name='', dagger_camera_names=[],
            is_double_arm=len(hardware.arms) == 2, fps=task.control_freq,
            metadata={**hardware.recording_metadata(task.control_freq), 'task_name': task.name,
                      'camera_settings': snapshot_camera_settings(names)})
        cleanup.callback(hardware.close)
        hardware.connect()
        recorder.start_episode(request, timeout=15)
        try:
            # Any failure, including Ctrl+C: damping until 'h', then home.
            with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
                for _ in range(task.episode_steps):
                    started = time.monotonic()
                    observation = hardware.capture(names, include_images=True)
                    target = provider(observation)
                    hardware.submit_target(target, interval_s=dt)
                    recorder.append_sample({**observation.recording_fields(), 'commanded_action': target})
                    remaining = dt - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            hardware.close()
            if recorder.active_session_id:
                recorder.stop_episode(retained=True, outputs={'rollout': {
                    'requested': True, 'decision': 'success'}},
                    incomplete=sys.exc_info()[0] is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', required=True)
    parser.add_argument('--target-provider', required=True, help='module:factory; factory(task) returns a native target provider')
    args = parser.parse_args()
    task = load_task_config(args.task_config)
    module, separator, name = args.target_provider.partition(':')
    if not separator or not module or not name:
        parser.error('--target-provider must be module:factory')
    provider = getattr(importlib.import_module(module), name)(task)
    record_targets(task, provider)


if __name__ == '__main__':
    main()
