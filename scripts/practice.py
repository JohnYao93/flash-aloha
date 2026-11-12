#!/usr/bin/env python3
"""Live gravity-compensated manual practice for ARX arms.

    uv run scripts/practice.py --arm both
    uv run scripts/practice.py --arm right

Support the arms before starting and stopping. Ctrl+C (or any failure) enters
damping; press 'h' to return home, after which the controllers close.
"""
import argparse
import importlib
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src',):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hardware.safe_exit import ARXDamping, damping_exit, raising_signals


class ARXPracticeController:
    def __init__(self, controller):
        self._controller = controller
        try:
            gain = controller.get_gain()
            gain.kp()[:] = 0.0
            gain.kd()[:] = 0.0
            gain.gripper_kp = gain.gripper_kd = 0.0
            controller.set_gain(gain)
        except BaseException:
            self.close()
            raise

    @property
    def controller(self):
        return self._controller

    def practice_step(self):
        # The native ARX worker applies gravity; this also surfaces SDK errors.
        self._controller.get_joint_state()

    def close(self):
        if self._controller is not None:
            controller, self._controller = self._controller, None
            try:
                controller.set_to_damping()
            finally:
                # Releasing the SDK owner joins its native threads.
                del controller


def build_arx_controllers(interfaces, model):
    from utils.hardware_utils import configure_controller
    directory = ROOT / 'src/hardware/arx5-sdk/python'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    sdk = importlib.import_module('arx5_interface')
    robot = sdk.RobotConfigFactory.get_instance().get_config(model)
    config = sdk.ControllerConfigFactory.get_instance().get_config('joint_controller', robot)
    configure_controller(config)
    config.gravity_compensation = True
    config.friction_compensation = False
    config.background_send_recv = True
    controllers = {}
    try:
        for role, interface in interfaces.items():
            print(f'Initializing ARX {role} arm on {interface}...', flush=True)
            controllers[role] = ARXPracticeController(sdk.Arx5JointController(robot, config, interface))
    except BaseException:
        close_controllers(controllers)
        raise
    return controllers


def close_controllers(controllers):
    success = True
    for role, controller in controllers.items():
        try:
            controller.close()
        except Exception as exc:
            success = False
            print(f'{role}: controller shutdown failed: {exc}', file=sys.stderr)
    return success


def selected_interfaces(args):
    roles = ('left', 'right') if args.arm == 'both' else (args.arm,)
    interfaces = {'left': args.left_can_interface, 'right': args.right_can_interface}
    if args.can_interface:
        ports = [port.strip() for port in args.can_interface.split(',')]
        if len(ports) != len(roles) or not all(ports):
            raise ValueError('--can-interface needs one interface per selected arm (both: can0,can1)')
        interfaces.update(zip(roles, ports))
    result = {role: interfaces[role] for role in roles}
    if not all(port.strip() for port in result.values()):
        raise ValueError('CAN interface names must not be empty')
    if len(set(result.values())) != len(result):
        raise ValueError('Each arm needs a distinct CAN interface')
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', choices=('arx',), default='arx', help='Backend (ARX)')
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='left', help='Arm selection (default: left)')
    parser.add_argument('--can-interface', help='Optional override: can0 for one arm or can0,can1 for both')
    parser.add_argument('--left-can-interface', default='can0')
    parser.add_argument('--right-can-interface', default='can1')
    parser.add_argument('--arx-model', choices=('L5', 'Trossen_L5'), default='L5')
    parser.add_argument('--controller-fps', type=float, default=200.0, help='Python update rate (default: 200)')
    parser.add_argument('--duration', type=float, default=0.0, help='Seconds; 0 runs until Ctrl+C')
    return parser


def run(args):
    limit = 500
    if not math.isfinite(args.controller_fps) or not 0 < args.controller_fps <= limit:
        raise ValueError(f'--controller-fps must be finite and in (0, {limit}]')
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError('--duration must be finite and nonnegative')
    interfaces = selected_interfaces(args)
    controllers = {}
    result = 0
    with raising_signals():
        def damping_target():
            if not controllers:
                return None
            return ARXDamping([controller.controller for controller in controllers.values()])

        try:
            # Ctrl+C or any failure: damping until 'h', then home, then disable.
            with damping_exit(damping_target):
                print('Support the arms during startup and shutdown.', flush=True)
                controllers.update(build_arx_controllers(interfaces, args.arx_model))
                for controller in controllers.values():
                    controller.practice_step()
                print('Gravity compensation active: ' + ', '.join(
                    f'{role}={port}' for role, port in interfaces.items())
                    + ". Ctrl+C: damping, then 'h' returns home.", flush=True)
                started = time.monotonic()
                while not args.duration or time.monotonic() - started < args.duration:
                    tick = time.monotonic()
                    for controller in controllers.values():
                        controller.practice_step()
                    time.sleep(max(0.0, 1.0 / args.controller_fps - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            print('\nStopping practice...', flush=True)
        except Exception as exc:
            print(f'Practice stopped: {exc}', file=sys.stderr)
            result = 1
        finally:
            if not close_controllers(controllers):
                result = 1
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
