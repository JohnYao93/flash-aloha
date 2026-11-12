#!/usr/bin/env python3
"""Extrinsic calibration for projected masks (operator-run; moves no arm by itself).

Writes the two per-arm files the masking config (calibration/masking/masking.yaml) names:

    # 1. Wrist camera on its end effector: view a fixed checkerboard from ~15 arm poses.
    python scripts/setup/calibrate_masking.py wrist-camera --arm left
    # 2. Arm base in the overhead camera: the board visible to the overhead and wrist cameras.
    python scripts/setup/calibrate_masking.py overhead --arm left

The arm is homed, then put in gravity-compensated passive mode: guide it by hand, press 's'
in the camera window to save a view and 'q' to finish; it then returns home. Ctrl+C or any
failure enters damping until 'h' homes the arm (docs/safe_exit.md). Stop the camera servers
first: this tool opens the camera devices itself. --solve-only recomputes from saved views
without connecting to the arm or cameras.

Intrinsics come first (scripts/cameras/camera_int.py, at the masking config's image size).
The arm model, links, cameras, image size and output files are read from the masking config.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from config.masking_config import load_masking_config  # noqa: E402
from utils import masking_calibration as calibration_math  # noqa: E402

DEFAULT_CAN_INTERFACES = {'left': 'can0', 'right': 'can1'}
WINDOW = 'Masking calibration'


def intrinsics(path):
    data = np.load(path, allow_pickle=True).item()
    return np.asarray(data['camera_matrix'], dtype=np.float64), np.asarray(data['dist_coeffs'], dtype=np.float64)


def view_directories(arm_config, mode):
    """Where a mode's saved views live: beside the file it writes."""
    if mode == 'wrist-camera':
        return {'wrist': Path(arm_config['camera_on_end_effector']).parent}
    root = Path(arm_config['base_in_overhead']).parent
    return {'wrist': root / 'robot', 'overhead': root / 'head'}


def capture_views(controller, chain, cameras, directories, count):
    """Show the cameras; 's' saves every camera's frame and the arm's end-effector pose, 'q' finishes."""
    from utils.masking import get_fk_matrix
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < count:
        frames = {}
        for name, capture in cameras.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Failed to read a frame from the {name} camera')
            frames[name] = frame
        height = min(frame.shape[0] for frame in frames.values())
        cv2.imshow(WINDOW, np.hstack([cv2.resize(frame, (frame.shape[1] * height // frame.shape[0], height))
                                      for frame in frames.values()]))
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            joints = np.asarray(controller.get_joint_state().pos()).copy()
            for name, frame in frames.items():
                cv2.imwrite(str(directories[name] / f'pose_{saved}_image.png'), frame)
            np.save(directories['wrist'] / f'pose_{saved}_transform.npy', get_fk_matrix(joints, chain))
            print(f'Saved view {saved}: joints {np.round(joints, 3)}')
            saved += 1
        elif key == ord('q'):
            break
    return saved


def capture(args, setup, arm_config, directories):
    """Home, passive gravity compensation, capture, home; every failure goes through the safe exit."""
    from calibrate_arx_arm import build_controller, load_arx5_interface, safe_home, set_passive_mode
    from hardware.safe_exit import ARXDamping, damping_exit
    from utils.masking import create_kdl_chain
    chain = create_kdl_chain(setup['urdf'], setup['links']['end_effector'])
    devices = {'wrist': arm_config['wrist_camera']}
    if args.mode == 'overhead':
        devices['overhead'] = setup['overhead_camera']['name']
    can_interface = args.can_interface or DEFAULT_CAN_INTERFACES[args.arm]
    arx5 = load_arx5_interface()
    _, _, controller = build_controller(arx5, args.model, can_interface, args.controller_fps)
    print(f'Connected to the {args.arm} ARX arm on {can_interface} (model {args.model}).')
    print("Ctrl+C or any failure: damping, then 'h' returns the arm home.")
    cameras, failed = {}, False
    try:
        with damping_exit(lambda: ARXDamping([controller])):
            controller.reset_to_home()
            set_passive_mode(controller)
            for name, device in devices.items():
                cameras[name] = cv2.VideoCapture(f'/dev/{device}')
                if not cameras[name].isOpened():
                    raise RuntimeError(f'Cannot open /dev/{device}; stop the camera servers first')
            print("Guide the arm by hand. 's' saves a view, 'q' finishes.")
            saved = capture_views(controller, chain, cameras, directories, args.poses)
            print(f'{saved} views saved')
    except KeyboardInterrupt:
        failed = True
        print('\nCalibration interrupted.')
    except BaseException:
        failed = True
        raise
    finally:
        for camera in cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        if not failed:  # The failure exit already homed (or disabled in place).
            safe_home(controller)
    return not failed


def load_views(directory, size):
    """[(index, image resized to the calibrated size)] for every saved view in a directory."""
    views = {}
    for path in directory.glob('pose_*_image.png'):
        image = cv2.imread(str(path))
        views[int(path.name.split('_')[1])] = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    return views


def solve(args, setup, arm_config, directories):
    board = (args.board_columns, args.board_rows)
    size = setup['image_size']
    wrist_matrix, wrist_distortion = intrinsics(arm_config['intrinsics'])
    wrist_views = load_views(directories['wrist'], size)
    overhead_views = load_views(directories['overhead'], size) if args.mode == 'overhead' else None
    if args.mode == 'overhead':
        overhead_matrix, overhead_distortion = intrinsics(setup['overhead_camera']['intrinsics'])
    arm_poses, wrist_boards, overhead_boards = [], [], []
    for index in sorted(wrist_views):
        corners = calibration_math.detect_board(wrist_views[index], board)
        other = (calibration_math.detect_board(overhead_views[index], board)
                 if overhead_views is not None and index in overhead_views else None)
        if corners is None or (args.mode == 'overhead' and other is None):
            print(f'View {index}: board not found in every camera; skipped')
            continue
        arm_poses.append(np.load(directories['wrist'] / f'pose_{index}_transform.npy'))
        wrist_boards.append(calibration_math.board_pose(corners, board, args.square_size, wrist_matrix, wrist_distortion))
        if args.mode == 'overhead':
            overhead_boards.append(calibration_math.board_pose(other, board, args.square_size, overhead_matrix,
                                                               overhead_distortion))
    print(f'{len(arm_poses)} of {len(wrist_views)} saved views are usable')
    if args.mode == 'wrist-camera':
        result = calibration_math.solve_wrist_camera(arm_poses, wrist_boards)
        np.save(arm_config['camera_on_end_effector'], result)
        print(f"Wrist camera pose in the end-effector frame -> {arm_config['camera_on_end_effector']}\n{np.round(result, 4)}")
    else:
        average, individual = calibration_math.solve_base_in_overhead(
            overhead_boards, wrist_boards, np.load(arm_config['camera_on_end_effector']), arm_poses)
        np.savez(arm_config['base_in_overhead'], individual_transforms=individual, average_transform=average)
        spread = np.ptp([T[:3, 3] for T in individual], axis=0)
        print(f"Arm base pose in the overhead camera frame -> {arm_config['base_in_overhead']}\n{np.round(average, 4)}")
        print(f'Translation spread across views (m): {np.round(spread, 4)}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('wrist-camera', 'overhead'))
    parser.add_argument('--arm', required=True, help='Arm name in the masking config (e.g. left, right)')
    parser.add_argument('--masking-config', help='Masking config YAML (default: calibration/masking/masking.yaml)')
    parser.add_argument('--can-interface', help='CAN interface (default: can0 for left, can1 for right)')
    parser.add_argument('--model', default='L5', help='ARX SDK model name')
    parser.add_argument('--controller-fps', type=float, default=500.0, help='ARX controller rate')
    parser.add_argument('--poses', type=int, help='Views to capture (default: 15 wrist-camera, 5 overhead)')
    parser.add_argument('--board-columns', type=int, default=10, help='Checkerboard inner corners per row')
    parser.add_argument('--board-rows', type=int, default=7, help='Checkerboard inner corners per column')
    parser.add_argument('--square-size', type=float, default=0.025, help='Checkerboard square size in metres')
    parser.add_argument('--solve-only', action='store_true', help='Recompute from saved views; no arm or camera')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup = load_masking_config(args.masking_config).get('calibration')
    if setup is None or args.arm not in setup['arms']:
        raise SystemExit(f'The masking config declares no arm {args.arm!r}')
    arm_config = setup['arms'][args.arm]
    if args.can_interface is None and args.arm not in DEFAULT_CAN_INTERFACES:
        raise SystemExit('--can-interface is required for this arm name')
    args.poses = args.poses or (15 if args.mode == 'wrist-camera' else 5)
    if args.poses < 3:
        raise SystemExit('At least three views are required')
    directories = view_directories(arm_config, args.mode)
    if not args.solve_only and not capture(args, setup, arm_config, directories):
        return 130
    solve(args, setup, arm_config, directories)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
