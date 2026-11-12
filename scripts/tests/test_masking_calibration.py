"""Masking extrinsic calibration: the math and the tool's solve step on rendered checkerboards. No robot."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from config.masking_config import DEFAULT_MASKING_CONFIG, load_masking_config
from utils import masking_calibration as math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/setup'))
import calibrate_masking  # noqa: E402

BOARD, SQUARE = (10, 7), 0.025


def pose(rotation_vector, translation):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    T[:3, 3] = translation
    return T


def close(actual, expected, metres, degrees):
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).as_rotvec()))
    return np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < metres and angle < degrees


def camera_views(count, seed):
    """Board poses in a camera that looks at it from different sides, about 0.4 m away."""
    rng = np.random.default_rng(seed)
    centre = np.array([(BOARD[0] - 1) * SQUARE / 2, (BOARD[1] - 1) * SQUARE / 2, 0])
    views = []
    for _ in range(count):
        rotation = Rotation.from_rotvec(rng.uniform(-0.45, 0.45, 3)).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.38, 0.5)]) - rotation @ centre
        views.append(T)
    return views


def render_board(T_camera_board, camera_matrix, size, pixels=40):
    """An image of the checkerboard (white margin, inner corners on the board's grid) seen from a camera."""
    columns, rows = BOARD[0] + 1, BOARD[1] + 1
    canvas = np.full(((rows + 2) * pixels, (columns + 2) * pixels), 255, np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                canvas[(row + 1) * pixels:(row + 2) * pixels, (column + 1) * pixels:(column + 2) * pixels] = 0
    # Canvas pixel -> board metres (the first inner corner is the board origin), then the board plane -> image.
    to_metres = np.array([[SQUARE / pixels, 0, -2 * SQUARE], [0, SQUARE / pixels, -2 * SQUARE], [0, 0, 1]])
    homography = camera_matrix @ T_camera_board[:3, [0, 1, 3]] @ to_metres
    image = cv2.warpPerspective(canvas, homography, size, borderValue=180, flags=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_hand_eye_and_base_solvers_recover_known_transforms():
    T_ee_camera = pose([0.2, -1.3, 0.1], [-0.13, -0.005, 0.1])
    T_base_board = pose([0.05, 0.02, 0.4], [0.45, -0.1, 0.02])
    T_overhead_base = pose([3.1, 0.05, -0.08], [0.38, 0.0, 0.78])
    boards = camera_views(12, seed=0)
    arm_poses = [T_base_board @ np.linalg.inv(view) @ np.linalg.inv(T_ee_camera) for view in boards]
    assert close(math.solve_wrist_camera(arm_poses, boards), T_ee_camera, 1e-6, 1e-4)
    overhead = [T_overhead_base @ T_base_board] * len(boards)
    average, individual = math.solve_base_in_overhead(overhead, boards, T_ee_camera, arm_poses)
    assert close(average, T_overhead_base, 1e-9, 1e-6) and len(individual) == len(boards)
    with pytest.raises(ValueError, match='at least three'):
        math.solve_wrist_camera(arm_poses[:2], boards[:2])
    # Averaging is a mean in translation and on the rotation manifold.
    half = math.average_transforms([pose([0, 0, 0.2], [0, 0, 0]), pose([0, 0, 0.4], [0.2, 0, 0])])
    assert close(half, pose([0, 0, 0.3], [0.1, 0, 0]), 1e-9, 1e-6)


def test_board_detection_and_pose_from_a_rendered_view():
    matrix = np.array([[640.0, 0, 320], [0, 640.0, 180], [0, 0, 1]])
    T = camera_views(1, seed=3)[0]
    image = render_board(T, matrix, (640, 360))
    corners = math.detect_board(image, BOARD)
    assert corners is not None and len(corners) == BOARD[0] * BOARD[1]
    assert close(math.board_pose(corners, BOARD, SQUARE, matrix, np.zeros(5)), T, 0.004, 1.0)
    assert math.detect_board(np.full((360, 640, 3), 128, np.uint8), BOARD) is None


def test_tool_solves_both_calibrations_from_saved_views(tmp_path, capsys):
    root = tmp_path / 'masking'
    shutil.copytree(DEFAULT_MASKING_CONFIG.parent, root)
    document = yaml.safe_load((root / 'masking.yaml').read_text())
    document.pop('definitions', None)
    document['calibration']['urdf'] = load_masking_config()['calibration']['urdf']
    (root / 'masking.yaml').write_text(yaml.safe_dump(document, sort_keys=False))
    setup = load_masking_config(root / 'masking.yaml')['calibration']
    arm = setup['arms']['left']
    scale = 4  # views are captured at the camera's resolution and resized to the calibrated size
    size = (setup['image_size'][0] * scale, setup['image_size'][1] * scale)
    wrist_matrix = np.diag([scale, scale, 1.0]) @ np.load(arm['intrinsics'], allow_pickle=True).item()['camera_matrix']
    overhead_matrix = np.diag([scale, scale, 1.0]) @ np.load(
        setup['overhead_camera']['intrinsics'], allow_pickle=True).item()['camera_matrix']
    T_ee_camera = pose([0.2, -1.3, 0.1], [-0.13, -0.005, 0.1])
    T_overhead_base = pose([3.1, 0.05, -0.08], [0.2, 0.0, 0.6])
    T_overhead_board = pose([0.2, -0.15, 0.3], [-0.12, -0.06, 0.45])  # the board faces the overhead camera
    T_base_board = np.linalg.inv(T_overhead_base) @ T_overhead_board

    def write_views(directories, count, seed):
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        for index, view in enumerate(camera_views(count, seed)):
            cv2.imwrite(str(directories['wrist'] / f'pose_{index}_image.png'), render_board(view, wrist_matrix, size))
            np.save(directories['wrist'] / f'pose_{index}_transform.npy',
                    T_base_board @ np.linalg.inv(view) @ np.linalg.inv(T_ee_camera))
            if 'overhead' in directories:
                cv2.imwrite(str(directories['overhead'] / f'pose_{index}_image.png'),
                            render_board(T_overhead_board, overhead_matrix, size))

    Path(arm['camera_on_end_effector']).unlink()
    Path(arm['base_in_overhead']).unlink()
    write_views(calibrate_masking.view_directories(arm, 'wrist-camera'), 12, seed=1)
    common = ['--arm', 'left', '--masking-config', str(root / 'masking.yaml'), '--solve-only']
    assert calibrate_masking.main(['wrist-camera', *common]) == 0
    assert close(np.load(arm['camera_on_end_effector']), T_ee_camera, 0.02, 3.0)
    write_views(calibrate_masking.view_directories(arm, 'overhead'), 5, seed=2)
    assert calibrate_masking.main(['overhead', *common]) == 0
    saved = np.load(arm['base_in_overhead'])
    assert close(saved['average_transform'], T_overhead_base, 0.03, 4.0) and len(saved['individual_transforms']) == 5
    assert '5 of 5 saved views are usable' in capsys.readouterr().out
    # The Masker reads what the tool wrote.
    from utils.masking import Masker
    assert Masker('end_effector', root / 'masking.yaml').is_projected
    with pytest.raises(SystemExit, match='declares no arm'):
        calibrate_masking.main(['overhead', '--arm', 'middle', '--masking-config', str(root / 'masking.yaml'), '--solve-only'])


def test_capture_uses_the_existing_arx_setup_and_the_safe_exit():
    import inspect
    source = inspect.getsource(calibrate_masking.capture)
    # The controller, passive mode and home move are the audited ARX setup tool's; failures damp until 'h'.
    assert 'from calibrate_arx_arm import build_controller, load_arx5_interface, safe_home, set_passive_mode' in source
    assert 'with damping_exit(lambda: ARXDamping([controller])):' in source
    assert source.index('damping_exit(') < source.index('controller.reset_to_home()') < source.index('set_passive_mode(controller)')
    assert 'if not failed:' in source and 'safe_home(controller)' in source
