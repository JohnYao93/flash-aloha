"""Extrinsic calibration math for the masking config (no robot, no camera).

Two calibrations feed ``utils.masking``:

- wrist camera on its end effector (eye-in-hand): from several arm poses that view a fixed
  checkerboard, ``solve_wrist_camera`` returns the camera's pose in the end-effector frame.
- arm base in the overhead camera: with the board seen by the overhead camera and the wrist
  camera at once, ``solve_base_in_overhead`` chains overhead -> board -> wrist camera ->
  end effector -> base, averaged over the poses.

Poses are 4x4 matrices; ``T_a_b`` is the pose of frame b in frame a.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def board_points(board_size, square_size):
    """Inner-corner coordinates (N, 3) of a checkerboard with ``board_size`` (columns, rows) inner corners."""
    points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
    return points


def detect_board(image, board_size):
    """Checkerboard inner corners (N, 1, 2) in a BGR image, or None when the board is not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    return corners if found else None


def board_pose(corners, board_size, square_size, camera_matrix, dist_coeffs):
    """Pose of the board in the camera frame from its detected corners."""
    ok, rvec, tvec = cv2.solvePnP(board_points(board_size, square_size), corners, camera_matrix, dist_coeffs)
    if not ok:
        raise ValueError('solvePnP failed for a detected board')
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(-1)
    return T


def solve_wrist_camera(T_base_ee, T_camera_board):
    """Camera pose in the end-effector frame from paired arm poses and board poses (hand-eye, AX = XB).

    Park and Martin's closed form over every pair of views: with A the end effector's motion
    and B the camera's motion between two views, A X = X B. The rotation is the least-squares
    alignment of the motions' rotation vectors, the translation a linear least squares.
    """
    if len(T_base_ee) != len(T_camera_board) or len(T_base_ee) < 3:
        raise ValueError('Hand-eye calibration needs at least three paired arm and board poses')
    motions = []
    for i in range(len(T_base_ee)):
        for j in range(i + 1, len(T_base_ee)):
            motions.append((np.linalg.inv(T_base_ee[j]) @ T_base_ee[i],
                            T_camera_board[j] @ np.linalg.inv(T_camera_board[i])))
    M = np.zeros((3, 3))
    for A, B in motions:
        M += np.outer(Rotation.from_matrix(B[:3, :3]).as_rotvec(), Rotation.from_matrix(A[:3, :3]).as_rotvec())
    eigenvalues, eigenvectors = np.linalg.eigh(M.T @ M)
    if eigenvalues.min() < 1e-12:
        raise ValueError('Hand-eye calibration needs arm rotations about at least two different axes')
    rotation = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T @ M.T
    left = np.vstack([A[:3, :3] - np.eye(3) for A, _ in motions])
    right = np.concatenate([rotation @ B[:3, 3] - A[:3, 3] for A, B in motions])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.linalg.lstsq(left, right, rcond=None)[0]
    return T


def average_transforms(transforms):
    """Mean translation and normalized mean quaternion (same hemisphere) of 4x4 poses."""
    quaternions = [Rotation.from_matrix(T[:3, :3]).as_quat() for T in transforms]
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[index]) < 0:
            quaternions[index] = -quaternions[index]
    mean = np.mean(quaternions, axis=0)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(mean / np.linalg.norm(mean)).as_matrix()
    T[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
    return T


def solve_base_in_overhead(T_overhead_board, T_wrist_camera_board, T_ee_camera, T_base_ee):
    """(average, individual) arm-base poses in the overhead camera frame, one per paired view."""
    if not (len(T_overhead_board) == len(T_wrist_camera_board) == len(T_base_ee)) or not T_base_ee:
        raise ValueError('Each view needs the board in both cameras and the arm pose')
    individual = [overhead @ np.linalg.inv(wrist) @ np.linalg.inv(T_ee_camera) @ np.linalg.inv(base_ee)
                  for overhead, wrist, base_ee in zip(T_overhead_board, T_wrist_camera_board, T_base_ee)]
    return average_transforms(individual), individual
