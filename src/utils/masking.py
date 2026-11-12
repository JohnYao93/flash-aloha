"""Shared image/proprioceptive masking for training, direct inference and visualization.

A mask is applied to the policy's images (after the standard resize, before channel/scaling
conversion) and to the raw qpos, identically in training and deployment. What a strategy
does is data, not code: the masking config (``config.masking_config``; by default
``calibration/masking/masking.yaml``) declares the calibration of one physical setup and
each strategy as a list of operations:

- ``project``: a polygon given in metres in a frame that follows a link of an arm (or sits
  at its base), drawn in the overhead camera or in a wrist camera through forward
  kinematics and the calibration.
- ``region``: a rectangle of a camera image, in fractions of its width and height.
- ``blank``: whole camera images are zeroed.
- ``zero: proprioception``: qpos is zeroed (and then not normalized).

Masks are resolution independent: intrinsics are rescaled from the calibrated image size
to the size of the image being masked, and regions are fractions. The state is one block
per arm in the config's arm order (its joints, then its gripper); operations that refer to
the other arm need two arms.
"""
import cv2
import numpy as np
import torch

from config.masking_config import load_masking_config, masking_contract, require_masking_strategy

__all__ = ['Masker', 'create_kdl_chain', 'get_fk_matrix', 'project']


def create_kdl_chain(urdf_path, tip_link):
    """The KDL chain from a URDF's root link to ``tip_link``."""
    import PyKDL  # noqa: F401  (native kinematics; only projected masks need it)
    from kdl_parser_py.urdf import treeFromUrdfModel
    from urdf_parser_py.urdf import URDF
    robot_urdf = URDF.from_xml_file(str(urdf_path))
    ok, tree = treeFromUrdfModel(robot_urdf)
    if not ok:
        raise ValueError(f'Failed to parse {urdf_path} into a PyKDL tree')
    return tree.getChain(robot_urdf.get_root(), tip_link)


def get_fk_matrix(joint_angles, kdl_chain):
    """4x4 pose of the chain's tip in its base for the arm's joint angles (radians)."""
    import PyKDL
    fk_solver = PyKDL.ChainFkSolverPos_recursive(kdl_chain)
    joints = PyKDL.JntArray(len(joint_angles))
    for i, angle in enumerate(joint_angles):
        joints[i] = float(angle)
    end_frame = PyKDL.Frame()
    fk_solver.JntToCart(joints, end_frame)
    T = np.eye(4)
    for i in range(3):
        for j in range(3):
            T[i, j] = end_frame.M[i, j]
        T[i, 3] = end_frame.p[i]
    return T


def project(points_camera, camera_matrix, limit=3000):
    """Pinhole projection of homogeneous camera-frame columns (4, N) to integer pixels (N, 2).

    Points behind the camera are clamped to 1 cm in front of it and pixels to +/-``limit``,
    so a partly visible polygon still fills its visible part.
    """
    points = np.array(points_camera[:3], dtype=np.float64)
    points[2] = np.clip(points[2], 0.01, np.inf)
    pixels, _ = cv2.projectPoints(points.T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  np.asarray(camera_matrix, dtype=np.float64), np.zeros(5))
    return np.clip(pixels, -limit, limit).reshape(-1, 2).astype(np.int32)


class Masker:
    def __init__(self, masking_strategy, config_path=None):
        self._masking_strategy = require_masking_strategy(masking_strategy, config_path)
        self._config_path = config_path
        self._operations, self._chains, self._setup, self._calibration = [], {}, None, None
        self.arm_names, self._wrist_cameras, self._overhead = (), {}, None
        if masking_strategy is None:
            return
        document = load_masking_config(config_path)
        self._operations = document['strategies'][masking_strategy]
        self._setup = document.get('calibration')
        if self._setup is not None:
            self.arm_names = tuple(self._setup['arms'])
            self._overhead = self._setup['overhead_camera']['name']
            self._wrist_cameras = {name: arm['wrist_camera'] for name, arm in self._setup['arms'].items()}
            self._joints = int(self._setup.get('joints_per_arm', 6))
        if self.is_projected:
            self._load_calibration()

    @property
    def strategy(self):
        return self._masking_strategy

    @property
    def operations(self):
        """The strategy's operations as loaded. Editing a ``project`` operation's corners in place
        changes the mask (used by the masking tuner); training and inference never edit them."""
        return self._operations

    @property
    def cameras(self):
        """The calibrated cameras: each arm's wrist camera, then the overhead camera."""
        return (*self._wrist_cameras.values(), *([self._overhead] if self._overhead else []))

    @property
    def is_projected(self):
        return any('project' in operation for operation in self._operations)

    @property
    def masks_proprioception(self):
        return any(operation.get('zero') == 'proprioception' for operation in self._operations)

    def contract(self):
        """What a checkpoint stores so deployment can require the same mask definition and calibration."""
        return None if self._masking_strategy is None else masking_contract(self._masking_strategy, self._config_path)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_chains'] = {}  # native KDL handles are rebuilt lazily in spawned data workers
        return state

    def _load_calibration(self):
        def intrinsics(path):
            return np.asarray(np.load(path, allow_pickle=True).item()['camera_matrix'], dtype=np.float64)
        try:
            self._calibration = {
                'overhead': intrinsics(self._setup['overhead_camera']['intrinsics']),
                'arms': {name: {
                    'camera_matrix': intrinsics(arm['intrinsics']),
                    # Wrist camera pose in the end-effector frame; arm base pose in the overhead camera frame.
                    'T_ee_camera': np.load(arm['camera_on_end_effector']),
                    'T_head_base': np.load(arm['base_in_overhead'])['average_transform'],
                    'sign': -1.0 if arm.get('mirror_y', False) else 1.0,
                } for name, arm in self._setup['arms'].items()}}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Masking strategy {self._masking_strategy!r} needs its calibration files '
                                    f'(see calibration/masking/README.md): {exc}') from exc

    def _chain(self, frame):
        if frame not in self._chains:
            self._chains[frame] = create_kdl_chain(self._setup['urdf'], self._setup['links'][frame])
        return self._chains[frame]

    # --- geometry -------------------------------------------------------------------------

    def _arm_joints(self, qpos, arm):
        index = self.arm_names.index(arm) * (self._joints + 1)
        values = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if len(values) < index + self._joints:
            raise ValueError(f'Masking strategy {self._masking_strategy!r} needs the joint positions of arms '
                             f'{list(self.arm_names)} ({len(self.arm_names) * (self._joints + 1)} values); '
                             f'got {len(values)}. Use a masking config that declares this task\'s arms.')
        return values[index:index + self._joints]

    def _frame_in_base(self, qpos, arm, frame):
        """The frame's pose in its arm's base: located at the link, with the base's axes."""
        T = np.eye(4)
        if frame != 'base':
            T[:3, 3] = get_fk_matrix(self._arm_joints(qpos, arm), self._chain(frame))[:3, 3]
        return T

    def _camera_matrix(self, matrix, image_size):
        """Intrinsics rescaled from the calibrated image size to ``image_size`` (width, height)."""
        width, height = self._setup['image_size']
        return np.diag([image_size[0] / width, image_size[1] / height, 1.0]) @ matrix

    def _pixel_limit(self, image_size):
        return 3000 * max(1.0, image_size[0] / self._setup['image_size'][0])

    def _project_operation(self, operation, qpos, image_size):
        """{camera name: [polygon pixels (N, 2)]} for one ``project`` operation, one polygon per arm."""
        polygons = {}
        for arm in self.arm_names:
            frame_arm = arm
            if operation['arm'] == 'other':
                others = [name for name in self.arm_names if name != arm]
                if len(others) != 1:
                    continue  # no single "other" arm: the operation does not apply
                frame_arm = others[0]
            own, frame_calibration = self._calibration['arms'][arm], self._calibration['arms'][frame_arm]
            # Float32 corners, as the masks were originally defined.
            corners = np.asarray(operation['corners'], dtype=np.float32).astype(np.float64)
            corners[:, 1] *= own['sign']
            points = self._frame_in_base(qpos, frame_arm, operation['frame']) @ np.vstack([corners.T, np.ones(len(corners))])
            if operation['camera'] == 'overhead':
                camera, matrix = self._overhead, self._calibration['overhead']
                points = frame_calibration['T_head_base'] @ points
            else:
                camera, matrix = self._wrist_cameras[arm], own['camera_matrix']
                T_base_camera = get_fk_matrix(self._arm_joints(qpos, arm), self._chain('end_effector')) @ own['T_ee_camera']
                if frame_arm == arm:
                    points = np.linalg.inv(T_base_camera) @ points
                else:  # through the overhead camera, which sees both arm bases
                    points = np.linalg.inv(own['T_head_base'] @ T_base_camera) @ (frame_calibration['T_head_base'] @ points)
            polygons.setdefault(camera, []).append(
                project(points, self._camera_matrix(matrix, image_size), self._pixel_limit(image_size)))
        return polygons

    def polygons(self, qpos, camera, image_size):
        """Every projected polygon of this strategy in ``camera`` at ``image_size`` (width, height)."""
        result = []
        for operation in self._operations:
            if 'project' in operation:
                result.extend(self._project_operation(operation['project'], qpos, image_size).get(camera, []))
        return result

    def link_pixels(self, qpos, image_size, frame='wrist'):
        """Each arm's link origin in overhead pixels at ``image_size``: {arm: (x, y)}. For checking a calibration."""
        matrix = self._camera_matrix(self._calibration['overhead'], image_size)
        return {arm: project(self._calibration['arms'][arm]['T_head_base'] @ self._frame_in_base(qpos, arm, frame)[:, 3:],
                             matrix, self._pixel_limit(image_size))[0] for arm in self.arm_names}

    def _camera_names(self, reference):
        if reference in ('overhead', 'wrist'):
            if self._setup is None:
                raise ValueError(f'Masking strategy {self._masking_strategy!r} refers to the {reference} camera; '
                                 'without a calibration section, name the camera explicitly')
            return [self._overhead] if reference == 'overhead' else list(self._wrist_cameras.values())
        return [reference]

    # --- application ----------------------------------------------------------------------

    def mask(self, image_data, qpos_data):
        """Masked copies of ``image_data`` ({camera: HWC uint8}) and ``qpos_data``.

        Inputs are not modified. Every image is the observation paired with ``qpos_data``; the
        masks are computed from that qpos alone, so a sample's mask does not depend on call order.
        """
        if not self._operations:
            return image_data, qpos_data
        image_data, copied = dict(image_data), set()

        def writable(name):
            if name not in copied:  # contiguous copy: frames may be read-only or strided views
                image_data[name] = np.array(image_data[name], copy=True, order='C')
                copied.add(name)
            return image_data[name]

        for operation in self._operations:
            if 'project' in operation:
                by_size = {}  # cameras usually share one policy resolution
                for name in self.cameras:
                    if name in image_data:
                        size = tuple(image_data[name].shape[1::-1])
                        if size not in by_size:
                            by_size[size] = self._project_operation(operation['project'], qpos_data, size)
                        for points in by_size[size].get(name, []):
                            cv2.fillPoly(writable(name), [points.reshape(-1, 1, 2)], (0, 0, 0))
            elif 'region' in operation:
                region = operation['region']
                for name in self._camera_names(region['camera']):
                    if name in image_data:
                        height, width = image_data[name].shape[:2]
                        x0, x1 = (int(round(v * width)) for v in region['x'])
                        y0, y1 = (int(round(v * height)) for v in region['y'])
                        writable(name)[y0:y1, x0:x1] = 0
            elif 'blank' in operation:
                for name in self._camera_names(operation['blank']):
                    if name in image_data:
                        image_data[name] = np.zeros_like(image_data[name])
                        copied.add(name)
        if self.masks_proprioception:
            qpos_data = torch.zeros_like(qpos_data) if torch.is_tensor(qpos_data) else np.zeros_like(qpos_data)
        return image_data, qpos_data
