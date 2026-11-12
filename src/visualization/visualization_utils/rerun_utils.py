from pathlib import Path
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from urdf_parser_py.urdf import URDF


def build_failure_frame_set(failure_values=None, failure_frames=None, num_frames=0):
    """Normalize failure annotations into a set of frame indices."""
    frame_set = set()

    def add_indices(values):
        arr = np.asarray(values).reshape(-1)
        for value in arr:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_frames:
                frame_set.add(idx)

    if failure_values is not None:
        failure_arr = np.asarray(failure_values)
        flat = failure_arr.reshape(-1)
        if flat.size == num_frames and (
            failure_arr.dtype == np.bool_
            or np.issubdtype(failure_arr.dtype, np.integer)
            or np.issubdtype(failure_arr.dtype, np.floating)
        ):
            numeric_flat = flat.astype(float)
            finite = numeric_flat[np.isfinite(numeric_flat)]
            is_mask = failure_arr.dtype == np.bool_ or (
                finite.size > 0 and np.all(np.isin(np.unique(finite), [0.0, 1.0]))
            )
            if is_mask:
                for idx in np.flatnonzero(numeric_flat > 0):
                    frame_set.add(int(idx))
            else:
                add_indices(flat)
        else:
            add_indices(flat)

    if failure_frames is not None:
        add_indices(failure_frames)

    return frame_set


def annotation_value_for_frame(values, frame_idx):
    """Return a displayable annotation value for the current frame."""
    if values is None:
        return None

    arr = np.asarray(values)
    if arr.size == 0:
        return None

    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        value = arr.reshape(-1)[0]
    else:
        idx = min(frame_idx, arr.shape[0] - 1)
        value = arr[idx]
        value = np.asarray(value).reshape(-1)
        if value.size == 1:
            value = value[0]
        else:
            value = value.tolist()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_annotation_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        if np.isfinite(value) and abs(value - round(value)) < 1e-5:
            return str(int(round(value)))
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_annotation_value(v) or "" for v in value) + "]"
    return str(value)


def _draw_text_block(image, lines, origin, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    if not lines:
        return image

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    line_gap = 8
    padding = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    block_w = max(width for width, _height in sizes) + padding * 2
    block_h = sum(height for _width, height in sizes) + line_gap * (len(lines) - 1) + padding * 2
    x, y = origin
    h, w = image.shape[:2]
    x = max(0, min(int(x), max(0, w - block_w - 1)))
    y = max(0, min(int(y), max(0, h - block_h - 1)))

    cv2.rectangle(image, (x, y), (x + block_w, y + block_h), bg_color, -1)
    cursor_y = y + padding
    for line, (_width, height) in zip(lines, sizes):
        cursor_y += height
        cv2.putText(
            image,
            line,
            (x + padding, cursor_y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_gap
    return image


def depisode_data_camera_overlay(image, frame_idx, is_subgoal_switch_frame=False, is_failure_frame=False):
    """Draw episode annotations directly onto a camera frame before Rerun logging."""
    if not is_subgoal_switch_frame and not is_failure_frame:
        return image

    annotated = image.copy()
    h, w = annotated.shape[:2]

    if is_failure_frame:
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (255, 0, 0), 4)
        _draw_text_block(
            annotated,
            [f"FAILURE FRAME {frame_idx}"],
            (10, 10),
            bg_color=(120, 0, 0),
            text_color=(255, 255, 255),
        )

    if is_subgoal_switch_frame:
        cv2.rectangle(annotated, (3, 3), (w - 4, h - 4), (0, 255, 255), 3)
        _draw_text_block(
            annotated,
            [f"SUBGOAL SWITCH {frame_idx}"],
            (10, h - 42),
            bg_color=(0, 90, 90),
            text_color=(255, 255, 255),
        )

    return annotated

def parse_urdf(urdf_path=None, robot_type='arx'):
    """
    Parse URDF file and extract robot structure information using urdf_parser_py.
    
    Args:
        urdf_path: Path to URDF file. If None, will try to get from robot_type.
        robot_type: Type of robot ('arx5' or 'arx'). Used to get default URDF path.
    
    Returns:
        Tuple of (links, joints) dictionaries
    """
    # If no path provided, try to get default based on robot type
    if urdf_path is None:
        if robot_type == 'arx':
            # Default ARX5 path (fallback)
            urdf_path = str(Path(__file__).resolve().parents[2] / "hardware/arx5-sdk/models/L5.urdf")
            print(f"[→] Using ARX5 URDF: {urdf_path}")
        else:
            raise ValueError(f"Unknown robot_type: {robot_type}. Must be 'arx' or 'arx5'")
    
    robot = URDF.from_xml_file(urdf_path)
    
    links = {}
    joints = {}
    
    # Parse links
    for link_name, link in robot.link_map.items():
        # Get visual origin
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if link.visual and link.visual.origin:
            if link.visual.origin.xyz:
                xyz = list(link.visual.origin.xyz)
            if link.visual.origin.rpy:
                rpy = list(link.visual.origin.rpy)
        
        links[link_name] = {
            'origin_xyz': xyz,
            'origin_rpy': rpy
        }
    
    # Parse joints
    for joint_name, joint in robot.joint_map.items():
        # Get joint origin
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if joint.origin:
            if joint.origin.xyz:
                xyz = list(joint.origin.xyz)
            if joint.origin.rpy:
                rpy = list(joint.origin.rpy)
        
        # Get joint axis
        axis = [0, 0, 1]  # default
        if joint.axis:
            axis = list(joint.axis)
        
        joints[joint_name] = {
            'type': joint.type,
            'parent': joint.parent,
            'child': joint.child,
            'origin_xyz': xyz,
            'origin_rpy': rpy,
            'axis': axis
        }
    
    return links, joints

def compute_forward_kinematics(joints, joint_positions):
    """Compute forward kinematics to get link transforms"""
    transforms = {'base_link': np.eye(4)}
    
    # Joint order (assuming this matches the action order)
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    
    for i, joint_name in enumerate(joint_names):
        if joint_name in joints:
            joint = joints[joint_name]
            parent_link = joint['parent']
            child_link = joint['child']
            
            # Get parent transform
            if parent_link in transforms:
                parent_transform = transforms[parent_link]
            else:
                parent_transform = np.eye(4)
            
            # Create joint transform
            joint_transform = np.eye(4)
            
            # Translation from joint origin
            joint_transform[:3, 3] = joint['origin_xyz']
            
            # Rotation from joint origin
            if any(joint['origin_rpy']):
                rot_matrix = R.from_euler('xyz', joint['origin_rpy']).as_matrix()
                joint_transform[:3, :3] = rot_matrix
            
            # Apply joint rotation if it's a revolute joint
            if joint['type'] == 'revolute' and i < len(joint_positions):
                joint_angle = joint_positions[i]
                axis = np.array(joint['axis'])
                
                # Create rotation matrix around joint axis
                axis_rot = R.from_rotvec(joint_angle * axis).as_matrix()
                joint_transform[:3, :3] = joint_transform[:3, :3] @ axis_rot
            
            # Combine with parent transform
            transforms[child_link] = parent_transform @ joint_transform
    
    return transforms

def log_robot_model(links, joints, joint_positions=None, gripper_pos=0.0, robot_name="robot"):
    """Log robot model as a stick figure with balls (joints) and sticks (links), and log camera frames."""
    if joint_positions is None:
        joint_positions = [0.0] * 6
    
    # Compute forward kinematics
    transforms = compute_forward_kinematics(joints, joint_positions)
    
    # Get joint positions for visualization
    joint_positions_3d = {}
    
    # Add base position
    if 'base_link' in transforms:
        joint_positions_3d['base'] = transforms['base_link'][:3, 3]
    
    # Get positions for each joint
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    for i, joint_name in enumerate(joint_names):
        if joint_name in joints:
            joint_info = joints[joint_name]
            parent_link = joint_info['parent']
            child_link = joint_info['child']
            
            # Get the transform of the child link (which represents the joint position)
            if child_link in transforms:
                joint_positions_3d[joint_name] = transforms[child_link][:3, 3]
    
    # Log balls for joints
    try:
        # Different colors for different joints
        joint_colors = {
            'base': [0.3, 0.3, 0.3],
            'joint1': [1.0, 0.0, 0.0],
            'joint2': [0.0, 1.0, 0.0], 
            'joint3': [0.0, 0.0, 1.0],
            'joint4': [1.0, 1.0, 0.0],
            'joint5': [1.0, 0.0, 1.0],
            'joint6': [0.0, 1.0, 1.0]
        }
        
        # Adjust colors for right arm to be distinguishable
        if robot_name == "right_arm":
            joint_colors = {
                'base': [0.5, 0.5, 0.5],
                'joint1': [0.8, 0.4, 0.4],
                'joint2': [0.4, 0.8, 0.4], 
                'joint3': [0.4, 0.4, 0.8],
                'joint4': [0.8, 0.8, 0.4],
                'joint5': [0.8, 0.4, 0.8],
                'joint6': [0.4, 0.8, 0.8]
            }
        
        # Log each joint as a ball
        for joint_name, position in joint_positions_3d.items():
            color = joint_colors.get(joint_name, [0.7, 0.7, 0.7])
            rr.log(f"{robot_name}/joints/{joint_name}", 
                   rr.Points3D(
                       positions=[position],
                       colors=[color],
                       radii=[0.02]
                   ))
        
        # Log sticks (line segments) connecting joints
        connections = [
            ('base', 'joint1'),
            ('joint1', 'joint2'),
            ('joint2', 'joint3'),
            ('joint3', 'joint4'),
            ('joint4', 'joint5'),
            ('joint5', 'joint6')
        ]
        
        # Collect all line segments for batch logging
        line_positions = []
        line_colors = []
        
        # Use different colors for different links
        if robot_name == "right_arm":
            link_colors = [
                [0.6, 0.3, 0.3],  # base to joint1
                [0.3, 0.6, 0.3],  # joint1 to joint2
                [0.3, 0.3, 0.6],  # joint2 to joint3
                [0.6, 0.6, 0.3],  # joint3 to joint4
                [0.6, 0.3, 0.6],  # joint4 to joint5
                [0.3, 0.6, 0.6]   # joint5 to joint6
            ]
        else:
            link_colors = [
                [0.8, 0.2, 0.2],  # base to joint1
                [0.2, 0.8, 0.2],  # joint1 to joint2
                [0.2, 0.2, 0.8],  # joint2 to joint3
                [0.8, 0.8, 0.2],  # joint3 to joint4
                [0.8, 0.2, 0.8],  # joint4 to joint5
                [0.2, 0.8, 0.8]   # joint5 to joint6
            ]
        
        for i, (start_joint, end_joint) in enumerate(connections):
            if start_joint in joint_positions_3d and end_joint in joint_positions_3d:
                start_pos = joint_positions_3d[start_joint]
                end_pos = joint_positions_3d[end_joint]
                
                # Add line segment
                line_positions.extend([start_pos, end_pos])
                color = link_colors[i] if i < len(link_colors) else [0.5, 0.5, 0.5]
                line_colors.extend([color, color])
        
        # Log all line segments at once
        if line_positions:
            rr.log(f"{robot_name}/links", 
                   rr.LineStrips3D(
                       strips=[line_positions[i:i+2] for i in range(0, len(line_positions), 2)],
                       colors=line_colors
                   ))
        
        # Add gripper visualization at the end effector (joint6)
        if 'joint6' in joint_positions_3d:
            log_gripper_visualization(transforms, gripper_pos, robot_name)
                           
    except Exception as e:
        print(f"Warning: Failed to log stick figure: {e}")

def log_gripper_visualization(transforms, gripper_pos=0.0, robot_name="robot"):
    """Log gripper visualization as a separate component"""
    if 'link6' not in transforms:
        return
        
    # Get end effector position and orientation
    end_effector_transform = transforms['link6']
    end_effector_pos = end_effector_transform[:3, 3]
    end_effector_orientation = end_effector_transform[:3, :3]
    
    # Apply 90-degree pitch rotation to the gripper orientation
    pitch_rotation = R.from_euler('y', np.pi/2).as_matrix()  # 90 degrees around Y-axis
    end_effector_orientation = end_effector_orientation @ pitch_rotation
    
    # Create gripper finger positions based on gripper_pos
    # gripper_pos typically ranges from 0 (closed) to some positive value (open)
    # Normalize gripper position to a reasonable range
    gripper_opening = max(0, min(1, abs(gripper_pos) * 10))  # Scale and clamp between 0 and 1
    finger_separation = 0.16 * gripper_opening  # Max separation of 16cm when fully open
    finger_length = 0.10  # Length of each finger
    
    # Define gripper base position (slightly forward from joint6)
    gripper_base_local = np.array([0, 0, 0.03])
    gripper_base_world = end_effector_pos + end_effector_orientation @ gripper_base_local
    
    # Define gripper finger positions relative to gripper base
    # Fingers open along the Y-axis of the end effector
    finger1_base_local = np.array([0, finger_separation/2, 0])
    finger2_base_local = np.array([0, -finger_separation/2, 0])
    finger1_tip_local = np.array([0, finger_separation/2, finger_length])
    finger2_tip_local = np.array([0, -finger_separation/2, finger_length])
    
    # Transform finger positions to world coordinates
    finger1_base_world = gripper_base_world + end_effector_orientation @ finger1_base_local
    finger2_base_world = gripper_base_world + end_effector_orientation @ finger2_base_local
    finger1_tip_world = gripper_base_world + end_effector_orientation @ finger1_tip_local
    finger2_tip_world = gripper_base_world + end_effector_orientation @ finger2_tip_local
    
    # Set gripper color based on opening state
    gripper_color = [0.2, 0.8, 0.2] if gripper_opening > 0.1 else [0.8, 0.2, 0.2]
    
    # Log gripper base as a larger point
    rr.log(f"{robot_name}/gripper/base", 
           rr.Points3D(
               positions=[gripper_base_world],
               colors=[[0.3, 0.3, 0.3]],
               radii=[0.015]
           ))
    
    # Log finger base joints
    rr.log(f"{robot_name}/gripper/finger_joints", 
           rr.Points3D(
               positions=[finger1_base_world, finger2_base_world],
               colors=[gripper_color, gripper_color],
               radii=[0.008, 0.008]
           ))
    
    # Log finger tips
    rr.log(f"{robot_name}/gripper/finger_tips", 
           rr.Points3D(
               positions=[finger1_tip_world, finger2_tip_world],
               colors=[gripper_color, gripper_color],
               radii=[0.006, 0.006]
           ))
    
    # Log finger lines (from base to tip for each finger)
    finger_lines = []
    finger_line_colors = []
    
    # Finger 1
    finger_lines.extend([finger1_base_world, finger1_tip_world])
    finger_line_colors.extend([gripper_color, gripper_color])
    
    # Finger 2  
    finger_lines.extend([finger2_base_world, finger2_tip_world])
    finger_line_colors.extend([gripper_color, gripper_color])
    
    # Connection from gripper base to finger bases
    finger_lines.extend([gripper_base_world, finger1_base_world])
    finger_line_colors.extend([[0.5, 0.5, 0.5], gripper_color])
    
    finger_lines.extend([gripper_base_world, finger2_base_world])
    finger_line_colors.extend([[0.5, 0.5, 0.5], gripper_color])
    
    rr.log(f"{robot_name}/gripper/fingers", 
           rr.LineStrips3D(
               strips=[finger_lines[i:i+2] for i in range(0, len(finger_lines), 2)],
               colors=finger_line_colors
           ))
           
    # Log gripper state as text
    gripper_state = "OPEN" if gripper_opening > 0.1 else "CLOSED"
    rr.log(f"{robot_name}/gripper/state", 
           rr.TextDocument(f"Gripper: {gripper_state} (opening: {gripper_opening:.2f}, raw: {gripper_pos:.3f})"))
    
    # Log gripper opening as a scalar for plotting
    rr.log(f"{robot_name}/gripper/opening_value", rr.Scalars(gripper_opening))

def log_arm_camera_pose(camera_transform, robot_name="robot"):
    """Log the arm camera pose in the robot base frame."""
    if 'link6' not in camera_transform:
        return
    
    if robot_name == "right_arm":
        camera_name = "video_right"
    else:
        camera_name = "video_left"

    # Get end effector position and orientation
    ee_transform = camera_transform['link6']

    # Get transformation from end effector to camera
    T_ee_robotcamera = np.load(f'calibration/masking/robot_{camera_name}_hand/{camera_name}_ee_to_camera.npy')

    # Apply the transformation to get camera position in robot base frame
    camera_transform = ee_transform @ T_ee_robotcamera
    
    # Log the camera pose as a Transform3D
    rr.log(f"cameras/{camera_name}/pose", 
           rr.Transform3D(
               translation=camera_transform[:3, 3],
               mat3x3=camera_transform[:3, :3],
           ))
    
    # Log the pose in world space
    rr.log(f"cameras/{camera_name}/pose", rr.Pinhole(
        width=2,
        height=2,
        focal_length=2,))


def log_overhead_camera_pose(base_position, frame_idx=None, robot_name="robot"):
    """Log camera coordinate frames in the robot base frame using extrinsics, and log the overhead camera as a pinhole camera looking down."""
    # Load extrinsics based on robot arm
    if robot_name == "right_arm":
        head_to_base = np.load('calibration/masking/head-to-robot_video_right_hand/head_to_robot_base.npz')
    else:
        head_to_base = np.load('calibration/masking/head-to-robot_video_left_hand/head_to_robot_base.npz')

    overhead_intrinsics = np.load('calibration/masking/video_overhead_intri_imgs/video_overhead_intrinsics.npy', allow_pickle=True).item()

    # head_to_base: should contain a 4x4 matrix, e.g. 'transform' or similar
    head_to_base_mat = head_to_base['average_transform']
    
    # We take the inverse of head to base and times the base position to get the head position.
    head_position = np.linalg.inv(head_to_base_mat) @ base_position

    # Log the overhead camera as a pinhole camera looking down
    camera_translation = head_position[:3, 3]
    camera_rotation = head_position[:3, :3]
    # The camera looks down the -Z axis in its own frame; rerun expects mat3x3 to be the rotation from parent to camera
    # If needed, adjust orientation here

    # Get intrinsics
    camera_matrix = overhead_intrinsics['camera_matrix']
    resolution = (int(camera_matrix[0,2]*2) // 10, int(camera_matrix[1,2]*2)//10)
    focal_length = float(camera_matrix[0,0]//10)

    rr.log(f"cameras/overhead", rr.Transform3D(
        translation=camera_translation,
        mat3x3=camera_rotation,
        from_parent=True
    ))
    rr.log(f"cameras/overhead/pinhole", rr.Pinhole(
        width=2,
        height=2,
        focal_length=2,
    ))

    # Optionally, log the arm camera as before (commented out)
    # if 'link6' in transforms:
    #     ee_pose = transforms['link6']
    #     cam_pose = ee_pose @ arm_ee_to_camera
    #     rr.log(f"cameras/arm", rr.Transform3D(
    #         translation=cam_pose[:3, 3],
    #         mat3x3=cam_pose[:3, :3],
    #         from_parent=True
    #     ))

def load_dual_arm_calibration():
    """Load calibration data for dual arm setup and compute robot base transforms"""
    # Load head-to-robot transformations (robot base relative to overhead camera)
    left_head_to_robot = np.load('calibration/masking/head-to-robot_video_left_hand/head_to_robot_base.npz')
    right_head_to_robot = np.load('calibration/masking/head-to-robot_video_right_hand/head_to_robot_base.npz')
    
    # Get the average transforms (these are head-to-robot, i.e., robot base in camera frame)
    left_head_to_robot_mat = left_head_to_robot['average_transform']
    right_head_to_robot_mat = right_head_to_robot['average_transform']
    
    # print("Raw head-to-robot transforms:")
    # print(f"Left arm in camera frame: {left_head_to_robot_mat[:3, 3]}")
    # print(f"Right arm in camera frame: {right_head_to_robot_mat[:3, 3]}")
    
    # For visualization, we want robot bases at z=0 and camera above
    # The head-to-robot transforms give us the relative positions in camera frame
    # We'll use these to position the robots relative to each other, but put them on the ground
    # Put on ground
    left_arm_transform = left_head_to_robot_mat.copy()
    left_arm_transform[2, 3] = 0.0  # Put on ground
    # Flip to face upwards
    flip = R.from_euler('x', np.pi).as_matrix()
    left_arm_transform[:3, :3] = flip @ left_arm_transform[:3, :3]  # Invert rotation to face upwards

    right_arm_transform = right_head_to_robot_mat.copy()
    right_arm_transform[2, 3] = 0.0  # Put on ground
    right_arm_transform[:3, :3] = flip @ right_arm_transform[:3, :3]  # Invert rotation to face upwards
        
    return left_arm_transform, right_arm_transform

def visualize_episode_rerun(
    links,
    joints,
    actions,
    image_dict,
    is_dual_arm=False,
    aloha=False,
    velocities=None,
    commanded_actions=None,
    inference_query_indices=None,
    inference_query_timestamps=None,
    inference_query_overhead_frame_numbers=None,
    subgoal_switch=None,
    subgoal_switch_frames=None,
    failure=None,
    failure_frames=None,
):
    """Visualize episode data using rerun with robot model and camera feeds

    Args:
        links: Robot link information from URDF
        joints: Robot joint information from URDF
        actions: Joint position actions (qpos) - actual positions
        image_dict: Dictionary of camera images
        is_dual_arm: Whether this is dual arm mode
        aloha: Whether using ALOHA robot (affects image color conversion)
        velocities: Optional joint velocities (qvel) to log
        commanded_actions: Optional commanded actions from /action (for rollout data).
                          When provided, both actual (actions) and commanded positions
                          are displayed on the same graph for comparison.
        inference_query_indices: Optional frame indices at which policy inference queries were run.
        inference_query_timestamps: Optional query timestamps aligned with inference_query_indices.
        inference_query_overhead_frame_numbers: Optional overhead frame numbers aligned with queries.
        subgoal_switch: Optional per-frame subgoal switch mask annotation.
        subgoal_switch_frames: Optional explicit subgoal switch frame indices.
        failure: Optional per-frame failure mask annotation.
        failure_frames: Optional explicit failure frame indices.
    """
    # Get camera names and number of frames
    cam_names = list(image_dict.keys())
    num_camera_frames = len(image_dict[cam_names[0]]) if cam_names else 0
    num_action_frames = len(actions)
    
    # Use the maximum of camera frames and action frames to ensure we don't cut off
    num_frames = max(num_camera_frames, num_action_frames)

    query_idx_set = set()
    query_timestamp_by_idx = {}
    query_overhead_frame_by_idx = {}

    if inference_query_indices is not None:
        query_indices = np.asarray(inference_query_indices, dtype=np.int64).reshape(-1)
        query_idx_set = set(int(i) for i in query_indices if 0 <= int(i) < num_frames)

        if inference_query_timestamps is not None:
            query_timestamps = np.asarray(inference_query_timestamps, dtype=np.float64).reshape(-1)
            pair_len = min(len(query_indices), len(query_timestamps))
            for i in range(pair_len):
                frame_idx = int(query_indices[i])
                if 0 <= frame_idx < num_frames:
                    query_timestamp_by_idx[frame_idx] = float(query_timestamps[i])

        if inference_query_overhead_frame_numbers is not None:
            query_overhead = np.asarray(inference_query_overhead_frame_numbers, dtype=np.int64).reshape(-1)
            pair_len = min(len(query_indices), len(query_overhead))
            for i in range(pair_len):
                frame_idx = int(query_indices[i])
                if 0 <= frame_idx < num_frames:
                    query_overhead_frame_by_idx[frame_idx] = int(query_overhead[i])
    subgoal_switch_frame_set = build_failure_frame_set(subgoal_switch, subgoal_switch_frames, num_frames)
    failure_frame_set = build_failure_frame_set(failure, failure_frames, num_frames)
    
    print(f"Camera frames: {num_camera_frames}, Action frames: {num_action_frames}")
    if not cam_names:
        print("No camera frames available. Visualizing robot state only.")
    if query_idx_set:
        print(f"Found {len(query_idx_set)} inference query points for rerun visualization")
    if subgoal_switch_frame_set:
        print(f"Subgoal switch overlay enabled: {len(subgoal_switch_frame_set)} frame(s)")
    if failure_frame_set:
        print(f"Failure frame overlay enabled: {len(failure_frame_set)} frame(s)")
    print(f"Starting animation with {num_frames} frames...")
    
    # Load calibration transformations for dual arm setup
    left_arm_transform = None
    right_arm_transform = None

    # if is_dual_arm:
    #     left_arm_transform, right_arm_transform = load_dual_arm_calibration()

    # Log images and actions synchronously frame by frame
    for frame_idx in range(num_frames):
        rr.set_time("frame", sequence=frame_idx)
        is_query_frame = frame_idx in query_idx_set
        is_failure_frame = frame_idx in failure_frame_set
        is_subgoal_switch_frame = frame_idx in subgoal_switch_frame_set

        rr.log("inference/is_query_frame", rr.Scalars(1.0 if is_query_frame else 0.0))
        rr.log("annotations/is_failure_frame", rr.Scalars(1.0 if is_failure_frame else 0.0))
        rr.log("annotations/is_subgoal_switch_frame", rr.Scalars(1.0 if is_subgoal_switch_frame else 0.0))
        if frame_idx in query_timestamp_by_idx:
            rr.log("inference/query_timestamp", rr.Scalars(query_timestamp_by_idx[frame_idx]))
        if frame_idx in query_overhead_frame_by_idx:
            rr.log(
                "inference/query_overhead_frame_number",
                rr.Scalars(float(query_overhead_frame_by_idx[frame_idx]))
            )
        if frame_idx < len(actions):
            action = actions[frame_idx]
        elif len(actions) > 0:
            action = actions[-1]
        else:
            action = np.zeros(14 if is_dual_arm else 7, dtype=np.float32)
        
        # Log all camera images for this frame (if available)
        # Play camera images at half speed to align with actions
        camera_frame_idx = frame_idx  # Each camera frame shows for 2 action frames
        images = {}
        for cam_name in cam_names:
            if camera_frame_idx < len(image_dict[cam_name]):
                image = image_dict[cam_name][camera_frame_idx]
                if not aloha:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
                images[cam_name] = rgb_image


        for cam_name in cam_names:
            if cam_name not in images:
                continue
            camera_image = np.array(images[cam_name])
            if is_query_frame:
                camera_image = camera_image.copy()
                h, w = camera_image.shape[:2]
                cv2.rectangle(camera_image, (0, 0), (w - 1, h - 1), (0, 255, 255), 3)
                cv2.putText(
                    camera_image,
                    "QUERY",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            camera_image = depisode_data_camera_overlay(
                camera_image,
                frame_idx,
                is_subgoal_switch_frame=is_subgoal_switch_frame,
                is_failure_frame=is_failure_frame,
            )
            rr.log(f"cameras/{cam_name}", rr.Image(camera_image))
        
        # Log actions for this frame (if available)
        if frame_idx < len(actions):
            action = actions[frame_idx]
            
            if is_dual_arm:
                # Split action into left arm (first 7) and right arm (last 7)
                left_action = action[:7]
                right_action = action[7:14]

                # Left arm
                left_joint_positions = left_action[:6]
                left_gripper_position = left_action[6]

                # Right arm
                right_joint_positions = right_action[:6]
                right_gripper_position = right_action[6]

                # Log joint positions for both arms
                # When commanded_actions is available, log both actual and commanded on same graph
                if commanded_actions is not None and frame_idx < len(commanded_actions):
                    commanded = commanded_actions[frame_idx]
                    left_commanded = commanded[:7]
                    right_commanded = commanded[7:14]

                    # Left arm - actual and commanded
                    for i, joint_val in enumerate(left_joint_positions):
                        rr.log(f"actions/left_arm/joint_positions/{i}/actual", rr.Scalars(joint_val))
                        rr.log(f"actions/left_arm/joint_positions/{i}/commanded", rr.Scalars(left_commanded[i]))
                    rr.log("actions/left_arm/gripper_position/actual", rr.Scalars(left_gripper_position))
                    rr.log("actions/left_arm/gripper_position/commanded", rr.Scalars(left_commanded[6]))

                    # Right arm - actual and commanded
                    for i, joint_val in enumerate(right_joint_positions):
                        rr.log(f"actions/right_arm/joint_positions/{i}/actual", rr.Scalars(joint_val))
                        rr.log(f"actions/right_arm/joint_positions/{i}/commanded", rr.Scalars(right_commanded[i]))
                    rr.log("actions/right_arm/gripper_position/actual", rr.Scalars(right_gripper_position))
                    rr.log("actions/right_arm/gripper_position/commanded", rr.Scalars(right_commanded[6]))
                else:
                    # No commanded actions - log actual only
                    for i, joint_val in enumerate(left_joint_positions):
                        rr.log(f"actions/left_arm/joint_positions/{i}", rr.Scalars(joint_val))
                    rr.log("actions/left_arm/gripper_position", rr.Scalars(left_gripper_position))

                    for i, joint_val in enumerate(right_joint_positions):
                        rr.log(f"actions/right_arm/joint_positions/{i}", rr.Scalars(joint_val))
                    rr.log("actions/right_arm/gripper_position", rr.Scalars(right_gripper_position))

                # Log velocities for dual arm if available
                if velocities is not None and frame_idx < len(velocities):
                    velocity = velocities[frame_idx]
                    left_vel = velocity[:7]
                    right_vel = velocity[7:14]

                    for i, vel_val in enumerate(left_vel[:6]):
                        rr.log(f"velocities/left_arm/joint_{i}", rr.Scalars(vel_val))
                    rr.log("velocities/left_arm/gripper", rr.Scalars(left_vel[6]))

                    for i, vel_val in enumerate(right_vel[:6]):
                        rr.log(f"velocities/right_arm/joint_{i}", rr.Scalars(vel_val))
                    rr.log("velocities/right_arm/gripper", rr.Scalars(right_vel[6]))

                # Update robot models with calibrated positions
                # log_robot_model_with_transform(links, joints, left_joint_positions, left_gripper_position,
                #                              left_arm_transform, robot_name="left_arm")
                # log_robot_model_with_transform(links, joints, right_joint_positions, right_gripper_position,
                                            #  right_arm_transform, robot_name="right_arm")

            else:
                # Single arm mode - use identity transform (robot at origin)
                # When commanded_actions is available, log both actual and commanded on same graph
                if commanded_actions is not None and frame_idx < len(commanded_actions):
                    commanded = commanded_actions[frame_idx]

                    # Log actual and commanded positions on the same graph
                    for i, joint_val in enumerate(action[:6]):
                        rr.log(f"joint_positions/joint_{i}/actual", rr.Scalars(joint_val))
                        rr.log(f"joint_positions/joint_{i}/commanded", rr.Scalars(commanded[i]))
                    rr.log("gripper_position/actual", rr.Scalars(action[-1]))
                    rr.log("gripper_position/commanded", rr.Scalars(commanded[-1] if len(commanded) > 6 else commanded[6]))
                else:
                    # No commanded actions - log actual only
                    for i, joint_val in enumerate(action[:6]):
                        rr.log(f"joint_positions/joint_{i}", rr.Scalars(joint_val))
                    rr.log("gripper_position", rr.Scalars(action[-1]))

                # Log velocities for single arm if available
                if velocities is not None and frame_idx < len(velocities):
                    velocity = velocities[frame_idx]
                    for i, vel_val in enumerate(velocity[:6]):
                        rr.log(f"velocities/joint_{i}", rr.Scalars(vel_val))
                    rr.log("velocities/gripper", rr.Scalars(velocity[6] if len(velocity) > 6 else 0.0))

                # Update robot model with current joint positions
                joint_positions = action[:6]  # First 6 values are joint positions
                gripper_position = action[-1]  # Last value is gripper position
                log_robot_model(links, joints, joint_positions, gripper_position)

                # Debug output every 50 frames
                if frame_idx % 50 == 0:
                    print(f"Frame {frame_idx}: Joint positions = {joint_positions}, Gripper = {gripper_position:.3f}")
        else:
            # If we run out of actions, keep the robot in its last position
            if len(actions) > 0:
                last_action = actions[-1]
                
                if is_dual_arm:
                    left_action = last_action[:7]
                    right_action = last_action[7:14]
                    
                    left_joint_positions = left_action[:6]
                    left_gripper_position = left_action[6]
                    right_joint_positions = right_action[:6]
                    right_gripper_position = right_action[6]
                    
                    # log_robot_model_with_transform(links, joints, left_joint_positions, left_gripper_position, 
                    #                              left_arm_transform, robot_name="left_arm")
                    # log_robot_model_with_transform(links, joints, right_joint_positions, right_gripper_position, 
                                                #  right_arm_transform, robot_name="right_arm")
                else:
                    joint_positions = last_action[:6]
                    gripper_position = last_action[-1]
                    log_robot_model(links, joints, joint_positions, gripper_position)
                
                if frame_idx % 50 == 0:
                    print(f"Frame {frame_idx}: Using last action (no more actions available)")


def log_robot_model_with_transform(links, joints, joint_positions=None, gripper_pos=0.0, base_transform=None, robot_name="robot"):
    """Log robot model with a transformation matrix for positioning"""
    
    if joint_positions is None:
        joint_positions = [0.0] * 6
    if base_transform is None:
        base_transform = np.eye(4)
    
    # Compute forward kinematics
    transforms = compute_forward_kinematics(joints, joint_positions)
    
    # Apply base transformation to all transforms
    for link_name in transforms:
        transforms[link_name] = base_transform @ transforms[link_name]
    
    # Get joint positions for visualization
    joint_positions_3d = {}
    
    # Add base position
    joint_positions_3d['base'] = transforms['base_link'][:3, 3]
    
    # Get positions for each joint
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    for i, joint_name in enumerate(joint_names):
        joint_info = joints[joint_name]
        parent_link = joint_info['parent']
        child_link = joint_info['child']
            
        # Get the transform of the child link (which represents the joint position)
        joint_positions_3d[joint_name] = transforms[child_link][:3, 3]
    
    # Different colors for different joints
    joint_colors = {
        'base': [0.3, 0.3, 0.3],
        'joint1': [1.0, 0.0, 0.0],
        'joint2': [0.0, 1.0, 0.0], 
        'joint3': [0.0, 0.0, 1.0],
        'joint4': [1.0, 1.0, 0.0],
        'joint5': [1.0, 0.0, 1.0],
        'joint6': [0.0, 1.0, 1.0]
    }
    
    # Adjust colors for right arm to be distinguishable
    if robot_name == "right_arm":
        joint_colors = {
            'base': [0.5, 0.5, 0.5],
            'joint1': [0.8, 0.4, 0.4],
            'joint2': [0.4, 0.8, 0.4], 
            'joint3': [0.4, 0.4, 0.8],
            'joint4': [0.8, 0.8, 0.4],
            'joint5': [0.8, 0.4, 0.8],
            'joint6': [0.4, 0.8, 0.8]
        }
    
    # Log each joint as a ball
    for joint_name, position in joint_positions_3d.items():
        color = joint_colors.get(joint_name, [0.7, 0.7, 0.7])
        rr.log(f"{robot_name}/joints/{joint_name}", 
                rr.Points3D(
                    positions=[position],
                    colors=[color],
                    radii=[0.04]
                ))
    
    # Log sticks (line segments) connecting joints
    connections = [
        ('base', 'joint1'),
        ('joint1', 'joint2'),
        ('joint2', 'joint3'),
        ('joint3', 'joint4'),
        ('joint4', 'joint5'),
        ('joint5', 'joint6')
    ]
    
    # Collect all line segments for batch logging
    line_positions = []
    line_colors = []
    
    # Use different colors for different links
    if robot_name == "right_arm":
        link_colors = [
            [0.6, 0.3, 0.3],  # base to joint1
            [0.3, 0.6, 0.3],  # joint1 to joint2
            [0.3, 0.3, 0.6],  # joint2 to joint3
            [0.6, 0.6, 0.3],  # joint3 to joint4
            [0.6, 0.3, 0.6],  # joint4 to joint5
            [0.3, 0.6, 0.6]   # joint5 to joint6
        ]
    else:
        link_colors = [
            [0.8, 0.2, 0.2],  # base to joint1
            [0.2, 0.8, 0.2],  # joint1 to joint2
            [0.2, 0.2, 0.8],  # joint2 to joint3
            [0.8, 0.8, 0.2],  # joint3 to joint4
            [0.8, 0.2, 0.8],  # joint4 to joint5
            [0.2, 0.8, 0.8]   # joint5 to joint6
        ]
    
    for i, (start_joint, end_joint) in enumerate(connections):
        if start_joint in joint_positions_3d and end_joint in joint_positions_3d:
            start_pos = joint_positions_3d[start_joint]
            end_pos = joint_positions_3d[end_joint]
            
            # Add line segment
            line_positions.extend([start_pos, end_pos])
            color = link_colors[i] if i < len(link_colors) else [0.5, 0.5, 0.5]
            line_colors.extend([color, color])
    
    # Log all line segments at once
    if line_positions:
        rr.log(f"{robot_name}/links", 
                rr.LineStrips3D(
                    strips=[line_positions[i:i+2] for i in range(0, len(line_positions), 2)],
                    colors=line_colors
                ))
    
    # Add gripper visualization at the end effector (joint6)
    if 'joint6' in joint_positions_3d:
        log_gripper_visualization(transforms, gripper_pos, robot_name)


    log_overhead_camera_pose(base_position=base_transform, robot_name=robot_name)

    log_arm_camera_pose(transforms, robot_name=robot_name)
