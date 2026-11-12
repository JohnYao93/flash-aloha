
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))
_SCRIPT_ROOT = _ACT_ROOT.parent / "scripts"
if str(_SCRIPT_ROOT) not in _path_sys.path:
    _path_sys.path.append(str(_SCRIPT_ROOT))
# get the camera intrinsic matrix

import cv2
import numpy as np
import os
import glob
import argparse

calib_path = './calibration/masking/'  # the masking calibration set (utils.masking)

def capture_checkerboard_images(camera_name, num_images=20):
    """
    Capture images of a checkerboard pattern using webcam and save them.
    
    Args:
        num_images (int): Number of images to capture
    """
    path = os.path.join(calib_path, camera_name + '_intri_imgs')
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Initialize webcam
    cap = cv2.VideoCapture(f'/dev/{camera_name}')
    
    img_count = 0
    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow('Checkerboard Capture', frame)
        
        # Wait for key press
        key = cv2.waitKey(1)
        
        # Press 's' to save image, 'q' to quit
        if key == ord('s'):
            img_name = f"{path}/checkerboard_{img_count}.png"
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            img_count += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def calibrate_camera(camera_name, checkerboard_size=(14,9), square_size=0.02, resolution=(320,180)):
    """
    Calibrate camera using saved charuco board images and save intrinsic parameters.
    
    Args:
        checkerboard_size (tuple): Internal corners of the checkerboard (width, height)
        square_size (float): Size of each square on the checkerboard in meters
    """
    # Arrays to store object points and image points from all images
    all_object_points = []
    all_image_points = []
    image_size = None

    path = os.path.join(calib_path, camera_name + '_intri_imgs')
    
    # Get list of images
    images = glob.glob(os.path.join(path, '*.png'))

    board = cv2.aruco.CharucoBoard((checkerboard_size[0], checkerboard_size[1]), square_size, 0.015, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250))
    charuco_detector = cv2.aruco.CharucoDetector(board)

    if not images:
        raise ValueError("No images found in {} directory".format(path))
    
    for fname in images:
        img = cv2.imread(fname)
        cv2.imshow('Image', img)
        img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
        
        # Detect charuco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        # If charuco corners are detected, store them
        if charuco_corners is not None and len(charuco_corners) > 3:
            # Get the 3D object points for detected ChArUco corners
            obj_points = board.getChessboardCorners()
            
            # Extract only the object points corresponding to detected corner IDs
            detected_obj_points = np.array([obj_points[id[0]] for id in charuco_ids], dtype=np.float32)
            
            all_object_points.append(detected_obj_points)
            all_image_points.append(charuco_corners)
            
            print(f"Image {fname}: Detected {len(charuco_corners)} charuco corners")
            
            # Draw detected markers and corners
            img_display = img.copy()
            if marker_corners is not None:
                cv2.aruco.drawDetectedMarkers(img_display, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(img_display, charuco_corners, charuco_ids)
            cv2.imshow('Corners', img_display)
            cv2.waitKey(500)
        else:
            corners_found = len(charuco_corners) if charuco_corners is not None else 0
            print(f"Image {fname}: Only detected {corners_found} charuco corners (need > 3)")
    
    cv2.destroyAllWindows()
    
    # Check if we have enough data for calibration
    print(f"Total images processed: {len(images)}")
    print(f"Images with valid charuco detections: {len(all_image_points)}")
    
    if len(all_image_points) < 4:
        raise ValueError(f"Not enough valid charuco detections for calibration. Found {len(all_image_points)}, need at least 4 images with detected corners.")
    
    # Calibrate camera using standard cv2.calibrateCamera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None
    )
    
    # Save the camera matrix and distortion coefficients
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist
    }
    np.save(os.path.join(path, camera_name+'_intrinsics.npy'), calibration_data)
    print("Camera calibration completed and saved to camera_intrinsics.npy")
    print(f"{calibration_data=}")
    
    return mtx, dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument('--camera_name', type=str, default="video_overhead", help="Name of the camera")
    parser.add_argument('--width', type=int, default=320, help="Width to resize images for calibration")
    parser.add_argument('--height', type=int, default=180, help="Height to resize images for calibration")
    args = parser.parse_args()
    camera_name = args.camera_name
    width = args.width
    height = args.height
    # capture_checkerboard_images(camera_name)
    calibrate_camera(camera_name, resolution=(width, height))