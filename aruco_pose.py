## ArUco Pose Detection + Multi-Model Classification Analysis (on caps list)
import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import pickle

# Load camera calibration
cameraMatrix, dist = pickle.load(open(r"calibration.pkl", "rb"))
print(f"Camera Matrix:\n{cameraMatrix}")
print(f"Distortion Coefficients: {dist.flatten()}")

# ArUco setup
aruco_dict_type = cv2.aruco.DICT_4X4_50
aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 50
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Marker size in meters (adjust to your actual marker size)
MARKER_SIZE = 0.06  # 5cm marker
TARGET_ARUCO_ID = 10

def estimate_camera_angles(rvec, tvec):
    """
    Estimate camera horizontal (yaw) and vertical (pitch) angles relative to the marker plane.
    When camera is directly facing the marker (head-on), both angles are 0 degrees.
    Uses rotation-based approach via find_angle decomposition.
    
    Horizontal angle: positive = camera is to the right of marker center
    Vertical angle: positive = camera is above marker center
    """
    R, _ = cv2.Rodrigues(rvec)
    # Marker normal in camera coordinates
    marker_normal_camera = R @ np.array([0, 0, 1])
    
    # Decompose into horizontal (yaw) and vertical (pitch) angles
    # When marker_normal_camera = [0, 0, 1] (facing camera), angles are 0
    nx, ny, nz = marker_normal_camera
    
    # Horizontal angle: deviation in XZ plane
    horizontal_angle = np.degrees(np.arctan2(nx, nz))
    
    # Vertical angle: deviation in YZ plane  
    vertical_angle = np.degrees(np.arctan2(-ny, nz))
    
    return horizontal_angle, vertical_angle

def get_distance_to_marker(tvec):
    """Calculate distance from camera to marker center in meters."""
    return np.linalg.norm(tvec)

def get_camera_angles_from_frame(frame, target_aruco_id=TARGET_ARUCO_ID, marker_size=MARKER_SIZE):
    """
    Given a frame, detect ArUco marker and return camera angles.
    
    Args:
        frame: BGR image (numpy array)
        target_aruco_id: ID of the ArUco marker to detect (default: 10)
        marker_size: Physical size of the marker in meters (default: 0.05m)
    
    Returns:
        dict with keys:
            - 'found': bool - whether the marker was found
            - 'horizontal_angle_deg': float - horizontal angle in degrees (if found)
            - 'vertical_angle_deg': float - vertical angle in degrees (if found)
            - 'distance_m': float - distance to marker in meters (if found)
            - 'tvec': numpy array - translation vector (if found)
            - 'rvec': numpy array - rotation vector (if found)
        Returns {'found': False} if marker not detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or target_aruco_id not in ids.flatten():
        return {'found': False}
    
    # Find index of target marker
    marker_idx = np.where(ids.flatten() == target_aruco_id)[0][0]
    marker_corners = corners[marker_idx]
    
    # Estimate pose
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(
        obj_points, 
        marker_corners.reshape(-1, 2),
        cameraMatrix, 
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if not success:
        return {'found': False}
    
    # Calculate angles
    angle = find_angle(rvec)
    distance = get_distance_to_marker(tvec)
    
    return {
        'found': True,
        'angle': angle,
        'distance_m': distance,
        'tvec': tvec,
        'rvec': rvec,
        'corners': marker_corners
    }


def find_angle(rvec):
    R, _ = cv2.Rodrigues(rvec)
    marker_normal_camera = R @ np.array([0, 0, 1])
    camera_direction = np.array([0, 0, -1])

    # Normalize vectors (just to be safe)
    marker_normal_camera = marker_normal_camera / np.linalg.norm(marker_normal_camera)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # Compute the angle
    dot_product = np.dot(marker_normal_camera, camera_direction)
    angle_rad = np.arccos(dot_product)

    # Convert to degrees if you like
    angle_deg = np.degrees(angle_rad)

    return angle_deg
