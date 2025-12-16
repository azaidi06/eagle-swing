from scipy.signal import savgol_filter
import pandas as pd
import numpy as np


# Use one diagonal to normalize
def normalize_by_torso_diagonal(kps, l_sh_to_r_hip=True):
    if l_sh_to_r_hip: #left shoulder to right hip
        shoulder = kps[:, 5, :]
        hip = kps[:, 12, :]
    else: # right shoulder to left hip
        shoulder = kps[:, 6, :]
        hip = kps[:, 11, :]
    
    ## slide into both to ensure confidence column is not in play
    if shoulder.shape[-1] == 3:
        shoulder = shoulder[:, :2]
        hip = hip[:, :2]

    # Ensure we are only using x,y for distance calculation
    # (If these attributes include conf, slice them: shoulder[:, :2])
    torso_diagonal = np.sqrt(np.sum((shoulder - hip)**2, axis=1))
    
    # Handle broadcasting shape: (N, 1, 1)
    epsilon = 1e-6
    scale_factor = torso_diagonal[:, np.newaxis, np.newaxis] + epsilon
    
    xy = kps[..., :2]   # Shape: (N, K, 2)
    conf = kps[..., 2:] # Shape: (N, K, 1) (Keep dim for concatenation)
    
    # 3. Normalize only X,Y
    normalized_xy = xy / scale_factor
    
    # 4. Recombine
    normalized_kps = np.concatenate([normalized_xy, conf], axis=-1)
    
    return normalized_kps

# Use BOTH diagonals to normalize
def normalize_by_average_torso(kps):
    '''
    This normalization is more robust....
    As the torso rotates, one diagonal usually shortens while the other lengthens (or stays relatively stable).
    Averaging the two mitigates the "breathing" effect where the golfer appears to grow and 
    shrink during the swing. It provides a much more stable reference scale.
    '''
    left_shoulder = kps[:, 5, :]
    right_hip = kps[:, 12, :]
    right_shoulder = kps[:, 6, :]
    left_hip = kps[:, 11, :]

    if left_shoulder.shape[-1] == 3:
        left_shoulder = left_shoulder[:, :2]
        right_hip = right_hip[:, :2]
        right_shoulder = right_shoulder[:, :2]
        left_hip = left_hip[:, :2]
    
    diagonal1 = np.sqrt(np.sum((left_shoulder - right_hip)**2, axis=1))
    diagonal2 = np.sqrt(np.sum((right_shoulder - left_hip)**2, axis=1))
    avg_torso = (diagonal1 + diagonal2) / 2.0
    epsilon = 1e-6
    scale_factor = avg_torso[:, np.newaxis, np.newaxis] + epsilon

    xy = kps[..., :2]   # Shape: (N, 17, 2)
    conf = kps[..., 2:] # Shape: (N, 17, 1)
    normalized_xy = xy / scale_factor
    
    #recombine to preserve original keypoint structure
    normalized_kps = np.concatenate([normalized_xy, conf], axis=-1)
    
    return normalized_kps



def center_by_average_torso(kps):
    '''
    This normalization is more robust....
    As the torso rotates, one diagonal usually shortens while the other lengthens (or stays relatively stable).
    Averaging the two mitigates the "breathing" effect where the golfer appears to grow and 
    shrink during the swing. It provides a much more stable reference scale.
    '''
    left_shoulder = kps[:, 5, :]
    right_hip = kps[:, 12, :]
    right_shoulder = kps[:, 6, :]
    left_hip = kps[:, 11, :]

    if left_shoulder.shape[-1] == 3:
        left_shoulder = left_shoulder[:, :2]
        right_hip = right_hip[:, :2]
        right_shoulder = right_shoulder[:, :2]
        left_hip = left_hip[:, :2]
    
    diagonal1 = np.sqrt(np.sum((left_shoulder - right_hip)**2, axis=1))
    diagonal2 = np.sqrt(np.sum((right_shoulder - left_hip)**2, axis=1))
    avg_torso = (diagonal1 + diagonal2) / 2.0
    epsilon = 1e-6
    scale_factor = avg_torso[:, np.newaxis, np.newaxis] + epsilon

    # 2. Calculate Center (e.g., Midpoint of Hips)
    # Hips are usually indices 11 (left) and 12 (right) in COCO format
    hip_center = (kps[:, 11, :2] + kps[:, 12, :2]) / 2.0
    hip_center = hip_center[:, np.newaxis, :] # Reshape for broadcasting

    # 3. Center then Scale
    xy = kps[..., :2]
    centered_xy = xy - hip_center  # Moves golfer to (0,0)
    normalized_xy = centered_xy / scale_factor # Scales golfer to unit size

    # Recombine
    conf = kps[..., 2:]
    return np.concatenate([normalized_xy, conf], axis=-1)


def align_vertical(kps):
    """
    Rotates the keypoints so the line from Mid-Ankles to Mid-Hips is vertical.
    """
    # 1. Define the "Body Vector" we want to make vertical
    # Using Mid-Ankle to Mid-Hip is usually stable
    left_hip = kps[:, 11, :2]
    right_hip = kps[:, 12, :2]
    left_ankle = kps[:, 15, :2]
    right_ankle = kps[:, 16, :2]
    
    # Midpoints
    mid_hip = (left_hip + right_hip) / 2.0
    mid_ankle = (left_ankle + right_ankle) / 2.0
    
    # Vector from Ankle UP to Hip
    body_vector = mid_hip - mid_ankle # Shape (N, 2)
    
    # 2. Calculate current angle of body vector
    # arctan2(y, x) gives angle from positive x-axis
    angles = np.arctan2(body_vector[:, 1], body_vector[:, 0])
    
    # 3. We want this vector to point UP (which is -90 degrees or -pi/2 in image coords)
    # Or DOWN (+90 degrees or +pi/2) depending on your coord system.
    # In standard image coords (y-down), Up is negative Y.
    target_angle = -np.pi / 2 
    
    # The rotation needed is: Target - Current
    rotation_angles = target_angle - angles
    
    # 4. Create Rotation Matrix for each frame
    # R = [[cos, -sin], [sin, cos]]
    c = np.cos(rotation_angles)
    s = np.sin(rotation_angles)
    
    # 5. Apply Rotation
    # We need to rotate around the center (which is (0,0) because you already centered hips)
    # Formula: x' = x*c - y*s,  y' = x*s + y*c
    
    xy = kps[..., :2]
    x = xy[..., 0]
    y = xy[..., 1]
    
    x_new = x * c[:, np.newaxis] - y * s[:, np.newaxis]
    y_new = x * s[:, np.newaxis] + y * c[:, np.newaxis]
    
    rotated_xy = np.stack([x_new, y_new], axis=-1)
    
    return np.concatenate([rotated_xy, kps[..., 2:]], axis=-1)


def rescale_for_visualization(norm_kps, canvas_size=512, zoom_factor=200):
    """
    Converts normalized/centered keypoints back to pixel coordinates for viewing.
    
    Args:
        norm_kps: (N, 17, 3) array from your normalize_and_center function
        canvas_size: Size of the square window you want to watch (e.g., 512x512)
        zoom_factor: Pixels per torso-length. Larger = closer zoom.
    """
    # 1. Copy data to avoid messing up your actual ML data
    vis_kps = norm_kps.copy()
    
    # 2. Extract just the X and Y coordinates (preserve confidence scores)
    xy = vis_kps[..., :2]
    
    # 3. Scale (Zoom In)
    # This turns "1.0 torso length" into "200 pixels"
    xy = xy * zoom_factor
    
    # 4. Translate (Move Center to Middle of Screen)
    # This moves (0,0) to (256, 256) so negative values become visible
    center_offset = np.array([canvas_size / 2, canvas_size / 2])
    xy = xy + center_offset
    
    # 5. Update the array and return
    vis_kps[..., :2] = xy
    
    return vis_kps


"""
This is the "Holy Grail" for swing comparison:

Translation Invariance: Centers the golfer at (0,0) (removes where they are standing in the frame).

Scale Invariance: Makes the golfer 1.0 units tall (removes how far the camera is).

Rotation Invariance: Rotates the golfer to stand straight up (removes camera tilt).

⚠️ CRITICAL WARNING: The "Erased Swing" Problem

You must be very careful with Rotation.

If you calculate the rotation angle for EVERY frame: You will force the shoulders to be horizontal
    in every single frame. This effectively deletes the swing. When the golfer tilts their shoulders 
    during the backswing, your code will rotate the whole image to make them flat again.

The Solution: Calculate the rotation angle based on Frame 0 (Address) only, and apply that 
    single rotation matrix to the entire video. This corrects the camera angle without destroying 
    the swing dynamics.

    
    *** DO NOT HAVE A CONSISTENT First frame yet
"""



def align_to_body_frame_static(kps):
    """
    Centers the golfer and fixes camera tilt based on the FIRST FRAME.
    Preserves swing dynamics (doesn't flatten shoulders in every frame).
    """
    # 1. Separate XY from Confidence
    xy = kps[..., :2]   # (N, 17, 2)
    conf = kps[..., 2:] # (N, 17, 1)

    # --- TRANSLATION (Centering) ---
    # Calculate mid-hip for every frame
    left_hip = xy[:, 11, :]
    right_hip = xy[:, 12, :]
    mid_hip = (left_hip + right_hip) / 2.0
    
    # Subtract mid_hip from all keypoints (Broadcasting)
    # This centers the golfer's hips at (0,0) for every frame
    xy_centered = xy - mid_hip[:, np.newaxis, :]

    # --- ROTATION (Camera Tilt Correction) ---
    # We use FRAME 0 ONLY to determine the 'upright' angle
    l_sh_0 = xy[0, 5, :]
    r_sh_0 = xy[0, 6, :]
    
    # Vector from Right Shoulder to Left Shoulder (Frame 0)
    shoulder_vec = l_sh_0 - r_sh_0
    
    # Calculate angle of shoulders relative to horizontal
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    
    # We want to rotate by negative angle to make it flat (0 degrees)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    
    # Create Rotation Matrix (2, 2)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # Apply rotation to all frames
    # (N, 17, 2) dot (2, 2) -> (N, 17, 2)
    xy_rotated = xy_centered @ rot_matrix.T
    
    # --- RECOMBINE ---
    return np.concatenate([xy_rotated, conf], axis=-1)




def align_vertical_rotation(kps):
    '''
    Rotates keypoints so the "gravity line" (Mid-Ankles to Mid-Hips) is perfectly vertical.
    This fixes camera tilt issues and standardizes the golfer's setup angle.
    
    kps shape: (Frames, 17, 3) or (Frames, 17, 2)
    '''
    # 1. Define the reference vector: Mid-Ankle -> Mid-Hip
    # COCO indices: 11=L.Hip, 12=R.Hip, 15=L.Ankle, 16=R.Ankle
    left_hip = kps[:, 11, :2]
    right_hip = kps[:, 12, :2]
    left_ankle = kps[:, 15, :2]
    right_ankle = kps[:, 16, :2]
    
    mid_hip = (left_hip + right_hip) / 2.0
    mid_ankle = (left_ankle + right_ankle) / 2.0
    
    # Vector pointing UP from ankles to hips
    body_vector = mid_hip - mid_ankle 
    
    # 2. Calculate the angle of this vector relative to vertical
    # In image space, "Up" is negative Y (angle = -90 degrees or -pi/2)
    current_angles = np.arctan2(body_vector[:, 1], body_vector[:, 0])
    target_angle = -np.pi / 2 
    
    # Calculate how much we need to rotate to hit the target
    rotation_needed = target_angle - current_angles # Shape: (N,)

    # 3. Create Rotation Matrix for each frame
    c = np.cos(rotation_needed)
    s = np.sin(rotation_needed)
    
    # 4. Apply Rotation
    # NOTE: We rotate around (0,0). 
    # Ensure this is called AFTER you have centered the hips at (0,0).
    xy = kps[..., :2]
    x = xy[..., 0]
    y = xy[..., 1]
    
    # Rotation Formula:
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    x_new = x * c[:, np.newaxis] - y * s[:, np.newaxis]
    y_new = x * s[:, np.newaxis] + y * c[:, np.newaxis]
    
    rotated_xy = np.stack([x_new, y_new], axis=-1)
    
    # Recombine with confidence scores
    if kps.shape[-1] == 3:
        return np.concatenate([rotated_xy, kps[..., 2:]], axis=-1)
    return rotated_xy





def normalize_static_vertical(kps):
    """
    Calculates rotation angle ONLY from the first frame (setup) and applies
    it to the entire video. This fixes camera tilt without distorting the swing dynamics.
    """
    # 1. Get Setup Frame (Frame 0)
    setup_frame = kps[0] 
    
    # 2. Calculate the "Gravity Vector" from the Setup Frame ONLY
    # L.Hip=11, R.Hip=12, L.Ankle=15, R.Ankle=16
    left_hip = setup_frame[11, :2]
    right_hip = setup_frame[12, :2]
    left_ankle = setup_frame[15, :2]
    right_ankle = setup_frame[16, :2]
    
    mid_hip = (left_hip + right_hip) / 2.0
    mid_ankle = (left_ankle + right_ankle) / 2.0
    
    # Vector pointing UP from ankles to hips in the SETUP frame
    setup_vector = mid_hip - mid_ankle 
    
    # 3. Calculate the Static Angle Correction
    current_angle = np.arctan2(setup_vector[1], setup_vector[0])
    target_angle = -np.pi / 2  # -90 degrees (Vertical Up in image coords)
    rotation_needed = target_angle - current_angle # Scalar value, not array
    
    # 4. Apply this ONE rotation to ALL frames
    c = np.cos(rotation_needed)
    s = np.sin(rotation_needed)
    
    # Rotation Matrix (2x2)
    R = np.array([[c, -s], 
                  [s, c]])
    
    # 5. Rotate all Keypoints
    # Assuming kps is already centered at (0,0) by your previous function
    xy = kps[..., :2] # Shape (Frames, 17, 2)
    
    # Einsum is cleaner for matrix multiplication across frames/joints
    # Multiplies every (x,y) pair by the rotation matrix R
    rotated_xy = np.einsum('ij,tfj->tfi', R, xy)
    
    # Recombine with confidence
    if kps.shape[-1] == 3:
        return np.concatenate([rotated_xy, kps[..., 2:]], axis=-1)
    return rotated_xy


def normalize_with_procrustes_only(kps, template):
    """
    Replaces all previous normalization. 
    Scales, Rotates, and Translates 'kps' to match 'template' in one go.
    """
    # 1. Setup: Get the first frame of the video to calculate the transform
    # We only calculate T (transform) once per video to preserve swing dynamics
    source_shape = kps[0, :, :2] 
    
    # 2. Calculate Centroids
    # We align based on stable points (Shoulders, Hips, Knees)
    # COCO Indices: 5,6 (Shoulders), 11,12 (Hips), 13,14 (Knees)
    align_indices = [5, 6, 11, 12, 13, 14]
    
    mu_source = np.mean(source_shape[align_indices], axis=0)
    mu_template = np.mean(template[align_indices], axis=0)
    
    # 3. Center the shapes
    source_centered = source_shape - mu_source
    template_centered = template - mu_template
    
    # 4. Compute Scale (Frobenius Norm of the centered stable points)
    # This replaces your "diagonal torso" logic
    source_scale = np.linalg.norm(source_centered[align_indices])
    template_scale = np.linalg.norm(template_centered[align_indices])
    scale_factor = template_scale / source_scale
    
    # 5. Compute Rotation (Kabsch)
    # Scale the source first for the rotation calculation
    source_norm = source_centered * scale_factor
    
    H = np.dot(source_norm[align_indices].T, template_centered[align_indices])
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 6. Apply Transform to ALL frames (Rotation + Scale + Translation)
    # (x - mu) * scale * R + mu_template
    
    all_xy = kps[..., :2]
    all_centered = all_xy - mu_source
    
    # Apply Scale
    all_scaled = all_centered * scale_factor
    
    # Apply Rotation (einsum for efficient broadcasting)
    all_rotated = np.einsum('ij,tfj->tfi', R, all_scaled)
    
    # Move to template location
    final_xy = all_rotated + mu_template
    
    # Recombine
    return np.concatenate([final_xy, kps[..., 2:]], axis=-1)