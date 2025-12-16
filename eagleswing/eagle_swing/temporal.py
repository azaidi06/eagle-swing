import numpy as np
from scipy.signal import savgol_filter

def get_signed_distance_from_line(p_wrist, p_shoulder, p_hip):
    """
    Returns > 0 if wrist is to the RIGHT of the shoulder-hip line.
    Returns < 0 if wrist is to the LEFT of the shoulder-hip line.
    
    Handles vertical body lines correctly.
    """
    x_w, y_w = p_wrist
    x_s, y_s = p_shoulder
    x_h, y_h = p_hip
    
    # Avoid division by zero if body is perfectly vertical
    if abs(y_h - y_s) < 1e-6:
        return x_w - x_s

    # Equation of line: x = x_s + (y - y_s) * slope_inv
    # We want x at the wrist's y-level (y_w)
    slope_inv = (x_h - x_s) / (y_h - y_s)
    x_line_at_wrist_y = x_s + (y_w - y_s) * slope_inv
    
    return x_w - x_line_at_wrist_y

def find_last_frame_wrist_is_right(keypoints, impact_frame=None):
    """
    Finds the last frame where the Right Wrist is clearly to the Right of the Body Line.
    
    Args:
        keypoints: (N, K, 2) array
        impact_frame: Optional integer. If known, we only search before this. 
                      If None, we search the whole swing.
    
    Returns:
        int: Frame index.
    """
    # COCO Keypoints
    R_WRIST = 10
    R_SHOULDER = 6
    R_HIP = 12
    
    # Default to searching from the end if impact is unknown
    start_search = impact_frame if impact_frame else len(keypoints) - 1
    
    # Backtrack from Impact -> Start
    for t in range(start_search, 0, -1):
        kp = keypoints[t][...,:2]
        
        # Calculate distance
        dist = get_signed_distance_from_line(kp[R_WRIST], kp[R_SHOULDER], kp[R_HIP])
        
        # The moment we find a positive distance (Wrist is Right), we stop.
        # Because we are walking backwards, this is the "Last" frame it was right.
        if dist > 0:
            return t
            
    return None


def find_last_frame_before_downswing(keypoints):
    """
    Identifies the last static frame before the hands accelerate down.
    
    Strategy:
    1. Detect the main Downswing event (Maximum Vertical Velocity).
    2. Walk backwards from that moment until velocity drops to near-zero.
    
    Args:
        keypoints: Numpy array of shape (num_frames, num_kps, 2)
        
    Returns:
        int: The frame index of the Top of Backswing.
    """
    # COCO Indices
    L_HAND, R_HAND = 9, 10 
    
    # 1. Extract Raw Y-Coordinates (Average of both hands for stability)
    # In images, Y increases downwards. 
    # Small Y = High Hands. Large Y = Low Hands.
    raw_ys = (keypoints[:, L_HAND, 1] + keypoints[:, R_HAND, 1]) / 2
    
    # 2. Smooth the Signal (Crucial for removing jitter)
    # Uses Savitzky-Golay filter (window=7, polyorder=2) if available, else simple moving average
    try:
        ys = savgol_filter(raw_ys, window_length=7, polyorder=2)
    except:
        ys = np.convolve(raw_ys, np.ones(5)/5, mode='same')

    # 3. Calculate Vertical Velocity
    # Positive Velocity = Moving Down (Y increasing)
    velocity = np.diff(ys) 
    
    # 4. Find the "Heart" of the Downswing
    # The moment of maximum downward acceleration/speed
    # We only look for max velocity in the middle 80% of frames to avoid start/end artifacts
    search_margin = len(velocity) // 10
    if len(velocity) < 20: search_margin = 0
    
    valid_vel = velocity[search_margin : len(velocity)-search_margin]
    if len(valid_vel) == 0: return None
    
    # Index of max velocity relative to the 'valid_vel' slice
    local_max_idx = np.argmax(valid_vel)
    # Convert back to global frame index
    max_downswing_frame = local_max_idx + search_margin

    # 5. Backtrack: Walk left until velocity is Zero or Negative
    # We look for the transition from "Moving Down" to "Static/Moving Up"
    top_frame = max_downswing_frame
    
    # Threshold: velocity must be consistently positive to be "downswing"
    # We stop when velocity drops below a small noise floor (e.g., 0.5 pixel/frame)
    noise_floor = 0.5 
    
    for t in range(max_downswing_frame, 0, -1):
        vel = velocity[t]
        
        # If velocity is negative (moving up) or near zero (static)
        if vel <= noise_floor:
            top_frame = t
            break
            
    return top_frame


def get_signed_distance(p_hand, p_shoulder, p_hip):
    """
    Returns > 0 if hand is to the RIGHT of the shoulder-hip line.
    Returns < 0 if hand is to the LEFT of the shoulder-hip line.
    Uses a vertical boundary when the hand is above the shoulder.
    """
    x_h, y_h = p_hand[:2]
    x_s, y_s = p_shoulder[:2]
    x_hip, y_hip = p_hip[:2]
    
    # 1. Vertical Projection Fix:
    # If hand is above the shoulder (y_h < y_s in image coords), 
    # use the shoulder's X as the vertical boundary.
    if y_h < y_s:
        return x_h - x_s

    # 2. Standard Line Projection (Below Shoulder):
    # Avoid division by zero
    if abs(y_hip - y_s) < 1e-6: 
        return x_h - x_s 
        
    # Calculate x of the body line at the hand's y-coordinate
    slope_inv = (x_hip - x_s) / (y_hip - y_s)
    x_line_at_hand_y = x_s + (y_h - y_s) * slope_inv
    
    return x_h - x_line_at_hand_y


def find_crossing_frames(keypoints):
    """
    Identifies frames where hands cross body lines in a specific direction (Right -> Left).
    
    Args:
        keypoints: Numpy array of shape (num_frames, num_kps, 2)
    
    Returns:
        dict: Lists of frame indices where the crossing event completes.
    """
    # COCO Indices
    L_SHOULDER, L_HIP = 5, 11
    R_SHOULDER, R_HIP = 6, 12
    L_HAND, R_HAND = 9, 10  # Using Wrists as proxies for Hands

    events = {
        'right_hand_crosses_left_body': [], # Right Hand crossing Left Body Line
        'left_hand_crosses_right_body': []  # Left Hand crossing Right Body Line
    } 
    holder = []
    # Iterate through frames
    for t in range(len(keypoints) - 1):
        kp_curr = keypoints[t]
        kp_next = keypoints[t+1]

        # --- Check 1: Right Hand crossing Left Body Line (Right -> Left) ---
        dist_curr = get_signed_distance(kp_curr[R_HAND], kp_curr[L_SHOULDER], kp_curr[L_HIP])
        dist_next = get_signed_distance(kp_next[R_HAND], kp_next[L_SHOULDER], kp_next[L_HIP])
        
        # Check for transition from Positive (Right) to Negative (Left)
        if dist_curr > 0 and dist_next <= 0:
            events['right_hand_crosses_left_body'].append(t + 1)
            holder.append(t)

        # --- Check 2: Left Hand crossing Right Body Line (Right -> Left) ---
        dist_curr_l = get_signed_distance(kp_curr[L_HAND], kp_curr[R_SHOULDER], kp_curr[R_HIP])
        dist_next_l = get_signed_distance(kp_next[L_HAND], kp_next[R_SHOULDER], kp_next[R_HIP])
        
        # Check for transition from Positive (Right) to Negative (Left)
        if dist_curr_l > 0 and dist_next_l <= 0:
            events['left_hand_crosses_right_body'].append(t + 1)
            
        # # --- Check 2: Left Hand crossing Right Body Line (Right -> Left) ---
        # dist_curr_r = get_signed_distance(kp_curr[R_HAND], kp_curr[R_SHOULDER], kp_curr[R_HIP])
        # dist_next_r = get_signed_distance(kp_next[R_HAND], kp_next[R_SHOULDER], kp_next[R_HIP])
        
        # # Check for transition from Positive (Right) to Negative (Left)
        # if dist_curr_r > 0 and dist_next_r <= 0:
        #     events['left_hand_crosses_right_body'].append(t + 1)
        #     holder.append(t)

    events['time_between_crossing'] = holder[1] - holder[0]

    return events