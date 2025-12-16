from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import pickle
#import cv2



def find_address_frame_velocity(kps, fps=60, stillness_thresh=2.0, lookback_buffer=5):
    """
    Finds the frame where the swing starts (Address).
    
    Strategy:
    1. Calculate velocity of the wrists (hands move first).
    2. Find the point of significant movement (takeaway).
    3. Walk backwards until velocity is near zero.
    """
    # 1. Extract Wrists (Left: 9, Right: 10)
    # Shape: (Frames, 2) - averaging L and R wrist for a stable 'hands' point
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    
    # 2. Calculate Velocity (Displacement between frames)
    # Simple Euclidean distance between frame t and t-1
    velocity = np.sqrt(np.sum(np.diff(hands, axis=0)**2, axis=1))
    
    # Smooth the velocity to remove jitter (optional but recommended)
    # A simple moving average
    window = 5
    velocity_smooth = np.convolve(velocity, np.ones(window)/window, mode='same')
    
    # 3. Define "Movement" (The Takeaway)
    # Find the first time velocity exceeds a "Movement Threshold" (e.g., 5x stillness)
    # We scan from the beginning. 
    movement_start_idx = -1
    movement_thresh = stillness_thresh * 3.0
    
    # Skip first 10 frames to avoid video artifact noise at start
    for i in range(0, len(velocity_smooth)):
        if velocity_smooth[i] > movement_thresh:
            # Ensure it's not a blip: check if the NEXT 5 frames are also moving
            if np.mean(velocity_smooth[i:i+5]) > movement_thresh:
                movement_start_idx = i
                break
    
    if movement_start_idx == -1:
        print("Warning: No swing movement detected.")
        return 0 # Default to start
    
    # 4. Walk Backwards to find "Stillness"
    # Go back from movement_start_idx until velocity drops below stillness_thresh
    address_idx = movement_start_idx
    for i in range(movement_start_idx, 0, -1):
        if velocity_smooth[i] < stillness_thresh:
            address_idx = i
            break
            
    # 5. Apply Buffer
    # Go back a few more frames to get the "settled" pose
    final_idx = max(0, address_idx - lookback_buffer)
    
    return final_idx


def find_address_frame_accel(kps, window_size=5, sensitivity=1.5):
    """
    Finds the exact frame the hands break stillness using rolling variance.
    
    kps: (Frames, 17, 3)
    window_size: How many frames to look at for 'stillness'
    sensitivity: Multiplier for the standard deviation threshold
    """
    # 1. Track Hands (Average of L/R Wrist)
    # Shape: (Frames, 2)
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    
    # 2. Calculate Euclidean Velocity (Speed)
    # diff[i] = hands[i+1] - hands[i]
    # Pad with 0 at start to keep length consistent
    deltas = np.diff(hands, axis=0, prepend=hands[0:1])
    speed = np.sqrt(np.sum(deltas**2, axis=1))
    
    # 3. Find the 'Takeaway' (Major movement)
    # We look for the first time speed exceeds the median speed of the whole clip
    # (This gets us roughly to the start of the swing, usually a bit late)
    global_median_speed = np.median(speed)
    swing_event_idx = np.argmax(speed > (global_median_speed * 4)) # 4x median is a safe 'moving' guess
    
    if swing_event_idx == 0:
        # Fallback if no major movement found
        return 0

    # 4. Fine-Grained Backtracking (The Fix)
    # We walk backwards from the "Swing Event" and check the VARIANCE of the hands
    # in a small window. When variance drops to near-zero, we are at address.
    
    start_idx = 0
    
    # Walk backwards from the big movement
    for i in range(swing_event_idx, window_size, -1):
        # Look at a small window of velocity BEFORE this frame
        window = speed[i-window_size:i]
        
        # Calculate noise floor of this window
        local_noise = np.std(window)
        local_avg = np.mean(window)
        
        # CRITERIA:
        # If the average speed in this previous window is very low
        # AND the standard deviation is tiny (stable), we found the static phase.
        if local_avg < (global_median_speed * sensitivity) and local_noise < 0.5:
            # This index 'i' is the first frame where it started speeding up
            # after being still.
            start_idx = i
            break
            
    # 5. Safety Buffer
    # Subtract 2-3 frames just to be safe (catch the trigger)
    return max(0, start_idx - 3)


def find_address_robust(kps, fps=60, lookahead_frames=10):
    """
    Scale-Invariant Address Finder.
    Uses 'Body Height' to normalize thresholds, making it work for 
    both pixel-space and normalized (0-1) keypoints.
    """
    # 1. Calculate Reference Scale (Torso Length or Body Height)
    # Nose: 0, Left Ankle: 15, Right Ankle: 16 (COCO format)
    # If ankles are missing/occluded, use Hips (11, 12) to Nose
    # We use the median position over the first 30 frames to get a stable scale
    
    # Shape: (Frames, 17, 2)
    valid_frames = min(30, len(kps))
    nose = kps[:valid_frames, 0, :2]
    l_ankle = kps[:valid_frames, 15, :2]
    r_ankle = kps[:valid_frames, 16, :2]
    
    # Fallback if ankles are zero (missing)
    if np.mean(l_ankle) < 1e-3: 
        l_hip = kps[:valid_frames, 11, :2]
        r_hip = kps[:valid_frames, 12, :2]
        lower_body = (l_hip + r_hip) / 2.0
    else:
        lower_body = (l_ankle + r_ankle) / 2.0
        
    # Calculate Body Height (Euclidean distance)
    ref_height = np.mean(np.linalg.norm(nose - lower_body, axis=1))
    
    if ref_height == 0: ref_height = 1.0 # Avoid div/0
    
    # 2. Extract Hands (Mean of L/R Wrist) & Smooth
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    window_len = max(7, int(fps * 0.1) | 1)
    hands_smooth = savgol_filter(hands, window_length=window_len, polyorder=3, axis=0)
    
    # 3. Compute Velocity (normalized by Body Height)
    # Units: "Body Heights per Frame"
    raw_disp = np.linalg.norm(np.diff(hands_smooth, axis=0, prepend=hands_smooth[0:1]), axis=1)
    velocity_norm = raw_disp / ref_height 
    
    # 4. Define Thresholds relative to Body Size
    # Waggle usually moves ~0.01 to 0.02 body heights
    # Takeaway moves > 0.05 body heights quickly
    
    # Noise Floor: 10th percentile of movement (stillness)
    noise_floor = np.percentile(velocity_norm, 10)
    
    # Trigger: 3x noise or at least 0.5% of body height per frame
    trigger_thresh = max(noise_floor * 3.0, 0.005) 
    
    candidates = np.where(velocity_norm > trigger_thresh)[0]
    
    true_takeaway_idx = -1
    
    for idx in candidates:
        if idx < 10: continue # Ignore artifacts at start
        if idx + lookahead_frames >= len(hands_smooth): continue
            
        # Lookahead: "Straightness" Check
        start_pt = hands_smooth[idx]
        end_pt = hands_smooth[idx + lookahead_frames]
        
        net_disp = np.linalg.norm(end_pt - start_pt) / ref_height
        
        # Calculate Path Length over window
        segment = hands_smooth[idx : idx + lookahead_frames + 1]
        segment_steps = np.linalg.norm(np.diff(segment, axis=0), axis=1)
        path_len = np.sum(segment_steps) / ref_height
        
        if path_len < 1e-5: continue
            
        ratio = net_disp / path_len
        
        # CRITERIA (Scale Invariant):
        # 1. Ratio > 0.8 (Motion is efficient/straight, not waggle)
        # 2. Net Displacement > 2% of Body Height (Significant move)
        if ratio > 0.80 and net_disp > 0.02: 
            true_takeaway_idx = idx
            break
    
    if true_takeaway_idx == -1:
        return 0 # Default

    # 5. Backtrack to "Quiet" Point
    # Search back 0.5s for the local minimum in velocity
    back_buffer = int(fps * 0.5)
    search_start = max(0, true_takeaway_idx - back_buffer)
    search_end = true_takeaway_idx
    
    # Find index of minimum velocity in this window
    local_min = np.argmin(velocity_norm[search_start:search_end+1])
    
    return search_start + local_min 


def find_address_optimized(kps, fps=60, lookback_seconds=1.2):
    """
    Robustly finds the Address frame by locating the point of minimum 
    hand velocity preceding the main swing event.
    """
    # 1. Extract Hands (Average of L/R Wrist: indices 9, 10)
    # Shape: (Frames, 2)
    hands = (kps[:, 9, :2] + kps[:, 10, :2]) / 2.0
    
    # 2. Calculate Velocity
    # Use simple diff first to preserve sharp edges
    deltas = np.diff(hands, axis=0, prepend=hands[0:1])
    # Euclidean speed
    speed = np.sqrt(np.sum(deltas**2, axis=1))
    
    # 3. Smart Smoothing
    # We want to smooth noise but keep the "cliff" of the takeaway sharp.
    # A small window median filter or SavGol is better than simple average.
    window_len = max(5, int(fps * 0.05) | 1)  # ~50ms window
    if len(speed) > window_len:
        speed_smooth = savgol_filter(speed, window_len, 2)
    else:
        speed_smooth = speed

    # 4. Find the "Main Event" (Trigger)
    # Instead of a fixed threshold, use a percentage of the Peak Velocity.
    # This adapts to slow swings vs fast swings.
    peak_velocity = np.percentile(speed_smooth, 98) # 98th percentile to ignore outliers
    trigger_thresh = peak_velocity * 0.40 # Trigger at 40% of max speed
    
    # Find frames where speed > trigger
    candidates = np.where(speed_smooth > trigger_thresh)[0]
    
    if len(candidates) == 0:
        print("Warning: No significant movement detected.")
        return 0

    # Take the first candidate that is part of a sustained movement
    # (Simple heuristic: just take the first crossing)
    trigger_idx = candidates[0]
    
    # 5. Look Back for the "Floor"
    # We define a window BEFORE the trigger to find the stillness.
    lookback_frames = int(fps * lookback_seconds)
    search_start = max(0, trigger_idx - lookback_frames)
    search_end = trigger_idx
    
    # Isolate the window of interest
    window_speed = speed_smooth[search_start:search_end+1]
    
    # CRITICAL FIX: Find the index of MINIMUM velocity in this window.
    # We don't stop at a threshold; we go to the absolute quietest point.
    # We bias towards the END of the window (closer to trigger) if there are ties
    # to avoid picking a random static point 1 second ago.
    
    # Add a small penalty for being too far from the trigger? 
    # Actually, purely min velocity is usually best for "Address".
    min_local_idx = np.argmin(window_speed)
    
    address_frame = search_start + min_local_idx
    
    # 6. Sanity Check / Fine Tuning
    # If the address frame velocity is still high (e.g. tight trim video),
    # it means the video started MID-swing.
    if speed_smooth[address_frame] > (peak_velocity * 0.1):
        # Fallback: The start of the video is the best guess
        return 0
        
    return address_frame



def calculate_angle(a, b, c):
    """Calculates the angle at point B in degrees."""
    # Note: Assumes a, b, c are NumPy arrays
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_geometric_score(frame_kps):
    """
    Calculates a 'badness' score for a single frame. Lower is better.
    Uses Left arm geometry as an example (adjust for golfer's handedness).
    """
    # Keypoint indices for COCO: L-Shoulder (5), L-Elbow (7), L-Wrist (9)
    shoulder_l = frame_kps[5, :2]
    elbow_l = frame_kps[7, :2]
    wrist_l = frame_kps[9, :2]

    # Score 1: Arm Straightness (Deviation from 180 degrees)
    arm_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
    # A perfectly straight arm is 180 degrees.
    score1 = abs(180.0 - arm_angle)

    # Score 2: Vertical Arm Hang (Deviation of shoulder-wrist from vertical)
    # A vertical line has a very small x-component change.
    vec = wrist_l - shoulder_l
    if np.linalg.norm(vec) < 1e-6:
        score2 = 0
    else:
        # Penalize horizontal deviation
        score2 = abs(vec[0]) / np.linalg.norm(vec)
        
    # Return a weighted combination of scores
    return score1 + (score2 * 50) # Weight verticality higher

def find_address_geometric(kps, velocity_trigger_idx, fps=60, lookback_seconds=1.5):
    """
    Scans backwards from a velocity trigger to find the frame with the
    best geometric posture for the Address.
    
    Args:
        kps: Keypoints for the entire video (Frames, 17, 3).
        velocity_trigger_idx: The frame index identified by the previous velocity method.
    """
    search_start = max(0, int(velocity_trigger_idx - (fps * lookback_seconds)))
    search_end = velocity_trigger_idx
    
    if search_start >= search_end:
        return search_start

    scores = []
    for i in range(search_start, search_end):
        score = get_geometric_score(kps[i])
        scores.append(score)
        
    if not scores:
        return search_start
        
    # Find the index corresponding to the minimum score in the lookback window
    best_local_idx = np.argmin(scores)
    
    return search_start + best_local_idx