from typing import TypeVar, Generic, List, Optional, Callable, Union
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from functools import partial
import pandas as pd
import numpy as np
import pickle
from .normalization import *
from .swing_events import *

'''
Provides the functionality to load swing keypoint data + metadata
Everything is in a class function so that the code is more organized,
    and modular
Will make the overaching class (KPExtractor) parameterized so that
    specific actions can be done on the data depenidng on what we're
    trying to do in that specific moment
'''

class SwingMetaData:
    def __init__(
        self,
        pkl_path: str,
        get_swing_idx: bool = False,
        get_swing_score: bool = False,
        swing_score: Optional[Union[int, str]] = None, 
        clip_name: Optional[str] = None,
    ):
        self.pkl_path = pkl_path
        self.video_path = self.get_video_path()
        self.str_path = self.pkl_path
        if clip_name:
            self.clip_name = clip_name
            self.swing_score = int(self.clip_name.split('_')[-1])
            self.swing_idx = int(self.clip_name.split('_')[-3])
        if get_swing_idx:
            self.swing_idx = self.get_swing_idx_or_score(swing_idx=True)
        if get_swing_score:
            self.swing_score = self.get_swing_idx_or_score(score_idx=True)
        elif swing_score is not None: 
            self.swing_score = int(swing_score)

    def get_video_path(self):
        # Just making sure the path being used is a str + not path object
        str_pkl = str(self.pkl_path)
        return f'{str_pkl[:-3]}mp4'
    
    def get_swing_idx_or_score(self, swing_idx=False, score_idx=False):
        swing_path_split_names = self.str_path.split('/')[-1].split('_')
        for idx, x in enumerate(swing_path_split_names):
            if swing_idx and x == 'swing':
                return swing_path_split_names[idx + 1]    
            elif score_idx and x == 'score':
                return swing_path_split_names[idx + 1].split('.')[0]
    
    

class SwingKeypointData(SwingMetaData):
    def __init__(self, pkl_path, **kwargs):
        super().__init__(pkl_path=pkl_path, **kwargs)
        self.kp_dicts = self.get_kp_dicts()
        self.key_points = self.get_keypoints()
        self.confidence_scores = self.get_confidence_scores()
        self.raw_kps = np.concatenate([self.key_points, 
                                   np.expand_dims(self.confidence_scores, -1)], axis=2)

    def get_kp_dicts(self):
        # Get the frame by frame output dicts from pose estimation models
        with open(self.str_path, 'rb') as f:
            loaded_dicts = pickle.load(f)
        return loaded_dicts
        
    def get_keypoints(self):
        kp_dicts = self.kp_dicts
        return np.stack([self.kp_dicts[key]['keypoints'] for key in kp_dicts.keys()])
        
    def get_confidence_scores(self):
        kp_dicts = self.kp_dicts
        return np.stack([self.kp_dicts[key]['keypoint_scores'] for key in kp_dicts.keys()])

    def get_frame(self, idx):
        ''' Just grabs a single frames keypoints and scores
        '''
        kps = self.key_points[idx]
        confidence_scores = self.confidence_scores[idx]
        return np.column_stack((kps, confidence_scores))

    def get_frames(self, indexes):
        '''
        num_idxs = len(Indexes)
        Takes a list of indexes and returns an array wof [num_idxs, 17, 3]
        [1] 17 keypoint markers
        [2] 3 keypoint values/certaintainty (X, Y, Score)
        '''
        return np.stack([self.get_frame(idx) for idx in indexes])

    def __len__(self):
        return len(self.key_points)



class KpExtractor(SwingKeypointData):
    def __init__(self, 
                 file_name,
                 threshold_value=None,
                 post_processors: List[Callable] = None,
                 start_idx: int = 0,
                 end_idx: Optional[int] = None,
                 **kwargs):
                 
        super().__init__(pkl_path=file_name, **kwargs)
        
        self.kps = self.threshold_score(threshold_value)
        if end_idx:
            self.kps = self.kps[start_idx:end_idx]

        if post_processors:
            for processor in post_processors:
                self.kps = processor(self.kps)

        self.coco_idxs = {"L_SH":5, "R_SH":6, "L_ELBOW":7, "R_ELBOW":8, 
             "L_WRIST":9, "R_WRIST":10, "L_HIP":11, "R_HIP":12, 
             "L_KNEE":13, "R_KNEE":14, "L_ANKLE":15, "R_ANKLE":16}
        
        # # Dynamically create attributes using setattr
        # for attr_name, coco_key in self.coco_idxs.items():
        #     setattr(self, attr_name.lower(), 
        #             self.kps[:, coco_key,
        #             :].astype(float).copy())
            
    # # Handles the case-insensitive lookup
    # def __getattr__(self, name):
    #     """
    #     Called when default attribute access fails. 
    #     Try to find the lowercase version of the attribute.
    #     """
    #     lower_name = name.lower()
    #     if lower_name in self.__dict__:
    #         return self.__dict__[lower_name]
    def __getattr__(self, name):
        lower_name = name.lower()
        upper_name = name.upper()
        
        # Check if the requested attribute is one of our body parts
        if hasattr(self, 'coco_idxs') and upper_name in self.coco_idxs:
            coco_key = self.coco_idxs[upper_name]
            # Dynamically slice the CURRENT kps array
            return self.kps[:, coco_key, :].astype(float)
            
        # Fallback for other attributes
        if lower_name in self.__dict__:
            return self.__dict__[lower_name]
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def threshold_score(self, threshold_value=0.4):
        if threshold_value is None: 
            return self.raw_kps.astype(float).copy()
        # punch up score values to a threshold
        kps = self.raw_kps.astype(float).copy()
        mask = kps[:, :, 2] < threshold_value
        kps[mask, 2] = threshold_value
        return kps
    

def interpolate_and_filter_pandas(keypoints):
    """
    keypoints: shape (N, 2) or (N, C) numpy array with NaNs
    """
    # 1. Capture original shape
    n_frames, n_keypoints, n_coords = keypoints.shape
    just_kps = keypoints[:, :, :2]  # Ignore confidence scores for interpolation/filtering
    
    # 2. Reshape to 2D: (Frames, Keypoints * XY)
    # This flattens (17, 2) into 34 columns. Each column is a specific coordinate trajectory.
    # shape becomes (180, 34)
    flattened_data = just_kps.reshape(n_frames, -1)
    
    # 3. Interpolate with Pandas
    df = pd.DataFrame(flattened_data)
    # 'linear' connects points with a straight line
    # 'limit_direction="both"' handles missing frames at the start/end of clip
    df_interp = df.interpolate(method='linear', limit_direction='both')
    
    # 3. Convert back to numpy
    clean_data = df_interp.to_numpy()
    
    # 4. Apply Savitzky-Golay filter
    # window_length must be odd; polyorder is typically 2 or 3
    smoothed_data = savgol_filter(clean_data, window_length=7, polyorder=3, axis=0)
    reshaped_data = smoothed_data.reshape(n_frames, n_keypoints, 2)
    confidence_scores = keypoints[:, :, 2:]
    reconstructed_data = np.concatenate([reshaped_data, confidence_scores], axis=2)
    return reconstructed_data


class SwingExtractor(KpExtractor):
    def __init__(self, 
                 row,
                 normalizer=True,
                 processors: List[Callable] = None,
                 post_processors: List[Callable] = None,
                 **kwargs):
        super().__init__(file_name=row.pkl_path, 
                         start_idx=row.start_idx,
                         end_idx=row.end_idx,
                         post_processors=processors,
                         clip_name=row.clip_name,
                         **kwargs)
        self.row = row
        self.clip_name = self.row.clip_name
        self.processors = processors
        self.kps = self.get_clip_kps(self.row, self.processors)
        if normalizer:
            self.kps.kps = self.normalize_kps_by_body(self.kps)
                     
        self.right_arm_angle = self.get_arm_angle(right=True)
        self.left_arm_angle = self.get_arm_angle(right=False)
        self.shoulder_angle = self.get_shoulder_angle(right=True)
        self.hip_angle = self.get_hip_angle(right=True)
        self.right_leg_angle = self.get_leg_angle(right=True)
        self.left_leg_angle = self.get_leg_angle(right=False)   
        self.x_factor = self.get_x_factor()
        self.x_torque = self.get_x_torque()
                     
        self.right_side_bend = self.get_side_bend(right=True)
        self.left_side_bend = self.get_side_bend(right=False)
                     
        self.vertical_extension = self.get_vertical_extension()
        # self.swing_radius = self.get_swing_radius()

        self.compute_all_derivatives()

        if post_processors:
            for processor in post_processors:
                self.kps = processor(self.kps)
    
    def get_clip_kps(self, row, processors=None):
        pkl_file, clip_name = row.pkl_path, row.clip_name
        start_idx, end_idx = row.start_idx, row.end_idx
        kpe = KpExtractor(pkl_file,
                           post_processors=processors,
                           start_idx = start_idx,
                           end_idx = end_idx
                          )
        return kpe
        
    def get_arm_angle(self, right=True):
        if right:
            first_pt = self.kps.r_sh
            second_pt = self.kps.r_elbow
            third_pt = self.kps.r_wrist
        else:
            first_pt = self.kps.l_sh
            second_pt = self.kps.l_elbow
            third_pt = self.kps.l_wrist
        angle = angle_3points_deg(first_pt, second_pt, third_pt)
        return np.abs(angle)
        
    
    def get_shoulder_angle(self, right=True):
        if right:
            first_pt = self.kps.r_sh
            second_pt = self.kps.l_sh
        else:
            first_pt = self.kps.l_sh
            second_pt = self.kps.r_sh
        return angle_2points_deg(first_pt, second_pt)
    
                 
    def get_hip_angle(self, right=True):
        if right:
            first_pt = self.kps.r_hip
            second_pt = self.kps.l_hip
        else:
            first_pt = self.kps.l_hip
            second_pt = self.kps.r_hip
        return angle_2points_deg(first_pt, second_pt)
    
                 
    def get_leg_angle(self, right=True):
        if right:
            first_pt = self.kps.r_hip
            second_pt = self.kps.r_knee
            third_pt = self.kps.r_ankle
        else:
            first_pt = self.kps.l_hip
            second_pt = self.kps.l_knee
            third_pt = self.kps.l_ankle
        return angle_3points_deg(first_pt, second_pt, third_pt)
    
    
    # def get_x_factor(self):
    #     """
    #     Calculates X-Factor (Shoulder Rotation - Hip Rotation).
    #     Note: This is an estimation based on 2D projected width.
    #     """
    #     # 1. Get Rotation Angles (0 deg = Square to camera, 90 deg = Perpendicular)
    #     shoulder_rot = self._calculate_rotation_from_width(self.kps.r_sh, self.kps.l_sh)
    #     hip_rot = self._calculate_rotation_from_width(self.kps.r_hip, self.kps.l_hip)
        
    #     # 2. Calculate X-Factor (The difference in rotation)
    #     # We use absolute difference because direction (backswing vs downswing) 
    #     # is ambiguous in simple width projection without temporal tracking.
    #     x_factor = np.abs(shoulder_rot - hip_rot)
        
    #     return x_factor
    # def get_x_factor(self):
    # # 1. Robust Max Width (Use 95th percentile to ignore outliers)
    #     s_width = np.linalg.norm(self.kps.r_sh - self.kps.l_sh, axis=1)
    #     h_width = np.linalg.norm(self.kps.r_hip - self.kps.l_hip, axis=1)
        
    #     # Smooth the widths FIRST to prevent jitter at the peaks
    #     # Window size must be odd; adjust based on fps (e.g., 15 for 60fps)
    #     s_width_smooth = savgol_filter(s_width, window_length=15, polyorder=2)
    #     h_width_smooth = savgol_filter(h_width, window_length=15, polyorder=2)

    #     # Calculate robust max (baseline)
    #     s_max = np.percentile(s_width_smooth, 98) 
    #     h_max = np.percentile(h_width_smooth, 98)

    #     # 2. Calculate Rotation (Face-On Logic: arccos)
    #     # Ratio = 1.0 at Address (0 deg), Ratio < 1.0 when turned
    #     s_ratio = np.clip(s_width_smooth / s_max, -1.0, 1.0)
    #     h_ratio = np.clip(h_width_smooth / h_max, -1.0, 1.0)

    #     s_rot = np.degrees(np.arccos(s_ratio))
    #     h_rot = np.degrees(np.arccos(h_ratio))

    #     # 3. Calculate X-Factor
    #     x_factor = np.abs(s_rot - h_rot)
        
    #     return x_factor
    def get_x_factor(self):
        # 1. Calculate Widths
        s_width_smooth = np.linalg.norm(self.kps.r_sh - self.kps.l_sh, axis=1)
        h_width_smooth = np.linalg.norm(self.kps.r_hip - self.kps.l_hip, axis=1)

        # 2. Define Max Width (Baseline for 90-degree turn)
        # CRITICAL: "Max" in DTL is the Top of Swing (widest visual point).
        # We assume the golfer turns at least near 90 degrees.
        s_max = np.percentile(s_width_smooth, 98) 
        h_max = np.percentile(h_width_smooth, 98)

        # 3. Calculate Rotation (DTL Logic: arcsin)
        s_ratio = np.clip(s_width_smooth / s_max, -1.0, 1.0)
        h_ratio = np.clip(h_width_smooth / h_max, -1.0, 1.0)

        # Use arcsin instead of arccos
        s_rot = np.degrees(np.arcsin(s_ratio))
        h_rot = np.degrees(np.arcsin(h_ratio))

        # 4. Clean up "Address" Offset
        # At address in DTL, you still have some width (profile view). 
        # This calculates a non-zero angle (e.g., 20 degrees) at address.
        # You can optionally zero this out by subtracting the min value.
        s_rot = s_rot - np.min(s_rot)
        h_rot = h_rot - np.min(h_rot)

        # 5. Calculate X-Factor
        x_factor = np.abs(s_rot - h_rot)
        
        return x_factor




    def get_x_torque(self):
        """
        Calculates 'X-Factor Stretch' (The rate of change of X-Factor).
        High derivative = rapid separation of hips/shoulders (good power indicator).
        """
        if not hasattr(self, 'x_factor'):
            self.x_factor = self.get_x_factor()
            
        # Gradient calculates the slope (rate of change) over the frames
        return np.gradient(self.x_factor)

    def _calculate_rotation_from_width(self, kp_right, kp_left):
      # Calculate Euclidean distance
        widths = np.linalg.norm(kp_right - kp_left, axis=1)
        max_width = np.max(widths)
        
        if max_width == 0: return np.zeros_like(widths)
        
        # DTL Logic: Width increases as you rotate
        # 0 deg (Address) = Min Width -> sin(0) = 0
        # 90 deg (Top) = Max Width -> sin(90) = 1
        rotation_ratio = np.clip(widths / max_width, -1.0, 1.0)
        
        # Use arcsin instead of arccos
        rotation_angles = np.degrees(np.arcsin(rotation_ratio))
        
        return rotation_angles

    def get_side_bend(self, right=True):
        first_pt, second_pt = (self.kps.r_sh, self.kps.r_hip) if right else (self.kps.l_sh, self.kps.l_hip)
        diff = first_pt - second_pt
        return np.linalg.norm(diff, axis=1)

    def get_vertical_extension(self):
        l_thigh = np.linalg.norm(self.kps.l_hip[0] - self.kps.l_knee[0])
        l_shin = np.linalg.norm(self.kps.l_knee[0] - self.kps.l_ankle[0])
        l_leg_len = l_thigh + l_shin
        r_thigh = np.linalg.norm(self.kps.r_hip[0] - self.kps.r_knee[0])
        r_shin = np.linalg.norm(self.kps.r_knee[0] - self.kps.r_ankle[0])
        r_leg_len = r_thigh + r_shin
        reference_length = (l_leg_len + r_leg_len) / 2
        
        l_hip, r_hip = self.kps.l_hip[:, 1], self.kps.r_hip[:, 1]
        l_ankle, r_ankle = self.kps.l_ankle[:, 1], self.kps.r_ankle[:, 1]
    
        mid_hip_y = (l_hip + r_hip) / 2
        mid_ankle_y = (l_ankle + r_ankle) / 2
        vertical_extension = mid_ankle_y - mid_hip_y
        normalized_extension = vertical_extension / reference_length
        return normalized_extension


    def normalize_kps_by_body(self,
                              kps, 
                              method='torso_height',
                             ):
        """
        Calculates robust normalization factor (scale) AND centers the subject.
        """
        # Extract coordinates (Frames, 2)
        l_hip, r_hip = kps.l_hip[:, :2], kps.r_hip[:, :2]
        l_knee, r_knee = kps.l_knee[:, :2], kps.r_knee[:, :2]
        l_ankle, r_ankle = kps.l_ankle[:, :2], kps.r_ankle[:, :2]
        l_shldr, r_shldr = kps.l_sh[:, :2], kps.r_sh[:, :2]
        
        # 1. Calculate the Root (Mid-Hip) for centering
        mid_hip = (l_hip + r_hip) / 2  # Shape: (Frames, 2)

        # Helper: Euclidean distance
        def dist(p1, p2):
            return np.linalg.norm(p1 - p2, axis=1)
    
        if method == 'leg_segment_sum':
            len_left = dist(l_hip, l_knee) + dist(l_knee, l_ankle)
            len_right = dist(r_hip, r_knee) + dist(r_knee, r_ankle)
            frame_scales = np.maximum(len_left, len_right)
            
        elif method == 'torso_height':
            mid_shldr = (l_shldr + r_shldr) / 2
            frame_scales = dist(mid_shldr, mid_hip)
    
        # Filter noise
        valid_scales = frame_scales[frame_scales > 10]
        if len(valid_scales) == 0:
            return kps.kps # Return original if failure
    
        global_scale = np.median(valid_scales)
        
        # 2. Apply Translation (Centering)
        # Create a copy to avoid modifying the original if needed
        kps_centered = kps.kps.copy() 

        # Reshape mid_hip for broadcasting: (Frames, 2) -> (Frames, 1, 2)
        mid_hip_reshaped = mid_hip[:, None, :]

        # Subtract mid_hip ONLY from the x, y coordinates (indices :2)
        # Shape (30, 17, 2) - Shape (30, 1, 2) = Shape (30, 17, 2) -- Valid Broadcasting
        kps_centered[:, :, :2] = kps.kps[:, :, :2] - mid_hip_reshaped

        # 3. Apply Scaling
        # Note: You likely want to scale the x,y coordinates, but NOT the confidence score.
        # If you divide the whole array, confidence scores (usually 0-1) will become tiny.
        normalized_kps = kps_centered.copy()
        normalized_kps[:, :, :2] = kps_centered[:, :, :2] / global_scale

        return normalized_kps
    
    # def normalize_kps_by_body(self,
    #                           kps, 
    #                           method='torso_height',
    #                          ):
    #     """
    #     Calculates a robust normalization factor (scale) for a single video.
    #     method: 'leg_segment_sum' or 'torso_height'
            
    #     Returns:
    #         scale_factor: Float representing the pixel length of the body part.
    #     """
    #     # Extract coordinates (Frames, 2)
    #     l_hip, r_hip = kps.l_hip[:, :2], kps.r_hip[:, :2]
    #     l_knee, r_knee = kps.l_knee[:, :2], kps.r_knee[:, :2]
    #     l_ankle, r_ankle = kps.l_ankle[:, :2], kps.r_ankle[:, :2]
        
    #     # Helper: Euclidean distance between two point arrays
    #     def dist(p1, p2):
    #         return np.linalg.norm(p1 - p2, axis=1)
    
    #     if method == 'leg_segment_sum':
    #         # Calculate lengths for BOTH legs to be safe (average them)q
    #         # Segment lengths: Hip->Knee + Knee->Ankle
    #         len_left = dist(l_hip, l_knee) + dist(l_knee, l_ankle)
    #         len_right = dist(r_hip, r_knee) + dist(r_knee, r_ankle)
            
    #         # Combine: You can take the max (usually the leg facing camera) 
    #         # or average (if facing front). For golf (side view), max is safer 
    #         # as the back leg might be occluded/foreshorterned.
    #         frame_scales = np.maximum(len_left, len_right)
            
    #     elif method == 'torso_height':
    #         # Mid-Hip to Mid-Shoulder
    #         l_shldr, r_shldr = kps.l_sh[:, :2], kps.r_sh[:, :2]
    #         mid_hip = (l_hip + r_hip) / 2
    #         mid_shldr = (l_shldr + r_shldr) / 2
    #         frame_scales = dist(mid_shldr, mid_hip)
    
    #     # ROBUST STATISTIC: Use Median or Trimmed Mean across frames
    #     # This ignores frames where detection failed (length ~ 0 or huge)
    #     # Filter out zeros first
    #     valid_scales = frame_scales[frame_scales > 10] # Threshold for noise
        
    #     if len(valid_scales) == 0:
    #         return 1.0 # Fallback
    
    #     global_scale = np.median(valid_scales)
    #     normalized_kps = kps.kps/global_scale
    #     return normalized_kps
    
    def add_derivatives(self, metric_name, window=7, poly=3):
        """
        Calculates velocity (d1) and acceleration (d2) for a given metric
        and attaches them to the instance as new attributes.
        Example: 'x_factor' -> 'x_factor_vel' and 'x_factor_acc'
        """
        if not hasattr(self, metric_name):
            return

        data = getattr(self, metric_name)
        
        # Calculate Velocity (1st Derivative)
        # delta=1 assumes frame-by-frame. If you have time, use delta=dt
        vel = savgol_filter(data, window_length=window, polyorder=poly, deriv=1, delta=1)
        
        # Calculate Acceleration (2nd Derivative)
        acc = savgol_filter(data, window_length=window, polyorder=poly, deriv=2, delta=1)

        # Set new attributes dynamically
        setattr(self, f"{metric_name}_vel", vel)
        setattr(self, f"{metric_name}_acc", acc)

    def compute_all_derivatives(self):
        # Define which metrics need derivatives
        target_metrics = [
            "x_factor", 
            "right_arm_angle", 
            "vertical_extension",
            "shoulder_angle", # Rate of tilt
            "hip_angle" 
        ]
        
        for metric in target_metrics:
            self.add_derivatives(metric)
    


def get_interpolator(window_length=7):
    return partial(interpolate_and_filter_pandas, window_length=window_length)    


def interpolate_and_filter_pandas(keypoints, 
                                  window_length=7,
                                  polyorder=3,
                                 ):
    """
    keypoints: shape (N, 2) or (N, C) numpy array with NaNs
    """
    # 1. Capture original shape
    n_frames, n_keypoints, n_coords = keypoints.shape
    just_kps = keypoints[:, :, :2]  # Ignore confidence scores for interpolation/filtering
    
    # 2. Reshape to 2D: (Frames, Keypoints * XY)
    # This flattens (17, 2) into 34 columns. Each column is a specific coordinate trajectory.
    # shape becomes (180, 34)
    flattened_data = just_kps.reshape(n_frames, -1)
    
    # 3. Interpolate with Pandas
    df = pd.DataFrame(flattened_data)
    # 'linear' connects points with a straight line
    # 'limit_direction="both"' handles missing frames at the start/end of clip
    df_interp = df.interpolate(method='spline', order=2, limit_direction='both')
    df_interp = df.interpolate(method='linear', limit_direction='both')

    # 3. Convert back to numpy
    clean_data = df_interp.to_numpy()
    
    # 4. Apply Savitzky-Golay filter
    # window_length must be odd; polyorder is typically 2 or 3
    smoothed_data = savgol_filter(clean_data, 
                                  window_length=window_length, 
                                  polyorder=polyorder, 
                                  axis=0)
    reshaped_data = smoothed_data.reshape(n_frames, n_keypoints, 2)
    confidence_scores = keypoints[:, :, 2:]
    reconstructed_data = np.concatenate([reshaped_data, confidence_scores], axis=2)
    return reconstructed_data


def angle_2points_deg(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    # Compute arctan2
    angles = np.arctan2(dy, dx)
    # Convert to degrees in-place to save memory allocation
    np.degrees(angles, out=angles)
    return angles



# def angle_3points_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
#     """
#     Returns signed angle (degrees) at vertex b, from vector ba -> bc.
#     Range: [-180, 180]
#     Positive = Clockwise (if y-axis is down/image coords)
#     """
#     # Create vectors relative to B
#     ba = a - b
#     bc = c - b
    
#     # Calculate determinant (2D cross product) and dot product
#     # det = x1*y2 - y1*x2
#     det = ba[:, 0] * bc[:, 1] - ba[:, 1] * bc[:, 0]
    
#     # dot = x1*x2 + y1*y2
#     dot = ba[:, 0] * bc[:, 0] + ba[:, 1] * bc[:, 1]
    
#     # arctan2(y, x) -> arctan2(det, dot)
#     angles = np.arctan2(det, dot)
    
#     # In-place conversion to degrees
#     np.degrees(angles, out=angles)
#     return angles

def angle_3points_deg(a, b, c):
    ba = a - b
    bc = c - b
    
    # Normalize vectors
    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)
    
    # Avoid division by zero
    ba_norm[ba_norm == 0] = 1e-6
    bc_norm[bc_norm == 0] = 1e-6
    
    # Calculate Cosine
    cosine_angle = np.sum(ba * bc, axis=1) / (ba_norm * bc_norm)
    
    # Clip to valid range [-1, 1] to prevent NaNs
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)