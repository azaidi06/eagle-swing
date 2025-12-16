import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def get_all_idx_bounds(highest_idxs, 
                       start_increment=105,
                       end_increment=105):
    final_idxs = [get_before_after_idx(idx, 
                                       start_increment=start_increment,
                                       end_increment=end_increment,
                                       ) for idx in highest_idxs]
    return final_idxs


def get_before_after_idx(idx_val, 
                         start_increment=90,
                         end_increment=90):
    init_idx = idx_val - start_increment
    final_idx = idx_val + end_increment
    return init_idx, final_idx


def find_each_first_higher_wrist(higher_wrist_idxs, 
                                 skip_frames=600, #60fps * 10 seconds
                                ):
    last_higher_idx = higher_wrist_idxs[0]
    highest_idxs = [last_higher_idx,] #holder for our highest frames
    while True:
        next_higher_idxs = np.where((higher_wrist_idxs - last_higher_idx) > skip_frames)[0]
        if len(next_higher_idxs) == 0:
            return highest_idxs
        last_higher_idx = higher_wrist_idxs[next_higher_idxs[0]]
        highest_idxs.append(last_higher_idx)


def find_all_higher_wrist_idxs(kps, conf_threshold=0.5):
    l_shoulder_y = kps[:, 5, 1]
    l_shoulder_conf = kps[:, 5, 2]
    r_shoulder_y = kps[:, 6, 1]
    r_shoulder_conf = kps[:, 6, 2]
    l_elbow_y = kps[:, 7, 1]
    l_elbow_conf = kps[:, 7, 2]
    r_elbow_y = kps[:, 8, 1]
    r_elbow_conf = kps[:, 8, 2]
    l_wrist_y = kps[:, 9, 1]
    l_wrist_conf = kps[:, 9, 2]
    r_wrist_y = kps[:, 10, 1]
    r_wrist_conf = kps[:, 10, 2]

    # Check confidence thresholds for all relevant keypoints
    all_confident = (
        (l_shoulder_conf >= conf_threshold) &
        (r_shoulder_conf >= conf_threshold) &
        (l_elbow_conf >= conf_threshold) &
        (r_elbow_conf >= conf_threshold) &
        (l_wrist_conf >= conf_threshold) &
        (r_wrist_conf >= conf_threshold)
    )
    #check that wrists are above elbows and shoulders
    left_wrist_above_elbow = l_wrist_y < l_elbow_y
    right_wrist_above_elbow = r_wrist_y < r_elbow_y
    left_wrist_above_sh = l_wrist_y < l_shoulder_y
    right_wrist_above_sh = r_wrist_y < r_shoulder_y
    
    
    combined_true = (all_confident & 
                     left_wrist_above_elbow & 
                     right_wrist_above_elbow & 
                     left_wrist_above_sh & 
                     right_wrist_above_sh)
    higher_idxs = np.where(combined_true)[0]
    return higher_idxs


def find_score_hand(kp_extractor,
                   min_consecutive=30,
                   max_gap=1,
                   confidence_threshold=0.3):
    left_shoulder = kp_extractor.l_sh
    right_shoulder = kp_extractor.r_sh
    left_wrist = kp_extractor.l_wrist
    right_wrist = kp_extractor.r_wrist
    left_knee = kp_extractor.l_knee
    right_knee = kp_extractor.r_knee
    right_elbow = kp_extractor.r_elbow
    left_elbow = kp_extractor.l_elbow
    num_frames = kp_extractor.kps.shape[0]
    
    valid_frames = (
        (left_shoulder[...,2] >= confidence_threshold) &
        (right_shoulder[...,2] >= confidence_threshold) &
        (left_wrist[...,2] >= confidence_threshold) &
        (right_wrist[...,2] >= confidence_threshold) &
        (right_elbow[...,2] >= confidence_threshold) &
        (left_elbow[...,2] >= confidence_threshold) 
    )

    # Label frames: 0=none, 1=left above, 2=right above
    left_above = (left_wrist[:, 1] < left_shoulder[:, 1]) \
                       & (left_wrist[:, 1] < left_elbow[:, 1]) \
                       & (right_wrist[:, 1] > right_shoulder[:, 1]) \
                       & (right_wrist[:, 1] < right_knee[:, 1])
    right_above = (right_wrist[:, 1] < right_shoulder[:, 1]) \
                        & (right_wrist[:,1] < right_elbow[:,1]) \
                        & (left_wrist[:, 1] > left_shoulder[:, 1]) \
                        & (left_wrist[:, 1] < left_knee[:, 1]) \
                        
    condition_label = np.zeros(num_frames, dtype=int)
    condition_label[left_above & valid_frames] = 1
    condition_label[right_above & valid_frames] = 2
    
    # 2. Find Consecutive Sequences with Gap Filling
    sequences = []
    i = 0
    while i < num_frames:
        current_label = condition_label[i]
        if current_label > 0:
            start = i
            end = i
            
            # Look ahead logic
            while i < num_frames:
                if condition_label[i] == current_label:
                    end = i
                    i += 1
                elif condition_label[i] == 0:
                    # Check if we can bridge this gap
                    gap_size = 0
                    found_reconnect = False
                    for lookahead in range(1, max_gap + 1):
                        if (i + lookahead) < num_frames:
                            if condition_label[i + lookahead] == current_label:
                                # Found reconnection, bridge the gap
                                i += lookahead
                                found_reconnect = True
                                break
                            elif condition_label[i + lookahead] != 0:
                                # Hit a different active label (e.g. switched Left to Right), stop immediately
                                break
                    
                    if not found_reconnect:
                        # Gap too big or switch occurred
                        break
                else:
                    # Switched to different non-zero label (e.g., 1 -> 2)
                    break
            
            length = end - start + 1
            if length >= min_consecutive:
                sequences.append({
                    'start_frame': start,
                    'end_frame': end,
                    'length': length,
                    'wrist_above': 'left' if current_label == 1 else 'right'
                })
        else:
            i += 1
    
    # 3. Create Mask
    frame_mask = np.zeros(num_frames, dtype=bool)
    for seq in sequences:
        frame_mask[seq['start_frame']:seq['end_frame']+1] = True
        
    return sequences, frame_mask



def get_top_idx(kps, y_axis_only=True,
               thresh_value=0.001):
    pos = kps[:, :2] #.r_wrist[:, :2]                 # (T, 2)
    pos_s = savgol_filter(pos, 
                          window_length=9, 
                          polyorder=2, 
                          axis=0)
    
    v = np.diff(pos_s, axis=0)               # (T-1, 2) frame-to-frame velocity
    speed = np.linalg.norm(v, axis=1)
    thresh = thresh_value * speed.max()
    if y_axis_only:
        vy = v[:, 1]                              # vertical component
        sign = np.sign(vy)
        cross = np.where((sign[:-1] < 0) & (sign[1:] > 0) &
                         (speed[:-1] > thresh) & (speed[1:] > thresh))[0]
        change_frame = cross[0] + 1 if len(cross) else None
    
    else:
        # 1. Compute dot product between consecutive velocity vectors
        # Shapes: (T-2, 2) * (T-2, 2) -> (T-2,)
        dot = np.sum(v[:-1] * v[1:], axis=1)
        
        # 2. Check speed for the SAME vectors used in the dot product
        # speed[:-1] corresponds to v[:-1] (velocity entering the turn)
        # speed[1:]  corresponds to v[1:]  (velocity exiting the turn)
        candidates = np.where((dot < 0) & 
                              (speed[:-1] > thresh) & 
                              (speed[1:] > thresh))[0]
        
        # 3. Adjust index
        # candidates[0] is the index in the 'dot' array.
        # dot[i] compares frame i and i+1 (in velocity space), which is frame i+1 and i+2 in position space.
        # We usually want the frame *at* the vertex (the point between the vectors).
        # If v[0] is pos[1]-pos[0], the "turn" happens at pos[1].
        # So index + 1 is generally correct for the "vertex" frame.
        change_frame = candidates[0] + 1 if len(candidates) > 0 else None
    return change_frame 




def plot_feature_across_instances(data_instances, 
                                  feature_name, 
                                  plot_names=None, 
                                  figsize=(15, 10), 
                                  highlight_frames_red=None,        # Original Red Highlight
                                  highlight_frames_orange=None,
                                  highlight_frames_magenta=None,  # New Magenta Highlight
                                  highlight_frames_eblue=None,  # New Electric Blue Highlight
                                  smooth=False,
                                  smooth_window=7,
                                  smooth_poly_order=3): # New Orange Highlight
    """
    Plots the trajectory of a SINGLE feature across MULTIPLE data instances
    in a 3-column grid with optional Red and Orange highlight points.
    
    Args:
        data_instances (list): A list of class instances containing the data.
        feature_name (str): The specific attribute name to plot.
        plot_names (list of str, optional): Titles for each subplot.
        figsize (tuple): Dimensions of the figure.
        highlight_frames (list of int, optional): List of frame indices to highlight in RED.
        highlight_frames_orange (list of int, optional): List of frame indices to highlight in ORANGE.
    """
    # 1. Setup Grid
    num_plots = len(data_instances)
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    
    if num_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    
    first_valid_plot = None
    
    # 2. Loop through instances
    for i, instance in enumerate(data_instances):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        try:
            data = getattr(instance, feature_name)
        except AttributeError:
            ax.text(0.5, 0.5, f"Feature '{feature_name}'\nnot found", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        if data.ndim == 3 and data.shape[1] == 1:
            data = data.squeeze(1)
            
        if data.ndim != 2 or data.shape[1] < 2:
            ax.text(0.5, 0.5, f"Invalid Shape\n{data.shape}", 
                    ha='center', va='center')
            continue
        if smooth:
            x = savgol_filter(data[:, 0], smooth_window, smooth_poly_order)
            y = savgol_filter(data[:, 1], smooth_window, smooth_poly_order)
        
        x = data[:, 0]
        y = data[:, 1]
        
        # Plot trajectory
        time_colors = np.arange(len(x))
        sc = ax.scatter(x, y, c=time_colors, cmap='viridis', s=15, alpha=0.8)

        # --- Red Highlight  ---
        if highlight_frames_red and i < len(highlight_frames_red):
            h_frame_red = highlight_frames_red[i]
            if h_frame_red is not None and 0 <= h_frame_red < len(x):
                ax.scatter(x[h_frame_red], y[h_frame_red], color='red', #alpha=0.8, 
                                                                s=200, 
                                                                zorder=5, 
                                                                edgecolors='black')

        # --- Orange Highlight  ---
        if highlight_frames_orange and i < len(highlight_frames_orange):
            h_frame_orange = highlight_frames_orange[i]
            if h_frame_orange is not None and 0 <= h_frame_orange < len(x):
                ax.scatter(x[h_frame_orange], y[h_frame_orange], color='orange', 
                                                                #alpha=0.6,
                                                                s=120, 
                                                                zorder=6, 
                                                                edgecolors='white')
        
        # --- Electric Blue ---
        if highlight_frames_eblue and i < len(highlight_frames_eblue):
            h_frame_eblue = highlight_frames_eblue[i]
            if h_frame_eblue is not None and 0 <= h_frame_eblue < len(x):
                ax.scatter(x[h_frame_eblue], y[h_frame_eblue], color='#0000FF', 
                                                                #alpha=0.6,
                                                                s=80, 
                                                                zorder=7, 
                                                                edgecolors='white')

        # --- Magenta  ---
        if highlight_frames_magenta and i < len(highlight_frames_magenta):
            h_frame_magenta = highlight_frames_magenta[i]
            if h_frame_magenta is not None and 0 <= h_frame_magenta < len(x):
                ax.scatter(x[h_frame_magenta], y[h_frame_magenta], color='magenta', 
                                                                    #alpha=0.6,
                                                                    s=40, 
                                                                    zorder=8, 
                                                                    edgecolors='white')

        # Title Logic
        if plot_names and i < len(plot_names):
            ax.set_title(plot_names[i])
        else:
            ax.set_title(f"Instance {i}: {feature_name}")
        
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if first_valid_plot is None:
            first_valid_plot = sc

    # 3. Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # 4. Add Colorbar
    if first_valid_plot:
        cbar = fig.colorbar(first_valid_plot, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Frame Progression')
    
    plt.suptitle(f"Comparison of '{feature_name}' Across Instances", fontsize=14)
    plt.show()


def plot_feature_trajectories(data_instance, 
                              feature_names=None, 
                              figsize=(15, 6),
                             highlight_frame=None):
    """
    Plots trajectories for specific attributes retrieved from a class instance.
    
    Args:
        data_instance: The class instance containing the data as attributes.
        feature_names (list of str): List of attribute names (e.g., ['r_wrist', 'hip_center']).
                                     Each attribute should return a numpy array of shape (Frames, 2).
        figsize (tuple): Dimensions of the entire figure.
    """
    if feature_names is None:
        feature_names = ['r_sh', 'l_sh', 
                         'r_elbow', 'l_elbow', 
                         'r_wrist', 'l_wrist', 
                         'r_hip', 'l_hip', 
                         'r_knee', 'l_knee', 
                         #'r_ankle', 'l_ankle'
                        ]
    # 1. Setup Grid
    num_plots = len(feature_names)
    cols = 5
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten()
    
    # 2. Loop through feature names
    # We define the color mapper inside the loop or based on the first valid item
    # to ensure it matches the frame count of the data.
    
    first_valid_plot = None
    
    for i, name in enumerate(feature_names):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        # --- KEY CHANGE: Dynamic Retrieval ---
        try:
            data = getattr(data_instance.kps, name)
        except AttributeError:
            ax.text(0.5, 0.5, f"Attribute '{name}'\not found", 
                    ha='center', va='center', transform=ax.transAxes)
            continue

        x = data[:, 0]
        y = data[:, 1]
        
        # Create Time Colors (based on length of this specific array)
        time_colors = np.arange(len(x))
        
        # Plot
        sc = ax.scatter(x, y, c=time_colors, cmap='viridis', s=15, alpha=0.8)

        # Highlight the specified frame with a red dot
        if highlight_frame is not None and 0 <= highlight_frame < len(x):
            ax.scatter(x[highlight_frame], y[highlight_frame], color='red', s=50, zorder=5, edgecolors='black')
        # --- KEY CHANGE: Labeling ---
        ax.set_title(name.replace('_', ' ').title()) # e.g., "r_wrist" -> "R Wrist"
        
        # Standard CV Formatting
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if first_valid_plot is None:
            first_valid_plot = sc

    # 3. Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # 4. Add Colorbar (referenced to the first valid plot found)
    if first_valid_plot:
        cbar = fig.colorbar(first_valid_plot, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Frame Progression')
    
    plt.show()