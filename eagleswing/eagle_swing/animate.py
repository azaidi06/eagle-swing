from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np


def animate_keypoints(keypoint_sequences, dark_mode=False, 
                      labels=None, vertical=False, fps=60):
    """
    Animates 1 to 9 keypoint sequences.
    
    Args:
        keypoint_sequences: List of arrays (Frames, KPs, 2).
        dark_mode (bool): Black background with lime elements if True.
        vertical (bool): If True and <=3 sequences, stacks plots vertically. 
                        Ignored for >3 sequences.
        fps (int): Frames per second.
        labels (list): Optional labels for each sequence.
    """
    import math
    
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9),          # Left Arm
        (6, 8), (8, 10),                 # Right Arm
        (5, 11), (6, 12),                # Torso
        (11, 12),                        # Hips
        (11, 13), (13, 15),              # Left Leg
        (12, 14), (14, 16)               # Right Leg
    ]
    
    # 1. Input Normalization
    if not isinstance(keypoint_sequences, list):
        keypoint_sequences = [keypoint_sequences]
    
    # Updated Limit: 9 videos max
    if len(keypoint_sequences) > 9:
        print("Warning: Limiting visualization to first 9 sequences.")
        keypoint_sequences = keypoint_sequences[:9]
        
    num_plots = len(keypoint_sequences)
    
    # Validate labels
    if labels and len(labels) != num_plots:
        print(f"Warning: Provided {len(labels)} labels for {num_plots} plots. Labels may not match.")
    
    # 2. Style Configuration
    if dark_mode:
        bg_color = 'black'
        line_color = 'lime'
        joint_color = 'white'
        text_color = 'white'
    else:
        bg_color = 'white'
        line_color = 'black'
        joint_color = 'red'
        text_color = 'black'

    # 3. Figure & Subplot Setup with 2/3 column logic
    if num_plots <= 3:
        # Original behavior for 1-3 plots
        if vertical:
            nrows, ncols = num_plots, 1
            figsize = (2, 2 * num_plots)
        else:
            nrows, ncols = 1, num_plots
            figsize = (2 * num_plots, 2)
    else:
        # Dynamic Grid Logic
        if num_plots == 4:
            ncols = 2  # Use 2 columns for exactly 4 plots (2x2)
        else:
            ncols = 3  # Use 3 columns for 5+ plots (up to 3x3)
            
        nrows = math.ceil(num_plots / ncols)
        # Reduced size: 1.5 inches per subplot
        figsize = (1.5 * ncols, 1.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    
    # Flatten axes array for consistent indexing
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Hide extra subplots (e.g., if you have 5 plots in a 3x2=6 grid)
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
        
    plot_objects = []

    # Loop through axes, data, and optionally labels
    for i, (ax, data) in enumerate(zip(axes[:num_plots], keypoint_sequences)):
        if data.shape[-1] == 3:
            data = data[..., :2]
            
        ax.set_facecolor(bg_color)
        ax.axis('off')
        ax.set_aspect('equal')

        # --- Add Label if provided ---
        if labels and i < len(labels):
            # Smaller font for grid layouts
            fontsize = 6 if num_plots > 3 else 8
            ax.set_title(labels[i], color=text_color, fontsize=fontsize, pad=4)
        
        # --- LIMITS CONFIGURATION ---
        all_x = data[..., 0].flatten()
        all_y = data[..., 1].flatten()
        valid_mask = (all_x > 0.1) & (all_y > 0.1)

        if valid_mask.any():
            vx, vy = all_x[valid_mask], all_y[valid_mask]
            pad = 50
            ax.set_xlim(vx.min() - pad, vx.max() + pad)
            ax.set_ylim(vy.max() + pad, vy.min() - pad)
        else:
            ax.set_xlim(0, 640)
            ax.set_ylim(480, 0)

        # Create graphics objects (scaled down for grid views)
        point_size = 15 if num_plots > 3 else 30
        line_width = 1 if num_plots > 3 else 2
        
        scat = ax.scatter([], [], s=point_size, c=joint_color, zorder=2)
        lines = [ax.plot([], [], color=line_color, lw=line_width)[0] for _ in skeleton]
        
        plot_objects.append({'scat': scat, 'lines': lines, 'data': data, 'ax': ax})

    if num_plots > 3:
        plt.tight_layout(pad=0.5)

    # 4. Animation Logic
    def init():
        all_artists = []
        for obj in plot_objects:
            obj['scat'].set_offsets(np.empty((0, 2)))
            for line in obj['lines']:
                line.set_data([], [])
            all_artists.append(obj['scat'])
            all_artists.extend(obj['lines'])
        return all_artists

    def update(frame_idx):
        all_artists = []
        for obj in plot_objects:
            data = obj['data']
            idx = min(frame_idx, len(data) - 1)
            current_frame = data[idx]
            
            mask = (current_frame[:, 0] > 0.1) & (current_frame[:, 1] > 0.1) & ~np.isnan(current_frame[:, 0])
            
            obj['scat'].set_offsets(current_frame[mask])
            all_artists.append(obj['scat'])
            
            for line, (start, end) in zip(obj['lines'], skeleton):
                if mask[start] and mask[end]:
                    line.set_data(
                        [current_frame[start, 0], current_frame[end, 0]],
                        [current_frame[start, 1], current_frame[end, 1]]
                    )
                else:
                    line.set_data([], [])
                all_artists.append(line)
                
        return all_artists

    max_frames = max(len(d) for d in keypoint_sequences)

    anim = FuncAnimation(
        fig, 
        update, 
        frames=max_frames, 
        init_func=init, 
        blit=True, 
        interval=1000/fps
    )
    
    plt.close()
    return anim
