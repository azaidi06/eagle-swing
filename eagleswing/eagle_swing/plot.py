from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import numpy as np
import math


def plot_attributes(instances, attr_list, 
                    instance_labels=None,
                    scatter=False,
                    #smooth=True,           # <--- New Toggle
                    #window_len=7,          # <--- Smoothing intensity (must be odd)
                    #poly_order=2
                    function=None):   
    """
    Plots attributes with distinct hues, connected scatter points, 
    and INVERTED Y-AXIS (video coordinate style).
    """
    n_plots = len(attr_list)
    cols = 3
    rows = math.ceil(n_plots / cols)

    if instance_labels is None:
        instance_labels = [f"Instance {i}" for i in range(len(instances))]

    # --- Pre-calculate Colors ---
    red_indices = [i for i, lbl in enumerate(instance_labels) if lbl == 1]
    green_indices = [i for i, lbl in enumerate(instance_labels) if lbl != 1]
    
    red_colors = cm.Reds(np.linspace(0.4, 0.9, len(red_indices)))
    green_colors = cm.Greens(np.linspace(0.4, 0.9, len(green_indices)))
    
    instance_colors = [None] * len(instances)
    for i, idx in enumerate(red_indices):
        instance_colors[idx] = red_colors[i]
    for i, idx in enumerate(green_indices):
        instance_colors[idx] = green_colors[i]

    # --- Plotting ---
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), constrained_layout=True)
    
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, ax in enumerate(axes_flat):
        if i < n_plots:
            attr_name = attr_list[i]
            ax.set_title(attr_name)
            
            # --- INVERT Y-AXIS HERE ---
            if scatter: 
                ax.invert_yaxis()
            
            for j, instance in enumerate(instances):
                if hasattr(instance, attr_name):
                    data = getattr(instance, attr_name)
                    
                    # # --- SMOOTHING LOGIC ---
                    # if smooth and len(data) > window_len:
                    #     # axis=0 ensures we smooth 'over time' (rows) for both X and Y columns
                    #     try:
                    #         data = savgol_filter(data, window_len, poly_order, axis=0) 
                    #     except Exception as e:
                    #         print(f"Skipping smoothing for {attr_name}: {e}")
                    
                    lbl = instance_labels[j] if i == 0 else "_nolegend_"
                    c = instance_colors[j]

                    if scatter:
                        if function:
                            print(function)
                            data = function(data)
                        # Assuming data is (N, 2)
                        ax.plot(data[:, 0], data[:, 1], 
                                color=c, label=lbl,
                                marker='o', linestyle='-', 
                                markersize=4, alpha=0.7)
                    else:
                        if function:
                            data = function(data)
                        # Assuming data is (N,) or (N, 1)
                        ax.plot(data, color=c, label=lbl)
                else:
                    print(f"Warning: Attribute '{attr_name}' missing in {instance_labels[j]}")

            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

    if len(axes_flat) > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(1, 0), bbox_transform=fig.transFigure)

    plt.show()



def animate_keypoints_interactive(keypoint_sequences, dark_mode=False, 
                                  labels=None, vertical=False):
    """
    Interactive keypoint viewer with slider control showing frame numbers.
    
    Args:
        keypoint_sequences: List of arrays (Frames, KPs, 2).
        dark_mode (bool): Black background with lime elements if True.
        vertical (bool): If True and <=3 sequences, stacks plots vertically.
        labels (list): Optional labels for each sequence.
    """
    import math
    plt.ioff()
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    
    if not isinstance(keypoint_sequences, list):
        keypoint_sequences = [keypoint_sequences]
    
    if len(keypoint_sequences) > 9:
        print("Warning: Limiting visualization to first 9 sequences.")
        keypoint_sequences = keypoint_sequences[:9]
        
    num_plots = len(keypoint_sequences)
    
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

    # Setup figure
    if num_plots <= 3:
        if vertical:
            nrows, ncols = num_plots, 1
            figsize = (2, 2 * num_plots)
        else:
            nrows, ncols = 1, num_plots
            figsize = (2 * num_plots, 2)
    else:
        ncols = 2 if num_plots == 4 else 3
        nrows = math.ceil(num_plots / ncols)
        figsize = (1.5 * ncols, 1.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
        
    plot_objects = []

    for i, (ax, data) in enumerate(zip(axes[:num_plots], keypoint_sequences)):
        if data.shape[-1] == 3:
            data = data[..., :2]
            
        ax.set_facecolor(bg_color)
        ax.axis('off')
        ax.set_aspect('equal')

        if labels and i < len(labels):
            fontsize = 6 if num_plots > 3 else 8
            ax.set_title(labels[i], color=text_color, fontsize=fontsize, pad=4)
        
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

        point_size = 15 if num_plots > 3 else 30
        line_width = 1 if num_plots > 3 else 2
        
        scat = ax.scatter([], [], s=point_size, c=joint_color, zorder=2)
        lines = [ax.plot([], [], color=line_color, lw=line_width)[0] for _ in skeleton]
        
        plot_objects.append({'scat': scat, 'lines': lines, 'data': data})

    if num_plots > 3:
        plt.tight_layout(pad=0.5)

    max_frames = max(len(d) for d in keypoint_sequences)

    def update_plot(frame_idx):
        for obj in plot_objects:
            data = obj['data']
            idx = min(frame_idx, len(data) - 1)
            current_frame = data[idx]
            
            mask = (current_frame[:, 0] > 0.1) & (current_frame[:, 1] > 0.1) & ~np.isnan(current_frame[:, 0])
            
            obj['scat'].set_offsets(current_frame[mask])
            
            for line, (start, end) in zip(obj['lines'], skeleton):
                if mask[start] and mask[end]:
                    line.set_data(
                        [current_frame[start, 0], current_frame[end, 0]],
                        [current_frame[start, 1], current_frame[end, 1]]
                    )
                else:
                    line.set_data([], [])
        
        fig.canvas.draw_idle()

    # Create slider with frame numbers
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max_frames - 1,
        step=1,
        description='Frame:',
        continuous_update=True,
        readout=True,
        readout_format='d'
    )
    
    slider.observe(lambda change: update_plot(change['new']), names='value')
    
    # Initial plot
    update_plot(0)
    
    # Display slider and figure
    display(widgets.VBox([slider, fig.canvas]))



def plot_swing_metrics(swings, metrics, labels=None, highlight_frame=None, figsize=(12, 12)):
    """
    Plots specified metrics vertically with a legend next to EVERY subplot.
    Includes a highlighter for a specific frame index.
    
    Args:
        highlight_frame (int): The frame index (x-value) to highlight on all plots.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
        
    if labels is None:
        labels = [f"Swing {i+1}" for i in range(len(swings))]
    
    # Color profile extracted from lower_body.py
    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    
    # Optional: lower_body.py also varies linestyles to help distinguish swings
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    for ax, metric_name in zip(axes, metrics):
        # 1. Draw the vertical reference line first (so it sits behind the dots)
        if highlight_frame is not None:
            ax.axvline(x=highlight_frame, color='black', linestyle=':', alpha=0.5, label='Reference Frame')

        for i, swing in enumerate(swings):
            # Cycle through colors/styles if there are more swings than colors
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            if hasattr(swing, metric_name):
                data = getattr(swing, metric_name)
                if data is not None:
                    # Plot the main line with the matched color and linestyle
                    ax.plot(data, label=labels[i], color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
                    
                    # 2. Plot the highlight dot if the frame exists in this data
                    if highlight_frame is not None and 0 <= highlight_frame < len(data):
                        y_val = data[highlight_frame]
                        # Plot a marker ('o') at the specific x,y coordinate
                        ax.plot(highlight_frame, y_val, marker='o', markersize=8, 
                                color=color, markeredgecolor='white', markeredgewidth=1)
            else:
                pass

        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Legend setup
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')

    axes[-1].set_xlabel('Frame Index')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.show()


