import ipywidgets as widgets
from IPython.display import display
from tqdm import tqdm
import numpy as np
import cv2

def get_frames(video_path, 
               per_second=False,
               start_idx=0,
               num_frames=10,
               resize_dim=(256,256),
               show_progress=True):
    
    capture = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = round(capture.get(cv2.CAP_PROP_FPS))
    
    # Determine step size
    if per_second:
        # E.g. 30 FPS / 6 = 5. We want every 5th frame.
        step_size = int(max(1, frames_per_second / 6)) 
    else: 
        step_size = 1

    # Handle dimensions
    if resize_dim:
        frame_width, frame_height = resize_dim
    else:
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # Handle num_frames default
    if num_frames is None:
        # Calculate how many frames we can actually fit
        remaining_frames = frame_count - start_idx
        num_frames = int(remaining_frames // step_size)

    # Pre-allocate array
    video_array = np.empty((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    
    # 1. INITIAL SEEK (Only done once!)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    frames_collected = 0
    
    # Progress bar logic
    pbar = tqdm(total=num_frames, disable=not show_progress)
    
    try:
        while frames_collected < num_frames:
            # Read the actual frame we want
            ret, frame = capture.read()
            if not ret:
                break
            
            # Process frame
            if resize_dim:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_LINEAR)
            
            # Convert color and store
            video_array[frames_collected] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_collected += 1
            pbar.update(1)
            
            # 2. FAST SKIP
            # Skip (step_size - 1) frames to get to the next target
            # grab() is much faster than read() because it skips decoding
            if step_size > 1:
                for _ in range(step_size - 1):
                    capture.grab()
                    
    finally:
        capture.release()
        pbar.close()

    # Trim array if we hit end of video early
    if frames_collected < num_frames:
        video_array = video_array[:frames_collected]

    return video_array, frames_per_second


def view_videos(video_list, fps=60):
    """
    Display up to videos stored as numpy arrays in a Jupyter Notebook 
    with global/individual sliders and playback controls.
    
    Args:
        video_list (list): List of numpy arrays. Each array should be (Frames, Height, Width, Channels).
                           If Channels is missing, it's assumed to be grayscale.
                           Pixel values can be 0-255 (uint8) or 0.0-1.0 (float).
        fps (int): Frames per second for the playback widget.
    """
    # 1. Input Validation and Standardization
    if not isinstance(video_list, list):
        video_list = [video_list]
        
    if len(video_list) > 3:
        print("Warning: limiting display to first 3 videos.")
        video_list = video_list[:3]

    # Standardize videos to uint8 (0-255) for display
    processed_videos = []
    for vid in video_list:
        # Normalize float 0-1 to uint8 0-255
        if vid.dtype != np.uint8:
            if vid.max() <= 1.0:
                vid = (vid * 255).astype(np.uint8)
            else:
                vid = vid.astype(np.uint8)
        processed_videos.append(vid)

    n_videos = len(processed_videos)
    max_frames = max(len(v) for v in processed_videos)
    
    # 2. Helper: Frame to JPEG Byte Conversion
    def array_to_bytes(frame_array):
        # Handle RGB vs BGR for OpenCV (assuming input is RGB)
        if frame_array.ndim == 3 and frame_array.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_array # Grayscale
            
        _, encoded_img = cv2.imencode('.jpg', frame_bgr)
        return encoded_img.tobytes()

    # 3. Create Image Widgets
    image_widgets = []
    for vid in processed_videos:
        # Initialize with the first frame
        initial_bytes = array_to_bytes(vid[0])
        img_w = widgets.Image(value=initial_bytes, format='jpg', width=300, height=300)
        image_widgets.append(img_w)

    # 4. Controls
    # Sync Checkbox
    sync_check = widgets.Checkbox(value=True, description='Sync Sliders')
    
    # Global Controls
    play_widget = widgets.Play(value=0, min=0, max=max_frames-1, step=1, interval=int(1000/fps))
    global_slider = widgets.IntSlider(value=0, min=0, max=max_frames-1, description='Global Frame')
    widgets.jslink((play_widget, 'value'), (global_slider, 'value'))
    
    # Individual Sliders
    ind_sliders = []
    for i in range(n_videos):
        s = widgets.IntSlider(value=0, min=0, max=len(processed_videos[i])-1, 
                              description=f'Vid {i+1}', disabled=True)
        ind_sliders.append(s)

    # 5. Update Logic
    def update_frames(change=None):
        # If Sync is ON, ignore individual slider changes (logic handled by global)
        if sync_check.value:
            current_frame = global_slider.value
            for i, vid in enumerate(processed_videos):
                # Clamp frame index to video length
                idx = min(current_frame, len(vid) - 1)
                image_widgets[i].value = array_to_bytes(vid[idx])
                # Update individual sliders visual state without triggering events
                ind_sliders[i].unobserve_all()
                ind_sliders[i].value = idx
                ind_sliders[i].observe(update_individual, names='value')
        else:
            pass # Managed by individual callbacks

    def update_individual(change):
        if not sync_check.value:
            # Find which slider changed
            for i, slider in enumerate(ind_sliders):
                if slider == change['owner']:
                    idx = change['new']
                    image_widgets[i].value = array_to_bytes(processed_videos[i][idx])

    def on_sync_change(change):
        sync = change['new']
        if sync:
            # Enable Global, Disable Individual
            global_slider.disabled = False
            play_widget.disabled = False
            for s in ind_sliders:
                s.disabled = True
            update_frames() # Snap to global position
        else:
            # Disable Global, Enable Individual
            global_slider.disabled = True
            play_widget.disabled = True
            for s in ind_sliders:
                s.disabled = False
            
    # 6. Wire Events
    global_slider.observe(update_frames, names='value')
    sync_check.observe(on_sync_change, names='value')
    
    for s in ind_sliders:
        s.observe(update_individual, names='value')

    # 7. Layout
    # Video Row
    video_box = widgets.HBox(image_widgets)
    
    # Control Rows
    global_ctrl_box = widgets.HBox([play_widget, global_slider, sync_check])
    ind_ctrl_box = widgets.HBox(ind_sliders)
    
    ui = widgets.VBox([video_box, global_ctrl_box, ind_ctrl_box])
    display(ui)


def view_videos_grid(video_list, fps=60, ncols=3):
    """
    Display multiple videos (e.g., 6 or 9) in a grid layout within Jupyter Notebook.
    
    Args:
        video_list (list): List of numpy arrays (Frames, Height, Width, Channels).
        fps (int): Playback speed.
        ncols (int): Number of columns in the grid (default 3).
    """
    # 1. Input Validation
    if not isinstance(video_list, list):
        video_list = [video_list]
        
    # Soft warning instead of hard cut-off
    if len(video_list) > 9:
        print(f"Warning: Displaying {len(video_list)} videos might cause performance lag.")

    # Standardize to uint8
    processed_videos = []
    for vid in video_list:
        if vid.dtype != np.uint8:
            if vid.max() <= 1.0:
                vid = (vid * 255).astype(np.uint8)
            else:
                vid = vid.astype(np.uint8)
        processed_videos.append(vid)

    n_videos = len(processed_videos)
    max_frames = max(len(v) for v in processed_videos)
    
    # 2. Frame to JPEG Conversion
    def array_to_bytes(frame_array):
        if frame_array.ndim == 3 and frame_array.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_array
        _, encoded_img = cv2.imencode('.jpg', frame_bgr)
        return encoded_img.tobytes()

    # 3. Create Image Widgets
    # Reduced width slightly (250px) to fit 3-4 columns comfortably
    image_widgets = []
    for vid in processed_videos:
        initial_bytes = array_to_bytes(vid[0])
        img_w = widgets.Image(value=initial_bytes, format='jpg', width=250, height=250)
        image_widgets.append(img_w)

    # 4. Controls
    sync_check = widgets.Checkbox(value=True, description='Sync Sliders')
    
    play_widget = widgets.Play(value=0, min=0, max=max_frames-1, step=1, interval=int(1000/fps))
    global_slider = widgets.IntSlider(value=0, min=0, max=max_frames-1, description='Global')
    widgets.jslink((play_widget, 'value'), (global_slider, 'value'))
    
    ind_sliders = []
    for i in range(n_videos):
        s = widgets.IntSlider(value=0, min=0, max=len(processed_videos[i])-1, 
                              description=f'V{i+1}', disabled=True, 
                              layout=widgets.Layout(width='95%')) # Fit within column
        ind_sliders.append(s)

    # 5. Update Logic (Unchanged functionality, scales with list size)
    def update_frames(change=None):
        if sync_check.value:
            current_frame = global_slider.value
            for i, vid in enumerate(processed_videos):
                idx = min(current_frame, len(vid) - 1)
                image_widgets[i].value = array_to_bytes(vid[idx])
                ind_sliders[i].unobserve_all()
                ind_sliders[i].value = idx
                ind_sliders[i].observe(update_individual, names='value')

    def update_individual(change):
        if not sync_check.value:
            for i, slider in enumerate(ind_sliders):
                if slider == change['owner']:
                    idx = change['new']
                    image_widgets[i].value = array_to_bytes(processed_videos[i][idx])

    def on_sync_change(change):
        if change['new']:
            global_slider.disabled = False
            play_widget.disabled = False
            for s in ind_sliders: s.disabled = True
            update_frames()
        else:
            global_slider.disabled = True
            play_widget.disabled = True
            for s in ind_sliders: s.disabled = False
            
    global_slider.observe(update_frames, names='value')
    sync_check.observe(on_sync_change, names='value')
    for s in ind_sliders:
        s.observe(update_individual, names='value')

    # 6. Grid Layout Construction
    def make_grid(widget_list, cols):
        rows = []
        for i in range(0, len(widget_list), cols):
            row_items = widget_list[i : i+cols]
            rows.append(widgets.HBox(row_items))
        return widgets.VBox(rows)

    # Layout videos and sliders in matching grids
    video_grid = make_grid(image_widgets, ncols)
    slider_grid = make_grid(ind_sliders, ncols)
    
    global_ctrl_box = widgets.HBox([play_widget, global_slider, sync_check])
    
    # Final Stack: Videos -> Global Controls -> Individual Sliders
    ui = widgets.VBox([video_grid, global_ctrl_box, slider_grid])
    display(ui)