from tqdm import tqdm
import numpy as np
import imageio
import cv2

def save_frames(frames, 
                fname, 
                fps=60):
    imageio.mimwrite(fname, 
                     frames, 
                     fps=fps, 
                     quality=8, 
                     macro_block_size=1)

def get_frames(video_path, 
               per_second=True,
               debug=False,
               start_idx=0,
               num_frames=10,
               resize_dim=(256,256),
               show_progress=True,
               ):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames == None:
        num_frames = frame_count
    frames_per_second = round(capture.get(cv2.CAP_PROP_FPS))
    if resize_dim:
        frame_width, frame_height = resize_dim  # (width, height)
    else:
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if per_second:
        idx_stepper = frames_per_second / 6
    else: 
        idx_stepper = 1
    if start_idx == None:
        start_idx = 0
    video_idxs = np.arange(start_idx, frame_count, idx_stepper)
    #num_frames = len(video_idxs)
    video_array = np.empty((num_frames, 
                            frame_height, 
                            frame_width, 
                            3), 
                            dtype=np.uint8)
    if start_idx:
       capture.set(cv2.CAP_PROP_POS_FRAMES, video_idxs[0]) # seek to start index
    for idx in tqdm(range(0,num_frames), disable=not show_progress):
        if per_second:
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_idxs[idx]) #duplicate first time
            ## otherwise will push you 60 frames back everytime after
        ret, frame = capture.read()
        if not ret:
            break
                # Resize if dimensions provided
        if resize_dim:
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_LINEAR)
        video_array[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return video_array, frames_per_second


import subprocess
import numpy as np

def get_frames_ffmpeg(video_path,
                      per_second=True,
                      start_idx=0,
                      num_frames=10,
                      resize_dim=(256,256)):
    # Get video info
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    fps_str = subprocess.check_output(probe_cmd).decode().strip()
    fps = eval(fps_str)
    
    step = int(fps / 6) if per_second else 1
    frame_indices = list(range(start_idx, start_idx + num_frames * step, step))[:num_frames]
    
    # Build select filter
    select_expr = '+'.join([f'eq(n\\,{idx})' for idx in frame_indices])
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"select='{select_expr}',scale={resize_dim[0]}:{resize_dim[1]}",
        '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'
    ]
    
    result = subprocess.run(cmd, capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8)
    frames = frames.reshape((-1, resize_dim[1], resize_dim[0], 3))
    
    return frames, fps