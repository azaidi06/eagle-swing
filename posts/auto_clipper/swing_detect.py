import pandas as pd
import numpy as np
import ffmpeg


def make_output_filename(fname, swing_idx, score=None):
    return f'{fname}_{swing_idx}_{score}'


def make_clip(input_file_path, 
              output_folder_path,
              row, 
              #duration_frames=90,  # Changed from time='0:03'
              crf='18',
              vcodec='libx264'):   # Changed from 'copy' since we need to use filter
    fname = input_file_path.name.split('.')[0]
    swing_idx, start_frame, end_frame = row.values
    output_file_name = make_output_filename(fname, swing_idx)
    output_file_path = f'{output_folder_path}/{output_file_name}.mp4'
    
    if os.path.isdir(output_folder_path) is False:
        os.mkdir(output_folder_path)
        
    # Use trim filter for frame-accurate cutting
    (
        ffmpeg.input(input_file_path)
        .trim(start_frame=start_frame, 
              end_frame=end_frame)
        .setpts('PTS-STARTPTS')  # Reset timestamps
        .output(output_file_path, 
                vcodec=vcodec,
                crf=crf, 
                acodec='aac')
        .global_args('-movflags', '+faststart')
        .overwrite_output()
        .run()
    )


def save_idx_df(fname, all_idx_bounds):
    start_idxs = [idxs[0] for idxs in all_idx_bounds]
    end_idxs = [idxs[1] for idxs in all_idx_bounds]
    swing_idxs = [x for x in range(len(all_idx_bounds))]
    df = pd.DataFrame([swing_idxs, start_idxs, end_idxs], 
                 index=['swing_idx', 'start_idx', 'end_idx']).T
    df.to_csv(f'{fname}.csv', index=False)


def get_all_idx_bounds(highest_idxs, frame_increment=90):
    final_idxs = [get_before_after_idx(idx, frame_increment) for idx in highest_idxs]
    return final_idxs


def get_before_after_idx(idx_val, increment=90):
    init_idx = idx_val - increment
    final_idx = idx_val + increment
    return init_idx, final_idx


def find_each_first_higher_wrist(higher_wrist_idxs, 
                                 skip_frames=600, #60fps * 10 seconds
                                ):
    #last_higher_idx = []
    last_higher_idx = higher_wrist_idxs[0]
    #last_higher_idx.append(higher_wrist_idxs[0])
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