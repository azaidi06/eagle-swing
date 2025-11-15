from swing_detect import *
from swing_data import *
from video_utils import *
from labeler import *
import pandas as pd
import ffmpeg
import os


def end_to_end_detect(fpath, start_idx=None, num_frames=None):
    df = find_each_swing(fpath, start_idx=start_idx, num_frames=num_frames,)#1500, )
    parent_dir = fpath.parent
    for x in range(len(df)):
        make_clip(input_file_path=fpath, 
                  output_folder_path=parent_dir,
                  row = df.iloc[x]
                 )
    return df


def make_output_filename(fname, swing_idx, score=None):
    return f'{fname}_swing_{swing_idx}_score_{score}'


def make_clip(input_file_path, 
              output_folder_path,
              row, 
              #duration_frames=90,  # Changed from time='0:03'
              crf='18',
              vcodec='libx264'):   # Changed from 'copy' since we need to use filter
    fname = input_file_path.name.split('.')[0]
    swing_idx, start_frame, end_frame = row.values
    output_file_name = make_output_filename(fname, swing_idx)
    output_file_path = f'{output_folder_path}/{fname}/{output_file_name}.mp4'
    import pdb
    #pdb.set_trace()
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


def ensure_out_dir(out_dir_fpath):
    if not os.path.isdir(out_dir_fpath):
        os.makedirs(out_dir_fpath)


def find_each_swing(video_path,
                    per_second=False, # only grab every fps frame
                    num_frames=None, #1500, # Pulls down all of the frames of video
                    start_idx=None, #600, # None starts from 0
                    resize_dim=(256,256),
                    show_progress=True,
                    model_type='vit', 
                    #out_dir='testing'
                   ):
    parent_dir = video_path.parent
    fname = video_path.name.split('.')[0]
    out_dir = f'{parent_dir}/{fname}'
    ensure_out_dir(out_dir)                        
    fname = video_path.name.split('.')[0]
    frames, fps = get_frames(video_path,
                             start_idx=start_idx,
                             per_second=per_second, # only grab every fps frame
                             num_frames=num_frames,#None, # Pulls down all of the frames of video
                             resize_dim=resize_dim,
                             show_progress=show_progress,
                            )
    output_filename = 'full_video.mp4'
    out_fpath = f'{out_dir}/{output_filename}'
    kp_fpath = f'{out_dir}/keypoints/{output_filename.split(".")[0]}.pkl'
    
    save_frames(frames=frames, fps=fps, 
            parent_dir=f'{parent_dir}/{fname}',
            output_filename=output_filename)
    #save_frames(frames=frames, fps=fps, fname=out_fpath)
                       
    process_label_video(out_fpath, out_dir=f'{out_dir}/keypoints')
    df = get_swing_idx_df(kps_fpath=kp_fpath, fname=fname, out_dir=out_dir)
    return df


def get_swing_idx_df(kps_fpath,
                     fname,
                     out_dir,
                     conf_threshold=0.7, 
                     frame_increment=90, # add 1.5 seconds before and the found idx
                     skip_frames=900, # 900 frames is 15 seconds
                     # ^ skips frames between swings
                     ):
    kps = KpExtractor(kps_fpath).keypoint_data.kps
    higher_idxs = find_all_higher_wrist_idxs(kps, conf_threshold=conf_threshold)
    highest_idxs = find_each_first_higher_wrist(higher_idxs, skip_frames=skip_frames) # 900 frames is 15 seconds
    all_idx_bounds = get_all_idx_bounds(highest_idxs, frame_increment=frame_increment)
    df = save_idx_df(fname, all_idx_bounds, out_dir)
    return df


def save_idx_df(fname, all_idx_bounds, out_dir):
    start_idxs = [idxs[0] for idxs in all_idx_bounds]
    end_idxs = [idxs[1] for idxs in all_idx_bounds]
    swing_idxs = [x for x in range(len(all_idx_bounds))]
    df = pd.DataFrame([swing_idxs, start_idxs, end_idxs], 
                 index=['swing_idx', 'start_idx', 'end_idx']).T
    df.to_csv(f'{out_dir}/{fname}.csv', index=False)
    return df