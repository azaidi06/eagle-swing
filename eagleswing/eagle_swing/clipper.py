from eagle_swing.data_class import *
from eagle_swing.find_landmarks import *
from eagle_swing.video_utils import *

def get_highest_wrist_only(pkl_file):
    keypoints = KpExtractor(pkl_file).kps
    all_higher_wrist_frames = find_all_higher_wrist_idxs(keypoints)
    first_higher_wrist_idx = find_each_first_higher_wrist(all_higher_wrist_frames)
    return first_higher_wrist_idx


def get_swing_idx_bounds(pkl_file,
                         start_increment=90,
                         end_increment=90,
                         post_processors=[
                             #normalize_by_average_torso,
                             #center_by_average_torso,
                             #align_procrustes,
                             rescale_for_visualization
                         ]
                             ):
    keypoints = KpExtractor(pkl_file,
                        post_processors=post_processors).kps
    all_higher_wrist_frames = find_all_higher_wrist_idxs(keypoints)
    first_higher_wrist_idx = find_each_first_higher_wrist(all_higher_wrist_frames)
    all_idx_bounds = get_all_idx_bounds(first_higher_wrist_idx, 
                                    start_increment=start_increment,
                                    end_increment=end_increment)
    return keypoints, all_idx_bounds


def get_swing_kps(keypoints, all_idx_bounds): 
    final_kps = [keypoints[idx[0]:idx[1]] for idx in all_idx_bounds]
    return final_kps, all_idx_bounds

def save_swing_frames(video_path, 
                      all_idx_bounds,
                      resize_dim=(256, 256)): 
    frames_list = []
    for idx_bound in all_idx_bounds:
        frames, fps = get_frames(video_path=video_path,
                            start_idx=idx_bound[0],
                            num_frames=idx_bound[1]-idx_bound[0],
                            resize_dim=resize_dim)
        frames_list.append(frames)
    return frames_list


def get_clips(pkl_file, 
              start_increment=120, 
              end_increment=600,
              resize_dim=(256, 256)):
    all_kps, video_idx_bounds = get_swing_idx_bounds(pkl_file, 
                                        start_increment=120,
                                        end_increment=600,)
    swing_kps, all_idx_bounds = get_swing_kps(all_kps, video_idx_bounds)
    video_fname = f"{pkl_file.name.split('.')[0]}.mp4"
    video_file = pkl_file.parent/video_fname
    swing_frames = save_swing_frames(video_file, 
                                     video_idx_bounds,
                                     resize_dim=resize_dim)
    return swing_kps, swing_frames, all_idx_bounds




# import itertools

# clip_holder = []
# scores_series = pd.Series(stef_scores.values()).map(lambda x: [f'swing_{idx}_score_{score}' for idx, score in enumerate(x)])
# init_df = pd.DataFrame([video_names, scores_series], index=['video_name','swing_info']).T
# for _, row in init_df.iterrows():
#     vname = row['video_name']
#     for item in row['swing_info']:
#         clip_holder.append(f'{vname}_{item}')


# final_df = pd.DataFrame(holder, columns=['clip_name'])
# final_df['video_name'] = final_df.clip_name.map(lambda x: '_'.join(x.split('_')[:2]))
# final_df['swing_idx'] = final_df.clip_name.map(lambda x: x.split('_')[3])
# final_df['score'] = final_df.clip_name.map(lambda x: x.split('_')[-1])
# highest_wrists_list = [get_highest_wrist_only(file) for file in pkl_files]
# flat_wrist_list = list(itertools.chain.from_iterable(highest_wrists_list))
# df['pkl_path'] = df.video_name.map(lambda x: fname2path[x])
# df['first_highest_wrist_idx'] = flat_wrist_list
###
### df.to_csv('stef_lbls.csv', index=False) ##
###