from mmpose.apis import MMPoseInferencer
from mmengine.logging import MMLogger
import pickle
import mmpose
from tqdm import tqdm
import numpy as np
import imageio
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def generate_labels(labeler, fpath, out_dir=None, return_results=False, save=True):
    if out_dir is None:
        out_dir = 'keypoints/'
    if save is False:
        out_dir = ''
    fname = str(fpath)
    lbl_generator = labeler(fname, show=False, vis_out_dir=out_dir)
    results = [result['predictions'] for idx, result in tqdm(enumerate(lbl_generator))]
    save_keypoints(results, fname=fname, out_dir=out_dir)


def save_keypoints(results, fname, out_dir):
    fname = fname.split('/')[-1].split('.')[0]
    indexed_data = {f"frame_{i}": frame_result[0][0] for i, frame_result in enumerate(results)}
    with open(f'{out_dir}/{fname}.pkl', 'wb') as f:
        pickle.dump(indexed_data, f)



def get_labeler(model_name='vit'):
    if model_name == 'vit':
        model_name = 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
    if model_name == 'edpose':
        model_name = 'edpose_res50_8xb2-50e_coco-800x1333'
    inferencer = MMPoseInferencer(model_name)
    return inferencer


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
    video_idxs = np.arange(start_idx, frame_count, idx_stepper)
    #num_frames = len(video_idxs)
    video_array = np.empty((num_frames, 
                            frame_height, 
                            frame_width, 
                            3), 
                            dtype=np.uint8)
    for idx in tqdm(range(0,num_frames), disable=not show_progress):
        capture.set(cv2.CAP_PROP_POS_FRAMES, video_idxs[idx])
        ret, frame = capture.read()
        if not ret:
            break
                # Resize if dimensions provided
        if resize_dim:
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_LINEAR)
        video_array[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return video_array, frames_per_second#video_idxs