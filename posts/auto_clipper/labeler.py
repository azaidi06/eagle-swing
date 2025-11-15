from mmpose.apis import MMPoseInferencer
from mmengine.logging import MMLogger
from tqdm import tqdm
import pickle
import mmpose
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def process_label_video(video_path, model_type='vit', out_dir='keypoints'):
    labeler = get_labeler('vit');
    generate_labels(labeler, video_path, out_dir=out_dir);    


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