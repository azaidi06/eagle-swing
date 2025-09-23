from fastai.vision.all import *
from IPython.display import Video
import cv2
import numpy as np


def get_frames(swing_path, 
               resize=True, 
               width=256, 
               height=256,
               debug=False):
    capture = cv2.VideoCapture(swing_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_array = np.empty((frame_count, frame_height, frame_width, 3), dtype=np.uint8)
    idx = 0
    while idx < frame_count:
        ret, frame = capture.read()
        if not ret:
            break
        video_array[idx] = frame
        idx += 1

    capture.release()
    if debug:
        print(video_array.shape)
    video_array = [convert_rgb(frame) for frame in video_array]
    if resize:
        video_array = np.array([resize_frame(frame, width, height) for frame in video_array])
    return video_array

def resize_frame(frame, width=256, height=256):
    return cv2.resize(frame, (width, height))

def convert_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_swing_df():
    all_file_paths = get_files('../../../data')

    swing_paths = [fp for fp in all_file_paths if str(fp).split('/')[-1][-3:] == 'mp4']
    full_vid_paths = [fp for fp in all_file_paths if str(fp).split('/')[-1][-3:] != 'mp4']
    swing_meta = [str(swing_paths[x]).split('.')[-2].split('/')[-1] for x in range(len(swing_paths))]


    video_origin = [('_').join(swing_meta[x].split('_')[:2]) for x in range(len(swing_paths))]
    og_vid_num = [swing_meta[x].split('_')[1] for x in range(len(swing_paths))]
    swing = [swing_meta[x].split('_')[3] for x in range(len(swing_paths))]
    score = [swing_meta[x].split('_')[-1] for x in range(len(swing_paths))]
    swing_dict = {'origin_video': video_origin,
                  'swing_index': swing,
                  'score': score,
                  'swing_path': swing_paths,
                  'og_vid_num': og_vid_num}
    swing_df = pd.DataFrame(swing_dict)
    return swing_df


def plot_three(image1, image2, image3):
    # Create figure with 1 row, 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each image on its respective axis
    axes[0].imshow(image1)
    axes[0].set_title('Frame 1')
    axes[0].axis('off')  # Remove axis ticks and labels

    axes[1].imshow(image2)
    axes[1].set_title('Frame 2')
    axes[1].axis('off')

    axes[2].imshow(image3)
    axes[2].set_title('Frame 3 or Diff')
    axes[2].axis('off')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()