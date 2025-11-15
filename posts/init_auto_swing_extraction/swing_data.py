from typing import TypeVar, Generic, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pickle
import cv2


class SwingMetaData:
    """Data Container for a swings video metadata
    """
    def __init__(self, path: str):
        """
        Initialize the swing data and associated metadata
        
        Args:
            path: file_path to data
        """
        self.path = path
        self.str_path = self.get_str_path()
        self.video_path = f'{self.str_path.split(".")[0]}.mp4'

    def get_str_path(self):
        # Just making sure the path being used is a str + not path object
        return str(self.path)

    def get_video_name(self):
        video = '_'.join(self.str_path.split('/')[1].split('_')[:2])
        return video
    


class SwingVideo(SwingMetaData):
    def __init__(self, video_path):
        super().__init__(video_path)
        self.video_path = video_path
        self.video = self.get_video_data()
    
    def get_video_data(self):
        return load_video_to_numpy(self.video_path)[0]

    def __len__(self) -> int:
        return len(self.video)




class SwingKeypointData(SwingMetaData):
    """
    Class to handle keypoint data
    Will take a videos metadata and pull down keypoint values and scores
    """
    
    def __init__(self, video_path):
        super().__init__(video_path)
        #self.metadata = metadata
        self.kp_dicts = self.get_kp_dicts()
        self.key_points = self.get_keypoints()
        self.scores = self.get_scores()
        self.kps = np.concatenate([self.key_points, np.expand_dims(self.scores, -1)], axis=2)


    def get_kp_dicts(self):
        # Get the frame by frame output dicts from pose estimation models
        with open(self.str_path, 'rb') as f:
            loaded_dicts = pickle.load(f)
        return loaded_dicts

        
    def get_keypoints(self):
        kp_dicts = self.kp_dicts
        return np.stack([self.kp_dicts[key]['keypoints'] for key in kp_dicts.keys()])
        
        
    def get_scores(self):
        kp_dicts = self.kp_dicts
        return np.stack([self.kp_dicts[key]['keypoint_scores'] for key in kp_dicts.keys()])


    def get_frame(self, idx):
        ''' Just grabs a single frames keypoints and scores
        '''
        kps = self.key_points[idx]
        scores = self.scores[idx]
        return np.column_stack((kps, scores))

    
    def get_frames(self, indexes):
        '''
        num_idxs = len(Indexes)
        Takes a list of indexes and returns an array wof [num_idxs, 17, 3]
        [1] 17 keypoint markers
        [2] 3 keypoint values/certaintainty (X, Y, Score)
        '''
        return np.stack([self.get_frame(idx) for idx in indexes])

    
    def __len__(self):
        return len(self.key_points)



def load_video_to_numpy(video_path):
    """Load video frames into a numpy array"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Pre-allocate array for all frames
    # Shape: (num_frames, height, width, channels)
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    
    # Read all frames
    frame_idx = 0
    while cap.isOpened() and frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frames[frame_idx] = frame
        frame_idx += 1
    
    cap.release()
    
    # Return only the frames that were successfully read
    return frames[:frame_idx], fps




class KpExtractor(SwingMetaData):
    def __init__(self, 
                 file_name, 
                 threshold_value=None):
        super().__init__(file_name)
        self.keypoint_data = SwingKeypointData(file_name)
        self.kps = self.threshold_score(self.keypoint_data, threshold_value)

        self.coco_idxs = {"L_SH":5, "R_SH":6, "L_ELBOW":7, "R_ELBOW":8, 
             "L_WRIST":9, "R_WRIST":10, "L_HIP":11, "R_HIP":12, 
             "L_KNEE":13, "R_KNEE":14, "L_ANKLE":15, "R_ANKLE":16}
        
        # Dynamically create attributes using setattr
        for attr_name, coco_key in self.coco_idxs.items():
            setattr(self, attr_name, 
                    self.kps[:, coco_key,#coco_idxs[coco_key],
                    :].astype(float).copy())

    def threshold_score(self, kps, threshold_value=0.5):
        if threshold_value is None: 
            return self.keypoint_data.kps.astype(float).copy()
        # punch up score values to a threshold
        kps = self.keypoint_data.kps.astype(float).copy()
        mask = kps[..., 2] < threshold_value
        kps[mask, 2] = threshold_value
        return kps