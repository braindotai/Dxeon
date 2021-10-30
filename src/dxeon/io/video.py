import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from ..utils import bgr2gray_cv2, rgb2bgr_cv2, get_aspect_ratio_dims

def read_moviepy(video_path: os.PathLike):
    clip = VideoFileClip(video_path)
    
    return clip, clip.fps, clip.size[0], clip.size[1], clip.duration

def read_cv2(video_path: os.PathLike):
    clip = cv2.VideoCapture(video_path)
    fps = cv2.CAP_PROP_FPS(clip)
    width = cv2.CAP_PROP_FRAME_WIDTH(clip)
    height = cv2.CAP_PROP_FRAME_HEIGHT(clip)
    frames = cv2.CAP_PROP_FRAME_COUNT(clip)

    return clip, clip.fps, width, height, frames / fps

class WriterCV2:
    def __init__(self, output_path, fps, max_dim = None, bgr2rgb = False, rgb2bgr = False) -> None:
        assert '.mp4' in output_path, f'\n\nOnly mp4 format is supported.\n'
        
        self.output_path = output_path
        self.fps = int(fps)
        self.max_dim = max_dim
        self.bgr2rgb = bgr2rgb
        self.rgb2bgr = rgb2bgr
        self.writer = None

    def write(self, frame: np.ndarray):
        if not self.writer:
            if self.max_dim:
                w, h = get_aspect_ratio_dims(frame, self.max_dim)
            else:
                w, h = frame.shape[1], frame.shape[0]

            self.writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        if self.bgr2rgb:
            frame = bgr2gray_cv2(frame)
        elif self.rgb2bgr:
            frame = rgb2bgr_cv2(frame)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()