import cv2
from moviepy.editor import VideoFileClip

def read_moviepy(video_path):
    clip = VideoFileClip(video_path)
    
    return clip, clip.fps, clip.size[0], clip.size[1], clip.duration

def read_cv2(video_path):
    clip = cv2.VideoCapture(video_path)
    fps = cv2.CAP_PROP_FPS(clip)
    width = cv2.CAP_PROP_FRAME_WIDTH(clip)
    height = cv2.CAP_PROP_FRAME_HEIGHT(clip)
    frames = cv2.CAP_PROP_FRAME_COUNT(clip)

    return clip, clip.fps, width, height, frames / fps