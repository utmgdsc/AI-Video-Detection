"""
Video preprocessing utilities.

Uses OpenCV for frame extraction and face detection.
"""

import cv2
import numpy as np


def extract_frames(video_path, sample_rate=1):
    """
    Extract frames from video.

    Args:
        video_path: Path to video file
        sample_rate: Extract every Nth frame (1 = all frames)

    Returns:
        list: List of frames as numpy arrays (BGR format)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def detect_faces(frames):
    """
    Detect and crop faces from frames.

    Args:
        frames: List of frames (numpy arrays)

    Returns:
        list: List of cropped face images
    """
    # TODO: Implement face detection
    # Options:
    # - cv2.CascadeClassifier (simple, fast)
    # - dlib (more accurate)
    # - MTCNN (deep learning based)

    raise NotImplementedError("Implement detect_faces()")


def separate_audio(video_path, output_path):
    """
    Extract audio track from video.

    Args:
        video_path: Path to input video
        output_path: Path to save extracted audio

    Returns:
        str: Path to extracted audio file
    """
    # TODO: Implement using ffmpeg or moviepy
    # Example with moviepy:
    # from moviepy.editor import VideoFileClip
    # video = VideoFileClip(video_path)
    # video.audio.write_audiofile(output_path)

    raise NotImplementedError("Implement separate_audio()")
