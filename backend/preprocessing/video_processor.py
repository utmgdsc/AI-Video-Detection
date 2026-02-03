"""
Video preprocessing utilities.

Uses OpenCV for frame extraction and face detection.
"""

import cv2
import numpy as np
from PIL import Image


def extract_frames(video_path, sample_rate=1):
    """
    Extract frames from video.

    Args:
        video_path: Path to video file
        mtcnn: mtcnn is the model that extract faces
        device: "cuda" or "cpu"
        batch_size: size of each batch
        sample_rate: Extract every Nth frame (1 = all frames)

    Returns:
        list: List of frames as numpy arrays (BGR format)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    # output_dir = "frame_extracted/"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()
    return frames


def detect_faces(frames, mtcnn, batch_size):
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
    batch = []
    batch_idx = 0
    frame_idx = 0
    faces = []
    while frame_idx < len(frames):
        batch.append(frames[frame_idx])
        frame_idx += 1
        batch_idx += 1
        # Process batch
        if batch_idx >= batch_size or frame_idx == len(frames):
            pil_batch = [Image.fromarray(f) for f in batch]

            batch_result = mtcnn(pil_batch)
            if batch_result is not None:
                for result in batch_result:
                    # no face
                    if result is None:
                        continue

                    # multiple faces in a frame
                    if result.ndim == 4:
                        for face in result:
                            faces.append(face)
                    else:
                        # one face in a frame
                        faces.append(result)
            batch_idx = 0
            batch = []
    return faces


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
