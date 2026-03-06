"""
Video preprocessing utilities.

Uses OpenCV for frame extraction and face detection.
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
import torch


# This is extracted from efficientnet repo extract_faces.py
# Initialize MTCNN
# logger.info("Initializing MTCNN...")
# mtcnn = FastMTCNN(
#     stride=args.stride,
#     margin=args.margin,
#     min_face_size=args.min_face_size,
#     device=device
# )

# This is extracted from efficientnet repo extract_faces.py
# def process_video(
#     video_path: str,
#     output_dir: str,
#     mtcnn: FastMTCNN,
#     batch_size: int = 60,
#     frame_skip: int = 30
# ) -> Tuple[int, int]:
#     """
#     Process a single video file.

#     Args:
#         video_path: Path to video file
#         output_dir: Output directory
#         mtcnn: FastMTCNN instance
#         batch_size: Number of frames to process at once
#         frame_skip: Process every Nth frame

#     Returns:
#         Tuple of (frames_processed, faces_detected)
#     """
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     frames = []
#     frames_processed = 0
#     faces_detected = 0

#     for frame_idx in range(total_frames):
#         ret, frame = cap.read()

#         if not ret:
#             break

#         if frame_idx % frame_skip == 0 or frame_idx == total_frames - 1:
#             # Convert BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)

#             if len(frames) >= batch_size or frame_idx == total_frames - 1:
#                 # Process batch
#                 video_name = Path(video_path).stem
#                 faces = mtcnn(frames, output_dir, prefix=video_name)

#                 frames_processed += len(frames)
#                 faces_detected += faces

#                 frames = []

#     cap.release()
#     return frames_processed, faces_detected

# this is how we initialize mtcnn in ensemble

# Initialize MTCNN
# logger.info("Initializing MTCNN...")
# mtcnn = MTCNN(
#     margin=margin,
#     min_face_size=min_face_size,
#     device=device,
#     keep_all=True,
# )

def extract_frames(video_path, sample_rate=30):
    """
    Extract frames from video exactly like EfficientNet's process_video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Match the EfficientNet frame_skip logic
        if frame_idx % sample_rate == 0 or frame_idx == total_frames - 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames

# this is how we extract faces in ensemble
def detect_faces(frames, mtcnn, batch_size):
    """
    Detect and crop faces using raw bounding boxes to prevent 
    MTCNN from automatically applying its own resizing/normalization.
    """
    faces = []
    
    # Process frames in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        
        # .detect() expects a list of numpy arrays and returns bounding boxes
        batch_boxes, _ = mtcnn.detect(batch_frames)
        
        for frame, boxes in zip(batch_frames, batch_boxes):
            if boxes is None:
                continue
                
            for box in boxes:
                box = [int(b) for b in box]
                
                # Manual numpy crop: frame[y1:y2, x1:x2]
                face = frame[box[1]:box[3], box[0]:box[2]]
                
                # Validate face size (filter out garbage)
                if len(face) == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                    continue
                    
                faces.append(face)

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

    video_path = str(video_path)
    output_path = str(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", "16000",        # 16kHz
        "-f", "wav",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
