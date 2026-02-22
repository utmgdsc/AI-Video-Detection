#!/usr/bin/env python3
"""
Face Extraction Script using MTCNN
Extracts faces from videos and images for deepfake detection.
"""

import argparse
import os
import glob
import time
import cv2
import torch
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import logging

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Error: facenet-pytorch not installed. Install with: pip install facenet-pytorch")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FastMTCNN:
    """
    Fast MTCNN implementation for efficient face detection.

    Processes frames with stride to balance speed and accuracy.
    """

    def __init__(
        self,
        stride: int = 1,
        resize: float = 1.0,
        margin: int = 50,
        min_face_size: int = 100,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.7,
        post_process: bool = True,
        select_largest: bool = True,
        keep_all: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize FastMTCNN.

        Args:
            stride: Detection stride (detect every N frames)
            resize: Frame resize factor
            margin: Margin around detected face
            min_face_size: Minimum face size to detect
            thresholds: MTCNN detection thresholds
            factor: MTCNN scale factor
            post_process: Whether to post-process detections
            select_largest: Select only largest face
            keep_all: Keep all detected faces
            device: Device to run on ('cuda' or 'cpu')
        """
        self.stride = stride
        self.resize = resize

        self.mtcnn = MTCNN(
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            select_largest=select_largest,
            keep_all=keep_all,
            device=device
        )

        logger.info(f"FastMTCNN initialized on {device}")

    def __call__(self, frames: List, output_dir: str, prefix: str = "face") -> int:
        """
        Extract faces from frames.

        Args:
            frames: List of frames (numpy arrays)
            output_dir: Output directory for extracted faces
            prefix: Prefix for saved images

        Returns:
            Number of faces extracted
        """
        if self.resize != 1.0:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        # Detect faces
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces_count = 0
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)

            if boxes[box_ind] is None:
                continue

            for box in boxes[box_ind]:
                box = [int(b) for b in box]

                # Extract face region
                face = frame[box[1]:box[3], box[0]:box[2]]

                # Validate face
                if len(face) == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                    continue

                # Save face
                timestamp = time.time()
                filename = f"{prefix}-{timestamp:.6f}.jpg"
                filepath = os.path.join(output_dir, filename)

                # Convert RGB to BGR for cv2
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, face_bgr)

                faces_count += 1

        return faces_count


def process_video(
    video_path: str,
    output_dir: str,
    mtcnn: FastMTCNN,
    batch_size: int = 60,
    frame_skip: int = 30
) -> Tuple[int, int]:
    """
    Process a single video file.

    Args:
        video_path: Path to video file
        output_dir: Output directory
        mtcnn: FastMTCNN instance
        batch_size: Number of frames to process at once
        frame_skip: Process every Nth frame

    Returns:
        Tuple of (frames_processed, faces_detected)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frames_processed = 0
    faces_detected = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_skip == 0 or frame_idx == total_frames - 1:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if len(frames) >= batch_size or frame_idx == total_frames - 1:
                # Process batch
                video_name = Path(video_path).stem
                faces = mtcnn(frames, output_dir, prefix=video_name)

                frames_processed += len(frames)
                faces_detected += faces

                frames = []

    cap.release()
    return frames_processed, faces_detected


def process_image(
    image_path: str,
    output_dir: str,
    mtcnn: FastMTCNN
) -> Tuple[int, int]:
    """
    Process a single image file.

    Args:
        image_path: Path to image file
        output_dir: Output directory
        mtcnn: FastMTCNN instance

    Returns:
        Tuple of (images_processed, faces_detected)
    """
    image = cv2.imread(image_path)

    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        return 0, 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_name = Path(image_path).stem
    faces = mtcnn([image_rgb], output_dir, prefix=image_name)

    return 1, faces


def main():
    parser = argparse.ArgumentParser(
        description='Extract faces from videos and images using MTCNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing videos/images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for extracted faces')

    # Processing options
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default='video',
                        help='Processing mode')
    parser.add_argument('--batch-size', type=int, default=60,
                        help='Batch size for processing')
    parser.add_argument('--frame-skip', type=int, default=30,
                        help='Process every Nth frame (video only)')

    # MTCNN parameters
    parser.add_argument('--stride', type=int, default=1,
                        help='Detection stride')
    parser.add_argument('--margin', type=int, default=50,
                        help='Margin around detected face')
    parser.add_argument('--min-face-size', type=int, default=100,
                        help='Minimum face size to detect')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    # Initialize MTCNN
    logger.info("Initializing MTCNN...")
    mtcnn = FastMTCNN(
        stride=args.stride,
        margin=args.margin,
        min_face_size=args.min_face_size,
        device=device
    )

    # Get input files
    if args.mode == 'video':
        patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    else:
        patterns = ['*.jpg', '*.jpeg', '*.png']

    input_files = []
    for pattern in patterns:
        input_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

    if not input_files:
        logger.error(f"No files found in {args.input_dir}")
        return

    logger.info(f"Found {len(input_files)} files to process")

    # Process files
    total_frames = 0
    total_faces = 0

    for filepath in tqdm(input_files, desc='Processing files'):
        try:
            if args.mode == 'video':
                frames, faces = process_video(
                    filepath,
                    args.output_dir,
                    mtcnn,
                    args.batch_size,
                    args.frame_skip
                )
            else:
                frames, faces = process_image(
                    filepath,
                    args.output_dir,
                    mtcnn
                )

            total_frames += frames
            total_faces += faces

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue

    logger.info(f"Processing complete!")
    logger.info(f"Total frames processed: {total_frames}")
    logger.info(f"Total faces detected: {total_faces}")
    logger.info(f"Faces saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
