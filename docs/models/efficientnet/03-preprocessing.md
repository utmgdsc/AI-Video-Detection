# EfficientNet — Preprocessing

## Dataset

- Name: AIGVDBench
- Source: https://github.com/LongMa-2025/AIGVDBench?tab=readme-ov-file
- Size: 300GB, 440k videos
- License: cc-by-4.0

## Storage location

- Path: /home/gdgteam1/AI-Video-Detection/backend/dataset/AIGVDBench
- Access command:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AIGVDBench/AIGVDBench', repo_type='dataset', local_dir='./AIGVDBench')"
```

## Data split

- Train/Val/Test ratio: 70/15/15
- Counts:
  - Train: 280000 videos
  - Val: 60000 videos
  - Test: 60000 videos
- Method: Random split with seed 42; each unique real video and its derivatives remain in the same split.

## Preprocessing steps

1. Initialize MTCNN based on input arguments.
2. Use OpenCV to split video into RGB frames.
3. Pass frames into MTCNN to detect faces.

## Code changes made for this preprocessing work

- Updated `backend/main.py`
- Updated `backend/handlers/video_handler.py`
- Updated `backend/preprocessing/video_processor.py`
- Added `backend/models/DeepFake-EfficientNet/extract_face_efficientNet.sh`
- Added `backend/models/DeepFake-EfficientNet/scripts/test.py`
- Modified `backend/models/DeepFake-EfficientNet/scripts/train.py`
- Added `backend/dataset/AIGVDBench/AIGVDBench/`
- Added `backend/dataset/AIGVDBench/AIGVDBench/split_videos_left_out_easyAnimate.py`
- Added `backend/dataset/AIGVDBench/AIGVDBench/split_videos_standard_split.py`

## How to run

```bash
# Step 1: Download/locate dataset
# Expected path: AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench/

# Step 2: Split the dataset
cd backend/dataset/AIGVDBench/AIGVDBench
python3 split_videos_standard_split
# or
python3 split_videos_left_out_easyAnimate.py

# Step 3: Run preprocessing for EfficientNet
cd backend/models/DeepFake-EfficientNet
./extract_face_efficientNet.sh
```

## Verification completed

- Ran preprocessing end-to-end
- Verified split counts match expectations
- Verified processed data looks correct (sample check)
- Confirmed no dataset files were committed
- Updated documentation

## Notes

- To run preprocessing on a single video via `main.py`:

```bash
python3 -m backend.main --input-dir backend/dataset/AIGVDBench/AIGVDBench/Real/videos/_6E6r_nfgMU_40_113to214.mp4 --mode video
```

- Expected output:
  - Initializing MTCNN...
  - FRAMES EXTRACTED!!!
  - FACES DETECTED!!!
  - BYE BYE!!!
- The preprocessing in `backend/main.py` differs from the EfficientNet repo pipeline.
- `backend/main.py` uses an MTCNN mode suitable for single-video processing and in-memory face handling.
- The EfficientNet repo pipeline is designed for large-scale extraction and writes faces to directories.
