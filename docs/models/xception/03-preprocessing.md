# SCRUM-XX: Preprocessing Pipeline for XceptionNet

## Dataset
- Name: AIGVDBench
- Source: https://github.com/LongMa-2025/AIGVDBench?tab=readme-ov-file
- Size: 300GB, 440k videos
- License: cc-by-4.0

## Storage Location
- Path: /home/gdgteam1/AI-Video-Detection/backend/dataset/AIGVDBench
- Access command:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AIGVDBench/AIGVDBench', repo_type='dataset', local_dir='./AIGVDBench')"
```

## Data Split
- Train/Val/Test ratio: 70/15/15
- Counts:
  - Train: 280000 videos
  - Val: 60000 videos
  - Test: 60000 videos
- Method: Random split with seed 42; each unique real video and its derivatives remain in the same split.

## Preprocessing Steps
1. Initialize MTCNN based on input arguments.
2. Use OpenCV to split video into RGB frames.
3. Pass frames into MTCNN to detect faces.
4. Generate a data list containing tagged face images. 

## Code Changes
- Updated `backend/main.py`
- Updated `backend/handlers/video_handler.py`
- Updated `backend/preprocessing/video_processor.py`
- Added `backend/models/DeepFake-EfficientNet/extract_face_efficientNet.sh`
- Added `backend/models/DeepFake-EfficientNet/scripts/test.py`
- Modified `backend/models/DeepFake-EfficientNet/scripts/train.py`
- Added `backend/dataset/AIGVDBench/AIGVDBench/`
- Added `backend/dataset/AIGVDBench/AIGVDBench/split_videos_left_out_easyAnimate.py`
- Added `backend/dataset/AIGVDBench/AIGVDBench/split_videos_standard_split.py`

## How to Run
```bash
# Step 1: Download/locate dataset
# Expected path: AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench/

# Step 2: Split the dataset
cd backend/dataset/AIGVDBench/AIGVDBench
python3 split_videos_standard_split
# or
python3 split_videos_left_out_easyAnimate.py

# Step 3: Run preprocessing
cd backend/models/DeepFake-EfficientNet
./extract_face_efficientNet.sh

# Step 4: Generate data list
cd backend/models/XceptionNet/datalist.sh
./datalist.sh
```

## Verification
- [X] Ran preprocessing end-to-end
- [X] Verified split counts match expectations
- [X] Verified processed data looks correct (sample check)
- [X] No dataset files committed
- [X] Documentation updated

## Notes
Note: The preprocessing pipeline of XceptionNet is basically same as the one of EfficientNet. To avoid repeating the work, XceptionNet uses the same preprocessing pipeline as EfficientNet. 

Results: 
For AIGVDBench dataset, XceptionNet got the following: 
	epoch train loss: 0.0000 Acc: 0.9996
	epoch val loss: 0.0061 Acc: 0.9492
	Best val Acc: 0.9507
	Test Acc: 0.9505
For FaceForensics++ dataset, XceptionNet got the following:
	epoch train loss: 0.0000 Acc: 0.9992
	epoch val loss: 0.0104 Acc: 0.9349
	Best val Acc: 0.9364
	Test Acc: 0.9016
