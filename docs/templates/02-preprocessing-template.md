# SCRUM-12: Preprocessing pipeline for Efficient Net

## Dataset
- **Name:** AIGVDBench
- **Source:** https://github.com/LongMa-2025/AIGVDBench?tab=readme-ov-file
- **Size:** 300GB, 440k videos
- **License:** cc-by-4.0

## Storage Location
- **Path:** /home/gdgteam1/AI-Video-Detection/backend/dataset/AIGVDBench
- **Access Method:** python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AIGVDBench/AIGVDBench', repo_type='dataset', local_dir='./AIGVDBench')"

## Data Split
- **Train/Val/Test Ratio:** 70% in train set, 15% in validation set, 15% in test set.

- **Counts:**
  - Train: 280000 videos
  - Val: 60000 videos
  - Test: 60000 videos
<!-- excluding close source videos -->

- **Method:** random split with seed 42, Every unique real video and its derivative are sure to be in the same set.

## Preprocessing Steps
1. Initialize MTCNN based on specified input argument
2. Use openCV to split video into individual rgb frames
3. Pass those frames into MTCNN to get every face in each frame

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
# Path should be at: `AI-VIDEO-DETECTION/backend/dataset/AIGVDBench/AIGVDBench/`

# Step 2: Split the dataset
at /AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench$
python3 split_videos_standard_split
or
python3 split_videos_left_out_easyAnimate.py

# Step 3: Run preprocessing
at ~/AI-Video-Detection/backend/models/DeepFake-EfficientNet$
./extract_face_efficientNet.sh

## Verification
- [x] Ran preprocessing end-to-end
- [x] Verified split counts match expectations
- [x] Verified processed data looks correct (sample check)
- [x] No dataset files committed
- [x] Documentation updated

## Notes
[Any gotchas, assumptions, or important details]
To run the script on a video:
python3 -m backend.main --input-dir backend/dataset/AIGVDBench/AIGVDBench/Real/videos/_6E6r_nfgMU_40_113to214.mp4 --mode video 

Checkout main.py for all available argument for main.py

you should see: 
Initializing MTCNN...
FRAMES EXTRACTED!!!
FACES DETECTED!!!
BYE BYE!!!

the preprocessing done in main.py is different than what is done in EfficientNet repo. 

In main.py, considering we only run this on one video instead of 440k videos, the type of mtcnn used is different. The mtcnn used in the efficientNet repo only allow outputting faces to a directory, but the mtcnn used in the main.py allow us to store faces in a variable.

In the efficientNet repo, I created extract_face_efficientNet.sh for AIGVDBench 
dataset, and this script call extract_faces.py in DeepFake-EfficientNet/scripts/ to process videos in AIGVDBench dataset.