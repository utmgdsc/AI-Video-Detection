# SCRUM-XX: Preprocessing pipeline for [Model Name]

## Dataset
- **Name:**
- **Source:**
- **Size:** [e.g., 500 videos, 150 GB]
- **License:**

## Storage Location
- **Path:**
- **Access Method:** [e.g., local disk, AWS S3, download from X]

## Data Split
- **Train/Val/Test Ratio:** [e.g., 70/15/15]
- **Counts:**
  - Train: [X videos]
  - Val: [X videos]
  - Test: [X videos]
- **Method:** [e.g., random split with seed 42, stratified by class]

## Preprocessing Steps
1. [Step 1 - e.g., Extract faces from video frames]
2. [Step 2 - e.g., Resize to 224x224]
3. [Step 3 - e.g., Normalize pixel values]

## Code Changes
- Added `backend/models/[MODEL]/preprocessing/preprocess.py`
- Added `backend/models/[MODEL]/preprocessing/split_dataset.py`
- Updated `backend/models/[MODEL]/requirements.txt`

## How to Run
```bash
# Step 1: Download/locate dataset
# Path should be at: [storage path]

# Step 2: Split the dataset
python backend/models/[MODEL]/preprocessing/split_dataset.py \
  --input [dataset_path] \
  --output [split_output_path] \
  --train-ratio 0.7 --val-ratio 0.15

# Step 3: Run preprocessing
python backend/models/[MODEL]/preprocessing/preprocess.py \
  --input [split_output_path]/train \
  --output [processed_output_path]
```

## Verification
- [x] Ran preprocessing end-to-end
- [x] Verified split counts match expectations
- [x] Verified processed data looks correct (sample check)
- [x] No dataset files committed
- [x] Documentation updated

## Notes
[Any gotchas, assumptions, or important details]
