# SCRUM-12: Preprocessing pipeline for Efficient Net

## Dataset
- **Name:** ASVspoof 2019 — Logical Access (LA
- **Source:** https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
- **Size:** 7.5 GB, 12k videos
- **License:** cc-by-4.0

## Storage Location
- **Path:** /home/gdgteam1/AI-Video-Detection/backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/
- **Access Method:** Dataset is downloaded manually from Kaggle and unzipped into backend/dataset/

## Data Split
- **Train/Val/Test Ratio:** Official ASVspoof 2019 split (deterministic, not percentage-based). 20/20/60 for Train/Val/Test. 

- **Counts:**
  - Train: 25,381 videos
  - Val: 24,987 videos
  - Test: 71,934  videos
<!-- excluding close source videos -->

- **Method:** Predefined dataset split provided by ASVspoof2019. No random sampling or seed is used. Labels (bonafide vs spoof) are obtained from CM protocol files.

## Preprocessing Steps
1. Load raw .flac audio files from ASVspoof2019 LA directories
2. Ensure consistent sampling rate
3. Trim or pad audio to a fixed length required by AASIST
4. Assign bonafide/spoof labels using CM protocol files
5. Output preprocessed data in a format compatible with AASIST training

## Code Changes
- Added `backend/models/AASIST/preprocessing/audionorm.py`
- Added `backend/models/AASIST/preprocessing/inspect_asvspoof2019_la.py`
- Added `backend/models/AASIST/scripts/make_baseline.py`
- Added `backend/models/AASIST/scripts/plot_accuracy.py`
- Added `backend/models/AASIST/train/train.py`
- Updated `docs/models/aasist/02-preprocessing-checklist.md`
- Updated `docs/models/aasist/02-preprocessing-template.md`
- Updated `docs/models/aasist/02-source-and-setup.md`

## How to Run
```bash
# Step 1: Ensure dataset is placed at:
# backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/`

# Step 2: Verify dataset counts
find dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_train -type f | wc -l
find dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_dev   -type f | wc -l
find dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_eval  -type f | wc -l

# Step 3: Run training (preprocessing happens internally)
python -m backend.models.AASIST.train.train_aasist_baseline \
  --train_wav_dir backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
  --train_protocol backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_wav_dir backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac \
  --dev_protocol backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --config_path backend/models/AASIST/aasist_detector/config/AASIST.conf


## Verification
- [x] Ran preprocessing end-to-end
- [x] Verified split counts match expectations
- [x] Verified processed data looks correct (sample check)
- [x] No dataset files committed
- [x] Documentation updated

## Notes
- Preprocessing is performed during training rather than as a separate preprocessing step that writes files to disk.
- Audio normalization and input handling are implemented in models/AASIST/preprocessing/audionorm.py.
- Dataset integrity and split correctness were verified using inspect_asvspoof2019_la.py and manual file counts.
- The training script must be run using python -m ... (or with PYTHONPATH set) so that imports under the backend package resolve correctly.
- Audio files are expected to be 16 kHz .flac files, consistent with ASVspoof2019 LA specifications.