# XceptionNet — Source & Setup

## Source location(s)

- GitHub repo link: https://github.com/HongguLiu/Deepfake-Detection
- Paper link: https://arxiv.org/abs/1610.02357
- Any pretrained weights link: https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8
- License notes (if known): Apache-2.0 license

## What I verified

- Does the repo match the paper description? (Yes/No/Partially) Partially
- Pretrained model available? (Yes/No) Yes
- Dataset expected by repo: face images extracted from video dataset

## Environment / dependencies

List exactly what you installed (with versions if possible):

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- PyTorch: 2.7.1+cu118
- CUDA (if used): 11.8
- OpenCV: 4.11.0
- Other deps:
opencv-python
dlib==19.18.0
numpy==1.22.0
pillow>=6.2.2

## Setup steps (copy/paste friendly)

### 1) Clone / install

```bash
python3.10 -m venv xception_env 
source xception_env/bin/activate 
git clone https://github.com/HongguLiu/Deepfake-Detection
cd Deepfake-Detection
python -m pip install -r requirements.txt
```

### 2) Any downloads needed (weights/datasets)

- What to download: You can download the pretrained models from https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8. 
- Where to place it: You can put it in Deepfake-Detection/. 

### 3) How to run

```bash
# To train
python train_CNN.py   -n xception   -tl ./data_list/train.txt   -vl ./data_list/val.txt   -bz 64   -e 20   -mn xception_ffpp_c23.pkl   --continue_train True   -mp ./ffpp_c23.pth

# To test
python test_CNN.py -bz 64 -tl data_list/test.txt -mp output/xception/best.pkl 

# To predict a video
python detect_from_video.py --video_path ./videos/003_000.mp4 --model_path ./pretrained_model/df_c0_best.pkl -o ./output --cuda

# To predict an image
python test_CNN.py -bz 32 --test_list ./data_list/Deepfakes_c0_299.txt --model_path ./pretrained_model/df_c0_best.pkl
```

## Output / results

- What output you saw (logs, metrics, saved files):
# Output of training
(xception_env) gdgteam1@lisa:~/AI-Video-Detection/backend/models/XceptionNet/Deepfake-Detection$ python train_CNN.py   -n xception   -tl ./data_list/train.txt   -vl ./data_list/val.txt   -bz 64   -e 20   -mn xception_ffpp_c23.pkl   --continue_train True   -mp ./ffpp_c23.pth
Using dropout 0.5
Epoch 1/20
----------
epoch train loss: 0.0065 Acc: 0.8341
epoch val loss: 0.0041 Acc: 0.8885
Epoch 2/20
----------
epoch train loss: 0.0029 Acc: 0.9249
epoch val loss: 0.0037 Acc: 0.9033
Epoch 3/20
----------
epoch train loss: 0.0020 Acc: 0.9521
epoch val loss: 0.0032 Acc: 0.9274
Epoch 4/20
----------
epoch train loss: 0.0013 Acc: 0.9672
epoch val loss: 0.0050 Acc: 0.8987
Epoch 5/20
----------
epoch train loss: 0.0010 Acc: 0.9752
epoch val loss: 0.0033 Acc: 0.9315
Epoch 6/20
----------
epoch train loss: 0.0005 Acc: 0.9886
epoch val loss: 0.0033 Acc: 0.9445
Epoch 7/20
----------
epoch train loss: 0.0003 Acc: 0.9920
epoch val loss: 0.0056 Acc: 0.9289
Epoch 8/20
----------
epoch train loss: 0.0004 Acc: 0.9935
epoch val loss: 0.0044 Acc: 0.9412
Epoch 9/20
----------
epoch train loss: 0.0002 Acc: 0.9947
epoch val loss: 0.0041 Acc: 0.9460
Epoch 10/20
----------
epoch train loss: 0.0003 Acc: 0.9948
epoch val loss: 0.0054 Acc: 0.9354
Epoch 11/20
----------
epoch train loss: 0.0001 Acc: 0.9980
epoch val loss: 0.0048 Acc: 0.9462
Epoch 12/20
----------
epoch train loss: 0.0001 Acc: 0.9983
epoch val loss: 0.0057 Acc: 0.9447
Epoch 13/20
----------
epoch train loss: 0.0001 Acc: 0.9984
epoch val loss: 0.0054 Acc: 0.9473
Epoch 14/20
----------
epoch train loss: 0.0001 Acc: 0.9987
epoch val loss: 0.0059 Acc: 0.9478
Epoch 15/20
----------
epoch train loss: 0.0002 Acc: 0.9987
epoch val loss: 0.0053 Acc: 0.9483
Epoch 16/20
----------
epoch train loss: 0.0002 Acc: 0.9993
epoch val loss: 0.0055 Acc: 0.9507
Epoch 17/20
----------
epoch train loss: 0.0000 Acc: 0.9996
epoch val loss: 0.0054 Acc: 0.9506
Epoch 18/20
----------
epoch train loss: 0.0001 Acc: 0.9995
epoch val loss: 0.0053 Acc: 0.9485
Epoch 19/20
----------
epoch train loss: 0.0000 Acc: 0.9997
epoch val loss: 0.0068 Acc: 0.9479
Epoch 20/20
----------
epoch train loss: 0.0000 Acc: 0.9996
epoch val loss: 0.0061 Acc: 0.9492
Best val Acc: 0.9507

# Output testing
(xception_env) gdgteam1@lisa:~/AI-Video-Detection/backend/models/XceptionNet/Deepfake-Detection$ python test_CNN.py -bz 64 -tl data_list/test.txt -mp output/xception/best.pkl 
Using dropout 0.5
Test Acc: 0.9505

- Add screenshots in assets/ if helpful.

## Issues encountered + fixes

- Issue: 
requirements.txt failed to install
- Cause:
Several package versions are outdated and not compatible with python 3.10
- Fix (steps/commands):
Removed outdated packages, installed newer versions of these packages

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?
If you have any question, dm on Discord. 
