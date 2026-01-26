# XceptionNet — Source & Setup

## Source location(s)

- GitHub repo link: https://github.com/i3p9/deepfake-detection-with-xception
- Paper link: https://arxiv.org/abs/1610.02357
- Any pretrained weights link: Keras ImageNet pre-trained Xception weights downloaded automatically when running train_dateset.py, via keras.applications.xception. 
- License notes (if known):

## What I verified

- Does the repo match the paper description? (Yes/No/Partially) Partially
- Pretrained model available? (Yes/No) Yes
- Dataset expected by repo: face images extracted from videos in the train_sample_videos folder (from https://www.kaggle.com/c/deepfake-detection-challenge/data)

## Environment / dependencies

List exactly what you installed (with versions if possible):

- OS: macOS Tahoe 26.2
- Python: 3.11.14
- PyTorch: Not used, Tensorflow is used
- CUDA (if used):
- OpenCV: 4.11.0
- Other deps:
tensorflow>=2.12,<2.16
numpy
h5py
pillow
opencv-python
mtcnn
tqdm
matplotlib
scikit-learn

## Setup steps (copy/paste friendly)

### 1) Clone / install

```bash
python3.11 -m venv xception_env 
source xception_env/bin/activate 
git clone https://github.com/i3p9/deepfake-detection-with-xception.git 
cd deepfake-detection-with-xception 
pip install -r requirements.txt 
```

### 2) Any downloads needed (weights/datasets)

- What to download: You can download the dataset from from https://www.kaggle.com/c/deepfake-detection-challenge/data, and from the train_sample_videos folder, extract faces from those videos. 
- Where to place it: Place real faces in dataset/real, and fake faces in dataset/fake. 

### 3) How to run

```bash
# To train
python train_dateset.py dataset/ classes.txt result/

# To predict an image
python image_prediction.py result/model_fine_final.h5 classes.txt input.jpg
```

## Output / results

- What output you saw (logs, metrics, saved files):
# Output of training
Training on 1 images and labels
Validation on 1 images and labels
  hist_pre = model.fit_generator(
Epoch 1/10
1/1 [==============================] - ETA: 0s - loss: 0.8182 - accuracy: 0.0000e+00
  saving_api.save_model(
1/1 [==============================] - 1s 828ms/step - loss: 0.8182 - accuracy: 0.0000e+00 - val_loss: 3.8684 - val_accuracy: 0.0000e+00
Epoch 2/10
1/1 [==============================] - 0s 124ms/step - loss: 0.0094 - accuracy: 1.0000 - val_loss: 6.1383 - val_accuracy: 0.0000e+00
Epoch 3/10
1/1 [==============================] - 0s 128ms/step - loss: 5.9122e-04 - accuracy: 1.0000 - val_loss: 7.7451 - val_accuracy: 0.0000e+00
Epoch 4/10
1/1 [==============================] - 0s 126ms/step - loss: 8.6065e-05 - accuracy: 1.0000 - val_loss: 9.0243 - val_accuracy: 0.0000e+00
Epoch 5/10
1/1 [==============================] - 0s 124ms/step - loss: 1.9073e-05 - accuracy: 1.0000 - val_loss: 10.0969 - val_accuracy: 0.0000e+00
Epoch 6/10
1/1 [==============================] - 0s 122ms/step - loss: 5.3644e-06 - accuracy: 1.0000 - val_loss: 11.0025 - val_accuracy: 0.0000e+00
Epoch 7/10
1/1 [==============================] - 0s 125ms/step - loss: 1.7881e-06 - accuracy: 1.0000 - val_loss: 11.7932 - val_accuracy: 0.0000e+00
Epoch 8/10
1/1 [==============================] - 0s 129ms/step - loss: 7.1526e-07 - accuracy: 1.0000 - val_loss: 12.4952 - val_accuracy: 0.0000e+00
Epoch 9/10
1/1 [==============================] - 0s 126ms/step - loss: 2.3842e-07 - accuracy: 1.0000 - val_loss: 13.1267 - val_accuracy: 0.0000e+00
Epoch 10/10
1/1 [==============================] - 0s 126ms/step - loss: 1.1921e-07 - accuracy: 1.0000 - val_loss: 13.6968 - val_accuracy: 0.0000e+00
  hist_fine = model.fit_generator(
Epoch 1/30
1/1 [==============================] - 3s 3s/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 17.3240 - val_accuracy: 0.0000e+00
Epoch 2/30
1/1 [==============================] - 0s 320ms/step - loss: 0.0041 - accuracy: 1.0000 - val_loss: 22.5205 - val_accuracy: 0.0000e+00
Epoch 3/30
1/1 [==============================] - 0s 317ms/step - loss: 0.0159 - accuracy: 1.0000 - val_loss: 19.1766 - val_accuracy: 0.0000e+00
Epoch 4/30
1/1 [==============================] - 0s 310ms/step - loss: 2.5510e-05 - accuracy: 1.0000 - val_loss: 22.7549 - val_accuracy: 0.0000e+00
Epoch 5/30
1/1 [==============================] - 0s 319ms/step - loss: 3.5524e-05 - accuracy: 1.0000 - val_loss: 25.2813 - val_accuracy: 0.0000e+00
Epoch 6/30
1/1 [==============================] - 0s 323ms/step - loss: 5.6028e-06 - accuracy: 1.0000 - val_loss: 27.9448 - val_accuracy: 0.0000e+00
Epoch 7/30
1/1 [==============================] - 0s 316ms/step - loss: 5.9605e-07 - accuracy: 1.0000 - val_loss: 30.2487 - val_accuracy: 0.0000e+00
Epoch 8/30
1/1 [==============================] - 0s 313ms/step - loss: 1.1921e-07 - accuracy: 1.0000 - val_loss: 33.0067 - val_accuracy: 0.0000e+00
Epoch 9/30
1/1 [==============================] - 0s 308ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 36.1712 - val_accuracy: 0.0000e+00
Epoch 10/30
1/1 [==============================] - 0s 311ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 39.5334 - val_accuracy: 0.0000e+00
Epoch 11/30
1/1 [==============================] - 0s 321ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 42.9692 - val_accuracy: 0.0000e+00
Epoch 12/30
1/1 [==============================] - 0s 316ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 46.0348 - val_accuracy: 0.0000e+00
Epoch 13/30
1/1 [==============================] - 0s 312ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 49.1165 - val_accuracy: 0.0000e+00
Epoch 14/30
1/1 [==============================] - 0s 312ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 52.1594 - val_accuracy: 0.0000e+00
Epoch 15/30
1/1 [==============================] - 0s 320ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 54.9095 - val_accuracy: 0.0000e+00
Epoch 16/30
1/1 [==============================] - 0s 317ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 57.3355 - val_accuracy: 0.0000e+00
Epoch 17/30
1/1 [==============================] - 0s 313ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 59.3735 - val_accuracy: 0.0000e+00
Epoch 18/30
1/1 [==============================] - 0s 329ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 61.0729 - val_accuracy: 0.0000e+00
Epoch 19/30
1/1 [==============================] - 0s 310ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 62.5793 - val_accuracy: 0.0000e+00
Epoch 20/30
1/1 [==============================] - 0s 311ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 63.9054 - val_accuracy: 0.0000e+00
Epoch 21/30
1/1 [==============================] - 0s 319ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 65.0355 - val_accuracy: 0.0000e+00
Epoch 22/30
1/1 [==============================] - 0s 319ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 65.8778 - val_accuracy: 0.0000e+00
Epoch 23/30
1/1 [==============================] - 0s 312ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 66.5389 - val_accuracy: 0.0000e+00
Epoch 24/30
1/1 [==============================] - 0s 318ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.0362 - val_accuracy: 0.0000e+00
Epoch 25/30
1/1 [==============================] - 0s 311ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.3468 - val_accuracy: 0.0000e+00
Epoch 26/30
1/1 [==============================] - 0s 314ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.4494 - val_accuracy: 0.0000e+00
Epoch 27/30
1/1 [==============================] - 0s 312ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.4444 - val_accuracy: 0.0000e+00
Epoch 28/30
1/1 [==============================] - 0s 330ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.3706 - val_accuracy: 0.0000e+00
Epoch 29/30
1/1 [==============================] - 0s 315ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.2196 - val_accuracy: 0.0000e+00
Epoch 30/30
1/1 [==============================] - 0s 319ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 66.9933 - val_accuracy: 0.0000e+00

# Output of image prediction
1/1 [==============================] - 0s 184ms/step
Top 1 =
Class: real
Probability: 98.86%
Top 2 =
Class: fake
Probability: 1.14%

- Add screenshots in assets/ if helpful.

## Issues encountered + fixes

- Issue: 
requirements.txt failed to install
- Cause:
Several package versions are outdated and not compatible with python 3.11
- Fix (steps/commands):
Removed outdated packages, installed newer versions of these packages

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?
If classes.txt is placed under dataset/, do this: mv dataset/classes.txt . 
