# MesoNet — Source & Setup

## Source location(s)

- GitHub repo link: https://github.com/DariusAf/MesoNet
- Paper link: https://arxiv.org/abs/1809.00888
- Any pretrained weights link: https://github.com/DariusAf/MesoNet/tree/master/weights
- License notes (if known): https://github.com/DariusAf/MesoNet/blob/master/LICENSE

## What I verified

- Does the repo match the paper description? Partially
- Pretrained model available? Yes
- Dataset expected by repo: The original link on the MesoNet GitHub no longer works, but the following link claims to use their dataset: https://www.kaggle.com/datasets/iamshahzaibkhan/deepfake-database

## Environment / dependencies

List exactly what you installed (with versions if possible):

- OS: Linux
- Python: 3.6.13
- PyTorch: Unused. Alternatively used TensorFlow and Keras
- CUDA (if used):
- OpenCV: 4.1.2.30
- Other deps:
    * Imageio 2.4.1
    * FFMPEG 0.5.1
    * face_recognition 1.3.0
    * CMake 3.28.4
    * DLib 19.22.0
    * H5py 2.10.0

## Setup steps (copy/paste friendly)
Due to the legacy software MesoNet relies on, Conda 25.7.0 was used to manage the virtual environment.

### 1) Clone / install
Using the Anaconda Prompt
```bash
conda create -n mesonet python=3.6 -y
# -y flag is used at the end to agree to all installation prompts for base packages

conda activate mesonet

# It is best to copy and paste one at a time, as you may need to agree to some prompts
conda install keras=2.1.5
conda install numpy=1.14.2
pip install tensorflow==1.8.0
pip install opencv-python==4.1.2.30
pip install cmake==3.28.4
pip install imageio==2.4.1 imageio-ffmpeg==0.5.1
pip install dlib==19.22.0
conda install h5py=2.10.0
pip install face-recognition
conda install ffmpeg=4.3.1 -c conda-forge
```
Note: imports and some parameter names were modified in this repository to support newer libraries

### 2) Any downloads needed (weights/datasets)

- What to download:
    * Pretrained weights are provided on the MesoNet GitHub repository
- Where to place it:
    * Testing Images: Place images (Ideally 256x256) into test_images, sorted into df or real.
    * Testing Videos: Place videos into the test_videos directory.

### 3) How to run
Ensure the virtual environment is activated (The installation guide activates the venv near the start).
```bash
# You can verify that the venv is activated by checking the Python version
python --version
# Expecting output similar to Python 3.6
# Alternatively, if the environment directory 'mesonet' was installed in the directory '~/miniconda3/bin/conda',
# running the activation script provided will activate it
. activate_conda_env.sh

python example.py
# Runs the example provided by the original developers
```

## Output / results

- What output you saw (logs, metrics, saved files):
- Add screenshots in assets/ if helpful.

Running the provided example.py program provided by the MesoNet developers (with some modifications) produced this output.
Additional test videos were also added.
```bash
(mesonet) example-user\using\PyCharm\terminal> python example.py
Using TensorFlow backend.
2026-01-25 23:40:51.018077: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Found 4 images belonging to 2 classes.
Predicted : [[0.0486937]] 
Real class : [0.]
Dealing with video  fake1.mp4
Face extraction warning :  0 - found face in full frame [(547, 516, 825, 238)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0
Predicting  fake1.mp4
Dealing with video  real1.mp4
WARNING:root:Warning: the frame size for reading (720, 1280) is different from the source frame size (1280, 720).
Face extraction warning :  0 - found face in full frame [(547, 516, 825, 238)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0
Predicting  real1.mp4
`fake1` video class prediction : 0.9310344827586207
`real1` video class prediction : 1.0
```
Notes:
* 4 images were provided and found by the the test_images directory. However, only one was selected and predicted. To have more images predicted, within example.py step 2, change the batch_size argument to the desired number of images to predict and produce an output similar to the followingZ:
```bash
Predicted : [[0.9897475 ]
 [0.0486937 ]
 [0.9977016 ]
 [0.04141179]] 
Real class : [1. 0. 1. 0.]
```
* If test_images is empty of any images then example.py will not run. Steps 2 and 3 must be removed in order to predict videos.
* The fake video received a score of 0.93, but was expected to be lower than 0.5. Possible causes are higher version incompatibility, or newer DeepFake videos are too sophisticated for older MesoNet models.

## Issues encountered + fixes

- Issue: Imports and some parameter names were modified
- Cause: This project uses later versions of software that may still support MesoNet. Parameter names may have changed.
- Fix (steps/commands):
    * Import Statements: Remove the inital tensorflow
    * Different Parameters in classifiers.py:
        * For calls to LeakyReLU(), replace negative_slope with alpha (two occurances)
        * For calls to Adam(), replace learning_rate with lr (three occurances)

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?

It may be best to try installing dependencies as close to the original development than try to find the most recent versions that can still support it. Additionally, the model may be too outdated for present-day DeepFake-edited videos.