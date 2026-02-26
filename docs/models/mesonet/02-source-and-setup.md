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
- Deps for ensemble integration (installed in the current environment, Python 3.6):
    * fastapi 0.63.0
    * uvicorn 0.13.4
- Ensemble deps (installed on the ensemble environment, Python 3.10)
    * requests 2.25.1

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

If using MesoNet through the ensemble, install the following:
```bash
# On the mesonet environment
pip install fastapi==0.63.0 uvicorn==0.13.4
```
```bash
# On the ensemble environment
pip install requests 2.25.1
```

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

If running through the ensemble, then on the ensemble environment:
```bash
# Activate the ensemble virtual environment, if not already activated
source venv/bin/activate
# (venv) should appear before the prompt: "(venv) user/.../AI-Video-Detection$"
# Then in the repository directory, running the command will make a prediction on the given video
python3 -m backend.main --input-dir "./backend/dataset/FaceForensics++/original/002.mp4"

# Alternatively, in the directory "/AI-Video-Detection/backend/models/MesoNet", we can run a test script for the ensemble connection
python3 mesonet_interface.py
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

If running through the ensemble, then server logs will be appended to the end of a text file located at "/AI-Video-Detection/backend/models/MesoNet/logs/meso_server.txt". A sample output for the meso_server.txt is provided for the example command:
`(venv) user:~/AI-Video-Detection$ python3 -m backend.main --input-dir "./backend/dataset/FaceForensics++/original/002.mp4"`
```bash
Using TensorFlow backend.
INFO:     Started server process [2793898]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
SERVER 0 =====: Writing test message to log (from mesonet_interface.py test_server(), expected in logs/meso_server.txt)
INFO:     127.0.0.1:52144 - "GET /test_server HTTP/1.1" 200 OK
SERVER 1 =====: Clearing previous model session (no affect if no models were loaded before).
SERVER 2 =====: Selecting architecture: 'Meso4'
2026-02-26 14:21:25.159637: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
SERVER 3 =====: Loading weight on path: 'weights/Meso4_custom_weight1_epoch7.h5'
SERVER 4 =====: MODEL SUCCESSFULLY LOADED.
INFO:     127.0.0.1:52156 - "POST /load_model HTTP/1.1" 200 OK
SERVER 5 =====: Loading images from faces file: 'temp/faces.npy'
SERVER 6 =====: Normalizing images, if not already normalized.
SERVER 7 =====: BEGIN MAKING PREDICTIONS...
SERVER 8 =====: PREDICTIONS MADE, RETURNING RESULTS AS:
SERVER 9 =====: [[1.0], [1.0], [0.9999998807907104], [1.0], [1.0], [0.9999985694885254], [1.0], [0.9999997615814209], [1.0], [0.9999998807907104], [0.9999998807907104], [1.0], [1.0], [0.9999998807907104], [1.0], [1.0]]
INFO:     127.0.0.1:52166 - "POST /process HTTP/1.1" 200 OK
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [2793898]

```
In the ensemble environment, the following is expected (assuming only MesoNet is connected)
```bash
2026-02-26 14:21:23,077 - INFO - Initializing MTCNN...
2026-02-26 14:21:23,734 - INFO - frame extracted
2026-02-26 14:21:24,133 - INFO - faces detected
DEBUG 0 =====: Initializing new MesoNet Client
DEBUG 1 =====: Checking server is running...
DEBUG 2 =====: Sending test POST
Starting MesoNet server...
DEBUG 3 =====: Trying to open server log
DEBUG 4 =====: Trying to run server
DEBUG 5 =====: Server started!
DEBUG 6 =====: Waiting until ready
DEBUG 7 =====: Testing connection...
DEBUG 8 =====: Testing connection...
DEBUG 9 =====: Testing connection...
DEBUG 10 =====: Server ready.
DEBUG 11 =====: Asking server to load model...
DEBUG 12 =====: Load status: 200
DEBUG 13 =====: Load text: {"success":true}
DEBUG 14 =====: Model loaded successfully.
DEBUG 15 =====: Process status: 200
DEBUG 16 =====: Process text: {"success":true,"predictions":[[1.0],[1.0],[0.9999998807907104],[1.0],[1.0],[0.9999985694885254],[1.0],[0.9999997615814209],[1.0],[0.9999998807907104],[0.9999998807907104],[1.0],[1.0],[0.9999998807907104],[1.0],[1.0]]}
```

## Issues encountered + fixes

- Issue: Imports and some parameter names were modified
- Cause: This project uses later versions of software that may still support MesoNet. Parameter names may have changed.
- Fix (steps/commands):
    * Import Statements: Remove the inital tensorflow
    * Different Parameters in classifiers.py:
        * For calls to LeakyReLU(), replace negative_slope with alpha (two occurances)
        * For calls to Adam(), replace learning_rate with lr (three occurances)

- Issue: In certain situations, the ensemble will fail due to a pre-existing running server.
- Cause: Another program is using the port, or the MesoNet server was started but never stopped.
- Fix (steps/commands):
    * Identify the server using `ps aux | grep uvicorn`
    * Kill the process with `kill [pid of the server]`

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?

It may be best to try installing dependencies as close to the original development than try to find the most recent versions that can still support it. Additionally, the model may be too outdated for present-day DeepFake-edited videos.
