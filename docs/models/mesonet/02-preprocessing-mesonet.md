# SCRUM-23: Preprocessing pipeline for MesoNet
Note: This file results uses a modified version of the preprocessing pipeline provided by the original MesoNet authors, through pipeline.py
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
1. Select a predefined number of frames from videos
2. Search, align, and extract location of the faces
3. Create a generator with the extracted faces, which will resize and help handle predictions

## Code Changes
- Added `backend/models/MesoNet/activate_conda_env.sh`
- Added `backend/models/MesoNet/make-video-predictions.py`
- Modified `02-source-and-setup.md`

## How to Run
First the Conda environment for this model must be activated. The environment is local to the terminal that is executing the scripts. If using the Anaconda Prompt terminal, or if Conda is automatically intiallized, then the environment can activated using the following:
```bash
# Conda is initialized if the '(base)' appears before the command prompt, for example:
# (base) username@computer:~$

conda activate mesonet
# Note: 'mesonet' is the name of the environment created in the setup guide, and may vary if you named the environmnent something else. You can list your Conda environments with the following line:
conda info --envs
``` 
If you are not using the Anaconda Prompt, then an activation script is provided. This script assumes that the Conda environment has been installed to Miniconda default path for Linux (~/miniconda3/bin/conda), as well as activates the environment if its name is 'mesonet'.
```bash
. activate_conda_env.sh

# In the command prompt, '(base)' should now be replaced with the name of the environment. For example:
# (mesonet) username@computer:~$
```
With the Conda environment activated, the legacy Python 3.6 scripts can now be run.
```bash
# For the original author's provided pipeline, place videos in the 'test_videos' directory and run example.py
# See 02-source-and-setup.md for MesoNet

# The modified pipeline make-video-predictions.py makes use of the same provided pipeline, but creates a CSV file to a preexisting 'predictions' directory on the same level.
# and takes at least 3 arguments: a weight, a directory in the test set, and the class of the test set (between 0 and 1)
# The default path to the test set is to the AIGVDBench standard split test set, 'AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_standard_splits/test/'
python3 -u make-video-predictions.py Meso4_DF.h5 real 1.0

# A fourth argument can be provided to specify a new path to the test set. 
python3 -u make-video-predictions.py Meso4_DF.h5 Deepfakes 0.0 '../../dataset/FaceForensics++ Dataset'
```

## Verification
- [x] Ran preprocessing end-to-end
- [x] Verified split counts match expectations
- [x] Verified processed data looks correct (sample check)
- [x] No dataset files committed
- [x] Documentation updated

## Notes
[Any gotchas, assumptions, or important details]
The provided MesoNet pipeline.py may not be used in the final implementation of this project. This pipeline will likely be replaced with a MCTNN face extractor that unifies the different facial-analyzing models.

The produced CSV file will label the predicted_class of a video as -2.0 if no faces were detected, and -1.0 if an error was encountered while processing the video. The expected outputs are 1.0 for real and 0.0 for fake.

An expected output of make-video-predictions.py may look similar to the following:
```bash
nohup: ignoring input
Using TensorFlow backend.
Weight file: Meso4_DF.h5
Video directory: original
Class: 1.0
2026-02-09 09:34:36.831490: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
0 / 1000 videos predicted
Face extraction warning :  0 - found face in full frame [(256, 1232, 656, 833)]
/home/gdgteam1/miniconda3/envs/mesonet/lib/python3.6/site-packages/scipy/ndimage/interpolation.py:583: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed.
  "the returned array has changed.", UserWarning)
Face extraction report of not_found : 1
Face extraction report of no_face : 0
1 / 1000 videos predicted
Face extraction warning :  0 - found face in full frame [(116, 701, 393, 423)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0
2 / 1000 videos predicted
Face extraction warning :  0 - found face in full frame [(156, 315, 316, 155)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0

...

998 / 1000 videos predicted
Face extraction warning :  0 - found face in full frame [(79, 326, 172, 233)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0
999 / 1000 videos predicted
Face extraction warning :  0 - found face in full frame [(120, 398, 213, 305)]
Face extraction report of not_found : 1
Face extraction report of no_face : 0
All videos predicted
Now writing to CSV
CSV has been completely written.
Script has finished all tasks and ended.
```

The original MesoNet preprocessing was run on the AIGVDBench standard split test set, as well as the FaceForensics++ DeepFake, Face2Face, and original videos. The originally provided Meso4_DF.h5 weight was used for all, and reported a balanced base accuracy of:
52.08% for AIGVDBench
61.65% for DeepFakes
49.75% for Face2Face
(Original weights can be found at: https://github.com/DariusAf/MesoNet/tree/master/test_images/df)

From these tests, MesoNet has been labelling roughly 80% of all AIGVDBench videos as real, and nearly 90% of FaceForensics++.
The AIGVDBench accuracy was not too surprising, as MesoNet was never trained on these kinds of video generation.

The Higher DeepFakes accuracy is slightly lower than expected, as the pretrained weight specialized for DeepFake and had been originally tested FaceForensics++. However, other papers have also reported the large drop in MesoNet's accuracy since 2018, as DeepFake videos become more sophisticated.
(Other paper referenced https://www.sciencedirect.com/science/article/pii/S1877050925013882, DOI: 10.1016/j.procs.2025.04.286)

The most surprising result was for Face2Face, which had a true negative rate of 2.7%, wheras the others had at least 18.55%. The original authors reported that the DeepFake and Face2Face weights were able to perform well on each other's datasets.

For all of the above tests, MesoNet had extracted roughly 30 frames from each video. However, when extracting 10 frames, MesoNet reported nearly identical results for DeepFake and Face2Face, even a slight 0.3% increase in accuracy as more videos were labelled fake, moreso in for fake videos than real.

Using the same MTCNN preprocessing to create AIGVDBench image datasets for training, validation, and testing, a new weight was trained and achieved a validation accuracy of 85.06%. However, with concerns that mixing face extractor methods may affect accuracy and reliability, this new weight has not been tested.

Minor issue: the pipeline provided has encountered a certain error and fail to make a prediction for certain videos. The current most likely cause is the outdated frames extraction count. Back in 2018, the original authors used imageio's reader.get_meta_data()['nframes'] to count the total number of frames in a video. However, the more recent way to do this is to use imageio.v3's improps to count the number of frames. However, the final MesoNet pipeline will likely be a MTCNN implementation instead of the provided pipeline.