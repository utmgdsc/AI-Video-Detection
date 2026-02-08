'''
Generates a CSV file for the predictions on a given directory of videos.
Note that the predictions are made using the video preprocessing provided by the original MesoNet repository.

CSV File labels:
1.0 is the real class
0.0 is the fake class
-1.0 is a generic error
-2.0 is no face were detected

Sample Bash commands:
# Use default path
python3 make-video-predictions.py Meso4_DF.h5 real 1.0
# Use specified path
python3 make-video-predictions.py Meso4_DF.h5 real 1.0 ../../dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_standard_splits/test/
'''
import csv
import sys
import math

import numpy as np
from classifiers import *
from pipeline import *

WEIGHT_DIR = "./weights"
# PATH is the file path from the current directory to the test set, not including class subdirectories
SET_PATH = '../../dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_standard_splits/test/'
# SET_PATH = '../../dataset/FaceForensics++ Dataset/ff-c23/versions/1/FaceForensics++_C23/'

# Indexes of argument from command line
ARG_WEIGHT = 1
ARG_DIR = 2
ARG_CLASS = 3
ARG_NEW_PATH = 4

if len(sys.argv) == 4 or len(sys.argv) == 5:
    weight_name = sys.argv[ARG_WEIGHT]
    print(f"Weight file: {weight_name}")
    set_dir = sys.argv[ARG_DIR]
    print(f"Video directory: {set_dir}")
    act_class = float(sys.argv[ARG_CLASS])
    print(f"Class: {act_class}")
    if len(sys.argv) == 5:
        SET_PATH = sys.argv[ARG_NEW_PATH]
        print(f"New path to video directory: {SET_PATH}")
else:
    print(f"Error: Unexpected arguments.")
    print(f"Expected: weight file, video directory, class of directory (0.0 for fake or 1.0 for real)")
    print(f"Weight must be in {WEIGHT_DIR}.")
    print(f"Directory must be in {SET_PATH} or in specified directory.")
    sys.exit(2)

# Load the model and its pretrained weights
classifier = Meso4()
classifier.load(join(WEIGHT_DIR, weight_name))

# CSV Data formatting
labels = ['actual_class', 'predicted_class', 'score', 'type', 'video_name']
data = [labels]

# Prediction for a video dataset

def CSV_compute_accuracy(classifier, dirname, frame_subsample_count = 30, output = data):
    filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    total = len(filenames)
    for index, vid in enumerate(filenames):
        print(f"{index} / {total} videos predicted")
        try:
            # Compute face locations and store them in the face finder
            face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
            skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
            face_finder.find_faces(resize=0.5, skipstep = skipstep)
            
            gen = FaceBatchGenerator(face_finder)
            p = predict_faces(gen, classifier)
            
            score = np.mean(p > 0.5)
            curr_pred = -1.0

            if (math.isnan(score)): # No faces were found in the video
                curr_pred = -2.0
            elif (score > 0.5):
                curr_pred = 1.0
            else:
                curr_pred = 0.0

            curr_row = [act_class, curr_pred, score, set_dir, vid] # Format of CSV rows
            output.append(curr_row)
        except Exception as error:
            print(f"Error on video {vid}:\n{error}")
            curr_row = [act_class, -1.0, -1.0, set_dir, vid]
            output.append(curr_row)
    print(f"All videos predicted")
    return output

final_data = CSV_compute_accuracy(classifier, join(SET_PATH, set_dir))

print(f"Now writing to CSV")

# Export to CSV
with open(f'predictions/{weight_name}-predictions-{set_dir}.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerows(final_data)

print(f"CSV has been completely written.")
print(f"Script has finished all tasks and ended.")