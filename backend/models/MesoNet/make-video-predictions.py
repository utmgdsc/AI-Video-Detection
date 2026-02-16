import csv
import argparse
from pathlib import Path
import traceback
from os import listdir
from os.path import isfile, join
import math
from math import floor 

import numpy as np
from classifiers import *
from pipeline import *

def get_cmd_args():
    PARSER_DESC = "Predict whether videos of the same class are real or fake. Uses " \
        "the original face-extractor pipeline provided by the MesoNet authors."
    parser = argparse.ArgumentParser(
        description=PARSER_DESC)
    parser.add_argument("weight",
                        help="Path to a valid Meso4 weight, typically .h5 file.",
                        type=Path)
    parser.add_argument("class",
                        help="A value denoting the classification of the directory (0.0 for fake or 1.0 for real).",
                        type=float)
    parser.add_argument("path",
                        help="Path to the dataset directory, containing videos (mp4, avi, mov).",
                        type=Path)
    return parser.parse_args()

def CSV_compute_accuracy(classifier, dirname, act_class, dir, frame_subsample_count = 30):
    labels = ['actual_class', 'predicted_class', 'score', 'type', 'video_name']
    csv_arrs = [labels]

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

            curr_row = [act_class, curr_pred, score, dir, vid] # Format of CSV rows
            csv_arrs.append(curr_row)
        except Exception as error:
            print(f"Error on video {vid}:\n{error}")
            traceback.print_exc()

            curr_row = [act_class, -1.0, -1.0, dir, vid]
            csv_arrs.append(curr_row)
    print(f"All videos predicted")
    return csv_arrs

def create_CSV(data, weight_name, type):
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True) 
    output_path = output_dir / f'{weight_name}-predictions-{type}.csv'
    # Export to CSV
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerows(data)

    print(f"CSV has been completely written.")
    print(f"Script has finished all tasks and ended.")

if __name__ == "__main__":
    args = get_cmd_args()

    # Extract arguments and other required variables
    weight_path = args.weight
    act_class = getattr(args, "class")
    dir_path = args.path
    weight_name = weight_path.stem
    set_dir = dir_path.name

    # Load the model and its pretrained weights
    print("Loading weight")
    classifier = Meso4()
    classifier.load(weight_path)

    # CSV Data formatting
    print("Beginning computations")
    final_data = CSV_compute_accuracy(classifier, str(dir_path), act_class, set_dir)
    print(f"Now writing to CSV")
    create_CSV(final_data, weight_name, set_dir)