#!/bin/bash
# Generates CSV files of every video in the given directory using Meso4_DF.h5.
# Default path is to ''../../dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_standard_splits/test/'

# These processes run in the background
nohup python3 -u make-video-predictions.py Meso4_DF.h5 real 1.0 > logs/log_real.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 i2v 0.0 > logs/log_i2v.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 t2v 0.0 > logs/log_t2v.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 v2v 0.0 > logs/log_v2v.txt 2>&1 &

# These processes run with the given directory '../../../dataset/FaceForensics++ Dataset'
# nohup python3 -u make-video-predictions.py Meso4_DF.h5 Deepfakes 0.0 '../../dataset/FaceForensics++ Dataset' > logs/log_Deepfakes.txt 2>&1 &
# nohup python3 -u make-video-predictions.py Meso4_DF.h5 Face2Face 0.0 '../../dataset/FaceForensics++ Dataset' > logs/log_Face2Face.txt 2>&1 &
# nohup python3 -u make-video-predictions.py Meso4_DF.h5 original 1.0 '../../dataset/FaceForensics++ Dataset' > logs/log_FF_Original.txt 2>&1 &