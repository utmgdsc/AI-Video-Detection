#!/bin/bash
# Generates CSV files of every video in the given directory using Meso4_DF.h5.
# Default path is to ''../../dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_standard_splits/test/'

# These processes run in the background
nohup python3 -u make-video-predictions.py Meso4_DF.h5 real 1.0 > logs/log_real.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 i2v 0.0 > logs/log_i2v.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 t2v 0.0 > logs/log_t2v.txt 2>&1 &
nohup python3 -u make-video-predictions.py Meso4_DF.h5 v2v 0.0 > logs/log_v2v.txt 2>&1 &