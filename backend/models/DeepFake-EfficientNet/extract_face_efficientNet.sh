# Define the splits and categories
SPLITS=("train" "val" "test")
CATEGORIES=("real" "i2v" "t2v" "v2v")

# Loop through every combination
for SPLIT in "${SPLITS[@]}"; do
    for CAT in "${CATEGORIES[@]}"; do
        echo "----------------------------------------------------"
        echo "Processing: $SPLIT / $CAT"
        echo "----------------------------------------------------"

        # Check if the input directory actually exists and has files
        if [ -d "/home/gdgteam1/AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_easyanimate_video_not_in_train_val/$SPLIT/$CAT" ]; then
            echo "Start extracting"
            python3 scripts/extract_faces.py \
              --input-dir "/home/gdgteam1/AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench/split_dataset/dataset_easyanimate_video_not_in_train_val/$SPLIT/$CAT" \
              --output-dir "ff_extracted/$SPLIT/$CAT" \
              --mode video \
              --batch-size 60 \
              --frame-skip 30 \
              --device cuda
        fi
    done
done