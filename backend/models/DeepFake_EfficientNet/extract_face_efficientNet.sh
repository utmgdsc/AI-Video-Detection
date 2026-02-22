# Define the splits and categories
SPLITS=("train" "val" "test")
CATEGORIES=("fake" "real")
FACE_DIR="/home/gdgteam1/AI-Video-Detection/backend/dataset/FaceForensics++ Dataset/split_dataset"

for SPLIT in "${SPLITS[@]}"; do
    for CAT in "${CATEGORIES[@]}"; do
        echo "----------------------------------------------------"
        echo "Processing: $SPLIT - $CAT"
        echo "----------------------------------------------------"
        
        # Added slashes and changed $SPLITS to $SPLIT
        INPUT_PATH="${FACE_DIR}/${SPLIT}/${CAT}"
        OUTPUT_PATH="ff_extracted/${SPLIT}/${CAT}"

        if [ -d "$INPUT_PATH" ]; then
            echo "Start extracting from $INPUT_PATH"
            python3 scripts/extract_faces.py \
                --input-dir "$INPUT_PATH" \
                --output-dir "$OUTPUT_PATH" \
                --mode video \
                --batch-size 60 \
                --frame-skip 30 \
                --device cuda
        else
            echo "Directory not found: $INPUT_PATH"
        fi
    done
done