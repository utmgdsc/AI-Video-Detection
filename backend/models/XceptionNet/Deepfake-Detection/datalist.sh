SRC="/home/gdgteam1/AI-Video-Detection/backend/models/DeepFake-EfficientNet/ff_extracted_faceforensic"
DST="/home/gdgteam1/AI-Video-Detection/backend/models/XceptionNet/Deepfake-Detection/data_list/ff"

for SPLIT in train val test; do
    INPUT="$SRC/$SPLIT/real"
    OUTPUT="$DST/${SPLIT}.txt"
    : > "$OUTPUT"
    find "$SRC/$SPLIT/real" -type f -iname "*.jpg" \
        | sort | awk '{print $0, 0}' >> "$OUTPUT"
    echo "$SPLIT real done"

    INPUT="$SRC/$SPLIT/fake"
    find "$SRC/$SPLIT/fake" -type f -iname "*.jpg" \
        | sort | awk '{print $0, 1}' >> "$OUTPUT"
    echo "$SPLIT fake done"
done
 
echo "DONE"

