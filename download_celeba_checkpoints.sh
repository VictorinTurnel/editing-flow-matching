#!/bin/bash

FILE_ID="1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF"
DEST_DIR="flow-matching/logs/celebahq/checkpoints/"
OUTPUT_NAME="checkpoint_10.pth"
FULL_PATH="$DEST_DIR/$OUTPUT_NAME"

mkdir -p "$DEST_DIR"

if [ ! -f "$FULL_PATH" ]; then
    echo "Downloading weights..."
    gdown --id $FILE_ID -O "$FULL_PATH" --quiet
    echo "Complete"
else
    echo "Already downloaded"
fi
