#!/bin/bash

# Set paths for data on network share
DATASET_DIR="/mnt/data_NIMHANS"  # This is mounted at \\LAPTOP-78NOUKL7\data_NIMHANS
OUTPUT_DIR="/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"
METADATA="$DATASET_DIR/age_gender.xlsx"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/first_results"
mkdir -p "$OUTPUT_DIR/freesurfer_results"
mkdir -p "$OUTPUT_DIR/stats"
mkdir -p "$OUTPUT_DIR/logs"

echo "Creating subject lists..."

# Check for WSL and mount point
if ! mount | grep -q "$DATASET_DIR"; then
    echo "Error: Data directory not mounted at $DATASET_DIR"
    echo "Please ensure the network share is mounted correctly in WSL."
    echo "You may need to run: sudo mkdir -p /mnt/data_NIMHANS && sudo mount -t drvfs '\\\\LAPTOP-78NOUKL7\\data_NIMHANS' /mnt/data_NIMHANS"
    exit 1
fi

# Check if directories exist
if [ ! -d "$DATASET_DIR/HC" ]; then
    echo "Warning: HC directory not found at $DATASET_DIR/HC"
    echo "Please check if the directory structure is correct."
    exit 1
else
    echo "Found HC directory. Processing..."
    # Create subject lists for each group
    find "$DATASET_DIR/HC" -maxdepth 1 -type d -name "sub-*" | xargs -I{} basename {} > "$OUTPUT_DIR/hc_subjects.txt"
    cat "$OUTPUT_DIR/hc_subjects.txt"
fi

if [ ! -d "$DATASET_DIR/PIGD" ]; then
    echo "Warning: PIGD directory not found at $DATASET_DIR/PIGD"
    echo "Please check if the directory structure is correct."
    exit 1
else
    echo "Found PIGD directory. Processing..."
    find "$DATASET_DIR/PIGD" -maxdepth 1 -type d -name "sub-*" | xargs -I{} basename {} > "$OUTPUT_DIR/pigd_subjects.txt"
    cat "$OUTPUT_DIR/pigd_subjects.txt"
fi

if [ ! -d "$DATASET_DIR/TDPD" ]; then
    echo "Warning: TDPD directory not found at $DATASET_DIR/TDPD"
    echo "Please check if the directory structure is correct."
    exit 1
else
    echo "Found TDPD directory. Processing..."
    find "$DATASET_DIR/TDPD" -maxdepth 1 -type d -name "sub-*" | xargs -I{} basename {} > "$OUTPUT_DIR/tdpd_subjects.txt"
    cat "$OUTPUT_DIR/tdpd_subjects.txt"
fi

# Create a combined subject list with group labels
echo "subject_id,group" > "$OUTPUT_DIR/all_subjects.csv"
cat "$OUTPUT_DIR/hc_subjects.txt" | while read subject; do
    if [ ! -z "$subject" ]; then
        echo "$subject,HC" >> "$OUTPUT_DIR/all_subjects.csv"
    fi
done

cat "$OUTPUT_DIR/pigd_subjects.txt" | while read subject; do
    if [ ! -z "$subject" ]; then
        echo "$subject,PIGD" >> "$OUTPUT_DIR/all_subjects.csv"
    fi
done

cat "$OUTPUT_DIR/tdpd_subjects.txt" | while read subject; do
    if [ ! -z "$subject" ]; then
        echo "$subject,TDPD" >> "$OUTPUT_DIR/all_subjects.csv"
    fi
done

echo "Subject lists created successfully!"