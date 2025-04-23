#!/bin/bash

# Performs three main preprocessing steps for each subject:

# 1. Bias field correction (removing intensity non-uniformities from the MRI)
# 2. Brain extraction (separating brain tissue from non-brain tissue)
# 3. Registration to standard MNI space (aligning all subjects to a common template)

# Set paths
DATASET_DIR="/mnt/data_NIMHANS"
OUTPUT_DIR="/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"

# Set the last processed subject
LAST_PROCESSED="sub-YLOPDHC17"
CONTINUE_PROCESSING=false

# Loop through all subjects
cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | while IFS=, read subject group; do
    # Check if we should start processing (either we already reached the last processed subject
    # or this is the first subject after the last processed one)
    if [[ "$CONTINUE_PROCESSING" == true || "$subject" == "$LAST_PROCESSED" ]]; then
        # If this is the last processed subject, set flag to true for next iteration
        if [[ "$subject" == "$LAST_PROCESSED" ]]; then
            CONTINUE_PROCESSING=true
            echo "Found last processed subject $LAST_PROCESSED, will continue with next subject"
            continue
        fi
        
        echo "Processing $subject from group $group"
        
        # Identify the correct path based on group
        if [[ "$group" == "HC" ]]; then
            SUBJECT_DIR="$DATASET_DIR/HC/$subject"
        elif [[ "$group" == "PIGD" ]]; then
            SUBJECT_DIR="$DATASET_DIR/PIGD/$subject"
        elif [[ "$group" == "TDPD" ]]; then
            SUBJECT_DIR="$DATASET_DIR/TDPD/$subject"
        fi
        
        # Path to T1 image
        T1_IMAGE="$SUBJECT_DIR/ses-01/anat/${subject}_ses-01_run-1_T1w.nii.gz"
        
        # Create output directory for this subject
        mkdir -p $OUTPUT_DIR/preprocessed/$subject
        
        # FSL preprocessing
        # 1. Bias field correction
        echo "Running bias field correction for $subject"
        fast -B -o $OUTPUT_DIR/preprocessed/$subject/bias_corr $T1_IMAGE
        
        # 2. Brain extraction
        echo "Running brain extraction for $subject"
        bet $OUTPUT_DIR/preprocessed/$subject/bias_corr_restore $OUTPUT_DIR/preprocessed/$subject/${subject}_brain -f 0.5 -R
        
        # 3. Registration to MNI space
        echo "Registering to MNI space for $subject"
        flirt -in $OUTPUT_DIR/preprocessed/$subject/${subject}_brain -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain -out $OUTPUT_DIR/preprocessed/$subject/${subject}_brain_mni -omat $OUTPUT_DIR/preprocessed/$subject/${subject}_brain_mni.mat
        
        echo "Preprocessing completed for $subject"
    else
        echo "Skipping already processed subject: $subject"
    fi
done

echo "T1 preprocessing completed for all subjects"