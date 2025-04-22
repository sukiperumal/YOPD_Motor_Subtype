#!/bin/bash

# Set paths
DATASET_DIR="/mnt/data_NIMHANS"
OUTPUT_DIR="$HOME/pd_analysis"
FREESURFER_DIR="$OUTPUT_DIR/freesurfer_results"

# Set FreeSurfer environment
export SUBJECTS_DIR=$FREESURFER_DIR

# Loop through all subjects
cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | while IFS=, read subject group; do
    echo "Running FreeSurfer for $subject from group $group"
    
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
    
    # Run FreeSurfer's recon-all
    recon-all -subject $subject -i $T1_IMAGE -all -qcache -openmp 4
    
    echo "FreeSurfer processing completed for $subject"
done

# Extract cortical thickness for regions of interest
mkdir -p $OUTPUT_DIR/stats/cortical_thickness

# Define ROIs for each group based on research hypotheses
PIGD_ROIS="supplementarymotor paracentral precentral lingual precuneus"
TDPD_ROIS="precentral postcentral superiortemporal inferiorparietal"

# Extract thickness for these regions
for hemi in lh rh; do
    for roi in $PIGD_ROIS $TDPD_ROIS; do
        # Create header for this ROI file
        echo "subject_id,group,${hemi}_${roi}" > $OUTPUT_DIR/stats/cortical_thickness/${hemi}_${roi}_thickness.csv
        
        # Extract data for each subject
        cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | while IFS=, read subject group; do
            thickness=$(mri_segstats --annot $subject $hemi aparc --i $FREESURFER_DIR/$subject/surf/${hemi}.thickness --sum /dev/null | grep $roi | awk '{print $5}')
            echo "$subject,$group,$thickness" >> $OUTPUT_DIR/stats/cortical_thickness/${hemi}_${roi}_thickness.csv
        done
    done
done

echo "Cortical thickness extraction completed for all subjects"