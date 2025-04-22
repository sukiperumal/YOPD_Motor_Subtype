#!/bin/bash

# Set paths
OUTPUT_DIR="/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"
ROI_LIST="L_Thal,R_Thal,L_Putamen,R_Putamen,L_Caud,R_Caud,L_Pall,R_Pall,L_Hipp,R_Hipp,L_Amyg,R_Amyg,L_Accu,R_Accu,BrStem"

# Configuration options
EXPLORATORY_MODE=false  # Set to true for faster exploratory runs
PARALLEL_PROCESSING=true  # Set to false if you want sequential processing
NUM_CORES=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) - 1 ))
if [ $NUM_CORES -lt 1 ]; then NUM_CORES=1; fi

# Create necessary directories in advance
mkdir -p $OUTPUT_DIR/stats
mkdir -p $OUTPUT_DIR/first_results

# Set boundary correction based on mode
BOUNDARY_OPTION=""
if [ "$EXPLORATORY_MODE" = true ]; then
    BOUNDARY_OPTION="-b none"
    echo "Running in EXPLORATORY mode: skipping boundary correction for faster processing"
else
    echo "Running in STANDARD mode: with full boundary correction"
fi

# Set OpenBLAS threading for better performance
export OPENBLAS_NUM_THREADS=1
echo "Set OPENBLAS_NUM_THREADS=1 to optimize multi-core performance"

# Define the processing function for a single subject
process_subject() {
    local subject=$1
    local group=$2
    
    echo "Running FSL-FIRST for $subject from group $group"
    
    # Set paths for this subject
    PREPROCESSED_T1="$OUTPUT_DIR/preprocessed/$subject/${subject}_brain"
    OUTPUT_FIRST="$OUTPUT_DIR/first_results/$subject"
    mkdir -p $OUTPUT_FIRST
    
    # Check if preprocessed file exists
    if [ ! -f "${PREPROCESSED_T1}.nii.gz" ]; then
        echo "ERROR: Preprocessed T1 file not found: ${PREPROCESSED_T1}.nii.gz"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        echo "Created placeholder CSV with NA values for $subject"
        return 1
    fi
    
    # Run FSL-FIRST for subcortical segmentation with boundary option
    echo "Running: run_first_all -i ${PREPROCESSED_T1} -o ${OUTPUT_FIRST}/${subject}_first -s ${ROI_LIST// /} ${BOUNDARY_OPTION}"
    run_first_all -i ${PREPROCESSED_T1} -o ${OUTPUT_FIRST}/${subject}_first -s ${ROI_LIST// /} ${BOUNDARY_OPTION}
    
    # Check if FIRST was successful by looking for the output segmentation file
    if [ ! -f "${OUTPUT_FIRST}/${subject}_first_all_fast_firstseg.nii.gz" ]; then
        echo "ERROR: FSL-FIRST failed to generate output for $subject"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        echo "Created placeholder CSV with NA values for $subject"
        return 1
    fi
    
    # Extract volumes from FIRST results
    echo "Extracting volumes for $subject"
    first_utils --vertexAnalysis --usebvars -i $OUTPUT_FIRST/${subject}_first_all_fast_firstseg.nii.gz -o $OUTPUT_FIRST/${subject}_volumes
    
    # Check if volumes file was generated
    if [ ! -f "$OUTPUT_FIRST/${subject}_volumes.bvars" ]; then
        echo "WARNING: Volume extraction failed for $subject - .bvars file not created"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        echo "Created placeholder CSV with NA values for $subject"
        return 1
    fi
    
    # Create a CSV with volumes for this subject
    echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
    for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
        if grep -q "$structure" $OUTPUT_FIRST/${subject}_volumes.bvars; then
            vol=$(grep "$structure" $OUTPUT_FIRST/${subject}_volumes.bvars | awk '{print $2}')
            echo "$structure,$vol" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        else
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        fi
    done
    
    echo "FSL-FIRST completed for $subject"
}

# Export the function for parallel use
export -f process_subject
export BOUNDARY_OPTION
export OUTPUT_DIR

echo "Processing will use $NUM_CORES parallel processes"

# Process subjects sequentially or in parallel
if [ "$PARALLEL_PROCESSING" = true ]; then
    # Check if GNU Parallel is available
    if command -v parallel >/dev/null 2>&1; then
        cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | parallel --colsep ',' -j $NUM_CORES "process_subject {1} {2}"
    else
        echo "GNU Parallel not found. Falling back to sequential processing."
        cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | while IFS=, read subject group; do
            process_subject "$subject" "$group"
        done
    fi
else
    cat $OUTPUT_DIR/all_subjects.csv | tail -n +2 | while IFS=, read subject group; do
        process_subject "$subject" "$group"
    done
fi

# Combine all subject results
echo "Combining all subcortical volume results"
echo "subject_id,group,structure,volume_mm3" > $OUTPUT_DIR/stats/all_subcortical_volumes.csv

# Use a more efficient approach to combine results
find $OUTPUT_DIR/first_results -name '*_subcortical_volumes.csv' | while read file; do
    subject=$(basename $(dirname "$file"))
    group=$(grep "^$subject," $OUTPUT_DIR/all_subjects.csv | cut -d, -f2)
    tail -n +2 "$file" | awk -v subj="$subject" -v grp="$group" '{print subj "," grp "," $0}' >> $OUTPUT_DIR/stats/all_subcortical_volumes.csv
done

echo "Subcortical segmentation and volume extraction completed for all subjects"