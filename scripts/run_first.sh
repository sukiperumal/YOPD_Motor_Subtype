#!/bin/bash

# Set paths - ensure we're using WSL compatible paths
if [[ "$OUTPUT_DIR" == /mnt/c/* ]]; then
    # Already in WSL format
    OUTPUT_DIR="/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"
elif [[ "$OUTPUT_DIR" == C:/* ]] || [[ "$OUTPUT_DIR" == c:/* ]]; then
    # Convert Windows path to WSL path
    OUTPUT_DIR="/mnt/c${OUTPUT_DIR#?:}"
else
    # Default path
    OUTPUT_DIR="/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"
fi

# Debug directory structure
ls -la "$OUTPUT_DIR/preprocessed" > /dev/null 2>&1 || { echo "Error: Preprocessed directory not found at $OUTPUT_DIR/preprocessed"; exit 1; }

ROI_LIST="L_Thal,R_Thal,L_Putamen,R_Putamen,L_Caud,R_Caud,L_Pall,R_Pall,L_Hipp,R_Hipp,L_Amyg,R_Amyg,L_Accu,R_Accu,BrStem"

# Configuration options
EXPLORATORY_MODE=false  # Set to true for faster exploratory runs
PARALLEL_PROCESSING=true  # Set to false if you want sequential processing
NUM_CORES=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) - 1 ))
if [ $NUM_CORES -lt 1 ]; then NUM_CORES=1; fi

# Setup logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [${level}] ${message}"
}

log_message "INFO" "=== FSL-FIRST Subcortical Segmentation Pipeline Starting ==="
log_message "INFO" "Using output directory: $OUTPUT_DIR"
log_message "INFO" "Configuration: EXPLORATORY_MODE=$EXPLORATORY_MODE, PARALLEL_PROCESSING=$PARALLEL_PROCESSING, NUM_CORES=$NUM_CORES"

# Create necessary directories in advance
log_message "INFO" "STEP 1: Creating output directories"
mkdir -p $OUTPUT_DIR/stats
mkdir -p $OUTPUT_DIR/first_results
mkdir -p $OUTPUT_DIR/logs

# Create a log file
LOG_FILE="$OUTPUT_DIR/logs/first_segmentation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
log_message "INFO" "Logging to: $LOG_FILE"

# Set boundary correction based on mode
BOUNDARY_OPTION=""
if [ "$EXPLORATORY_MODE" = true ]; then
    BOUNDARY_OPTION="-b none"
    log_message "INFO" "Running in EXPLORATORY mode: skipping boundary correction for faster processing"
else
    log_message "INFO" "Running in STANDARD mode: with full boundary correction"
fi

# Set OpenBLAS threading for better performance
export OPENBLAS_NUM_THREADS=1
log_message "INFO" "Set OPENBLAS_NUM_THREADS=1 to optimize multi-core performance"

# Define the processing function for a single subject
process_subject() {
    local subject=$1
    local group=$2
    local roi_list=$3
    
    log_message "INFO" "SUBJECT $subject (GROUP: $group): Starting FSL-FIRST processing"
    
    # Set paths for this subject
    PREPROCESSED_T1="$OUTPUT_DIR/preprocessed/$subject/${subject}_brain"
    OUTPUT_FIRST="$OUTPUT_DIR/first_results/$subject"
    mkdir -p $OUTPUT_FIRST
    
    # Debug information - List directory contents
    log_message "DEBUG" "Subject directory contains: $(ls -la $OUTPUT_DIR/preprocessed/$subject)"
    
    # Check if preprocessed file exists
    if [ ! -f "${PREPROCESSED_T1}.nii.gz" ]; then
        log_message "ERROR" "SUBJECT $subject: Preprocessed T1 file not found: ${PREPROCESSED_T1}.nii.gz"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        log_message "INFO" "SUBJECT $subject: Created placeholder CSV with NA values"
        return 1
    fi
    
    # Run FSL-FIRST for subcortical segmentation with boundary option
    log_message "INFO" "SUBJECT $subject: STEP 1/3 - Running subcortical segmentation"
    log_message "DEBUG" "SUBJECT $subject: Command: run_first_all -i ${PREPROCESSED_T1} -o ${OUTPUT_FIRST}/${subject}_first -s $roi_list ${BOUNDARY_OPTION}"
    run_first_all -i ${PREPROCESSED_T1} -o ${OUTPUT_FIRST}/${subject}_first -s $roi_list ${BOUNDARY_OPTION}
    
    # Check if FIRST was successful by looking for the output segmentation file
    if [ ! -f "${OUTPUT_FIRST}/${subject}_first_all_fast_firstseg.nii.gz" ]; then
        log_message "ERROR" "SUBJECT $subject: FSL-FIRST failed to generate output segmentation"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        log_message "INFO" "SUBJECT $subject: Created placeholder CSV with NA values"
        return 1
    fi
    
    # Extract volumes from FIRST results
    log_message "INFO" "SUBJECT $subject: STEP 2/3 - Extracting subcortical volumes"
    first_utils --vertexAnalysis --usebvars -i $OUTPUT_FIRST/${subject}_first_all_fast_firstseg.nii.gz -o $OUTPUT_FIRST/${subject}_volumes
    
    # Check if volumes file was generated
    if [ ! -f "$OUTPUT_FIRST/${subject}_volumes.bvars" ]; then
        log_message "WARNING" "SUBJECT $subject: Volume extraction failed - .bvars file not created"
        echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        done
        log_message "INFO" "SUBJECT $subject: Created placeholder CSV with NA values"
        return 1
    fi
    
    # Create a CSV with volumes for this subject
    log_message "INFO" "SUBJECT $subject: STEP 3/3 - Creating CSV output"
    echo "structure,volume_mm3" > $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
    for structure in L_Thal R_Thal L_Putamen R_Putamen L_Caud R_Caud L_Pall R_Pall L_Hipp R_Hipp L_Amyg R_Amyg L_Accu R_Accu BrStem; do
        if grep -q "$structure" $OUTPUT_FIRST/${subject}_volumes.bvars; then
            vol=$(grep "$structure" $OUTPUT_FIRST/${subject}_volumes.bvars | awk '{print $2}')
            echo "$structure,$vol" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        else
            echo "$structure,NA" >> $OUTPUT_FIRST/${subject}_subcortical_volumes.csv
        fi
    done
    
    log_message "INFO" "SUBJECT $subject: FSL-FIRST processing completed successfully"
}

# Export the function for parallel use
export -f process_subject
export BOUNDARY_OPTION
export OUTPUT_DIR
export ROI_LIST
export -f log_message

log_message "INFO" "STEP 2: Identifying subjects from preprocessed directory"
log_message "DEBUG" "Preprocessed directory contains: $(ls -la $OUTPUT_DIR/preprocessed)"

# Find all subject directories and process directly
declare -a SUBJECTS=()
declare -a GROUPS=()
SUBJECT_COUNT=0

# Direct approach to finding subject directories
log_message "DEBUG" "Looking for subject directories using find command"
for subject_dir in $(find "$OUTPUT_DIR/preprocessed" -type d -name "sub-*" 2>/dev/null); do
    subject=$(basename "$subject_dir")
    log_message "DEBUG" "Examining potential subject directory: $subject"
    
    # Show exact list of files to help diagnose the issue
    log_message "DEBUG" "NIfTI files in directory: $(find "$subject_dir" -name "*.nii.gz" -exec basename {} \; | tr '\n' ' ')"
    
    # Determine group based on naming convention
    if [[ "$subject" == *"HC"* ]]; then
        group="HC"
    else
        group="PD"
    fi
    
    # Check specifically for brain file with exact name pattern
    brain_file="$subject_dir/${subject}_brain.nii.gz"
    if [ -f "$brain_file" ]; then
        log_message "DEBUG" "Found brain file: $brain_file"
        SUBJECTS+=("$subject")
        GROUPS+=("$group")
        log_message "INFO" "Found subject $subject (group: $group) with brain file"
        SUBJECT_COUNT=$((SUBJECT_COUNT + 1))
    else
        log_message "WARNING" "Subject $subject: No brain file found at expected path: $brain_file"
    fi
done

log_message "INFO" "Found $SUBJECT_COUNT subjects to process"

# If no subjects found with exact matching, try with a more relaxed naming pattern
if [ $SUBJECT_COUNT -eq 0 ]; then
    log_message "INFO" "Trying alternative approach to find brain files..."
    
    for subject_dir in $(find "$OUTPUT_DIR/preprocessed" -type d -name "sub-*" 2>/dev/null); do
        subject=$(basename "$subject_dir")
        
        # Determine group based on naming convention
        if [[ "$subject" == *"HC"* ]]; then
            group="HC"
        else
            group="PD"
        fi
        
        # Find any file with "brain" in the name
        brain_files=$(find "$subject_dir" -name "*brain*.nii.gz")
        if [ -n "$brain_files" ]; then
            # Use the first brain file found
            brain_file=$(echo "$brain_files" | head -n1)
            log_message "DEBUG" "Using alternative brain file: $brain_file"
            
            # Create a symbolic link with the expected name if it doesn't exist
            expected_brain_file="$subject_dir/${subject}_brain.nii.gz"
            if [ ! -f "$expected_brain_file" ]; then
                ln -sf "$brain_file" "$expected_brain_file"
                log_message "DEBUG" "Created symbolic link from $brain_file to $expected_brain_file"
            fi
            
            SUBJECTS+=("$subject")
            GROUPS+=("$group")
            log_message "INFO" "Found subject $subject (group: $group) with alternative brain file"
            SUBJECT_COUNT=$((SUBJECT_COUNT + 1))
        else
            log_message "WARNING" "Subject $subject: No brain files found in directory"
        fi
    done
    
    log_message "INFO" "Found $SUBJECT_COUNT subjects with alternative brain file naming"
fi

# Process subjects sequentially or in parallel
if [ $SUBJECT_COUNT -gt 0 ]; then
    log_message "INFO" "STEP 3: Running FSL-FIRST segmentation on all subjects"
    
    if [ "$PARALLEL_PROCESSING" = true ]; then
        # Check if GNU Parallel is available
        if command -v parallel >/dev/null 2>&1; then
            log_message "INFO" "Using GNU Parallel with $NUM_CORES processes"
            
            # Create a temporary file with subject and group info for parallel processing
            TEMP_SUBJ_FILE=$(mktemp)
            for ((i=0; i<${#SUBJECTS[@]}; i++)); do
                echo "${SUBJECTS[$i]},${GROUPS[$i]}" >> $TEMP_SUBJ_FILE
            done
            
            # Display the content of the temp file for debugging
            log_message "DEBUG" "Subject list for processing: $(cat $TEMP_SUBJ_FILE)"
            
            # Use parallel to process the subjects
            cat $TEMP_SUBJ_FILE | parallel --colsep ',' -j $NUM_CORES "process_subject {1} {2} $ROI_LIST"
            
            # Remove the temporary file
            rm $TEMP_SUBJ_FILE
        else
            log_message "WARNING" "GNU Parallel not found. Falling back to sequential processing."
            
            for ((i=0; i<${#SUBJECTS[@]}; i++)); do
                process_subject "${SUBJECTS[$i]}" "${GROUPS[$i]}" "$ROI_LIST"
            done
        fi
    else
        log_message "INFO" "Sequential processing selected"
        
        for ((i=0; i<${#SUBJECTS[@]}; i++)); do
            process_subject "${SUBJECTS[$i]}" "${GROUPS[$i]}" "$ROI_LIST"
        done
    fi
else
    log_message "ERROR" "No valid subjects found to process"
fi

# Combine all subject results
log_message "INFO" "STEP 4: Combining all subcortical volume results"
echo "subject_id,group,structure,volume_mm3" > $OUTPUT_DIR/stats/all_subcortical_volumes.csv

# Use a more efficient approach to combine results
find $OUTPUT_DIR/first_results -type d -name 'sub-*' 2>/dev/null | while read dir; do
    subject=$(basename "$dir")
    
    # Only process directories that follow our subject naming convention
    if [[ ! "$subject" =~ ^sub-[A-Za-z0-9]+ ]]; then
        continue
    fi
    
    # Check if CSV exists
    if [ ! -f "$dir/${subject}_subcortical_volumes.csv" ]; then
        log_message "WARNING" "No subcortical volumes CSV found for $subject - skipping"
        continue
    fi
    
    # Determine group based on naming convention
    if [[ "$subject" == *"HC"* ]]; then
        group="HC"
    else
        group="PD"
    fi
    
    log_message "DEBUG" "Adding results from $subject (group: $group) to combined output"
    tail -n +2 "$dir/${subject}_subcortical_volumes.csv" | awk -v subj="$subject" -v grp="$group" '{print subj "," grp "," $0}' >> $OUTPUT_DIR/stats/all_subcortical_volumes.csv
done

# Check if any results were combined
if [ "$(wc -l < $OUTPUT_DIR/stats/all_subcortical_volumes.csv)" -le 1 ]; then
    log_message "WARNING" "No subcortical volume results were found to combine"
else
    log_message "INFO" "Successfully combined results from $(( $(wc -l < $OUTPUT_DIR/stats/all_subcortical_volumes.csv) - 1 )) volume measurements"
fi

log_message "INFO" "=== FSL-FIRST Subcortical Segmentation Pipeline Completed ==="
log_message "INFO" "Results saved to $OUTPUT_DIR/stats/all_subcortical_volumes.csv"