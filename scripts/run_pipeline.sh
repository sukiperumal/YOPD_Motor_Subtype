#!/bin/bash

# Master script to run the entire PD motor subtype analysis pipeline

# Set paths
ANALYSIS_DIR="$HOME/pd_analysis"
SCRIPT_DIR="$ANALYSIS_DIR/scripts"

# Create directories
mkdir -p $ANALYSIS_DIR
mkdir -p $SCRIPT_DIR

# Copy all scripts to the script directory
# (assuming scripts are saved with proper names)
cp organize_data.sh $SCRIPT_DIR/
cp preprocess_t1.sh $SCRIPT_DIR/
cp run_first.sh $SCRIPT_DIR/
cp run_freesurfer.sh $SCRIPT_DIR/
cp analyze_results.R $SCRIPT_DIR/

# Make all scripts executable
chmod +x $SCRIPT_DIR/*.sh
chmod +x $SCRIPT_DIR/*.R

# Run the analysis pipeline
echo "Starting PD motor subtype neuroimaging analysis pipeline..."
echo "Step 1: Organizing data"
$SCRIPT_DIR/organize_data.sh

echo "Step 2: Preprocessing T1 images"
$SCRIPT_DIR/preprocess_t1.sh

echo "Step 3: Running FSL-FIRST for subcortical segmentation"
$SCRIPT_DIR/run_first.sh

echo "Step 4: Running FreeSurfer for cortical thickness analysis"
$SCRIPT_DIR/run_freesurfer.sh

echo "Step 5: Statistical analysis and visualization"
$SCRIPT_DIR/analyze_results.R

echo "Analysis pipeline completed. Results are in $ANALYSIS_DIR/stats directory"