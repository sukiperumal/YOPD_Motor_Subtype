# YOPD Motor Subtype ML Analysis

This directory contains scripts for machine learning-based analysis of motor subtypes in Young-Onset Parkinson's Disease (YOPD). The analysis focuses on classifying and distinguishing between PIGD (Postural Instability and Gait Difficulty) and TDPD (Tremor Dominant) subtypes using structural brain measurements and demographic information.

## Overview

The analysis pipeline consists of three main components:

1. **Data Exploration** (`data_exploration.py`): Performs exploratory data analysis on subcortical volumes, cortical thickness, and demographic data.
2. **PD Subtype Classification** (`pd_subtype_classifier.py`): Trains and evaluates machine learning models (SVM and Random Forest) to classify PD subtypes.
3. **Workflow Runner** (`run_ml_analysis.py`): Orchestrates the complete analysis workflow.

## Data Sources

The scripts use the following data sources:

- **Subcortical volumes**: `stats/all_subcortical_volumes.csv`
- **Cortical thickness**: `thickness_output/all_subjects_regional_thickness.csv` 
- **Demographic data**: `age_gender.xlsx`

## Running the Analysis

To run the complete analysis pipeline:

```bash
python scripts/ml_analysis/run_ml_analysis.py
```

This will:
1. Run the data exploration script
2. Generate visualizations and statistical analyses
3. Train machine learning models
4. Evaluate model performance using leave-one-out cross-validation
5. Identify important features for classification

## Output

Results are saved to the `ml_results` directory:

- **Data exploration**: `ml_results/exploration/`
  - Statistical tests
  - Group comparison visualizations
  - Exploration summary report
- **Classification results**: `ml_results/`
  - Model performance metrics
  - ROC curves
  - Confusion matrices
  - Feature importance rankings

## Key Features

1. **Multimodal Classification**: Combines structural brain measurements with demographic information
2. **Feature Selection**: Identifies the most important features for classifying PD subtypes
3. **Statistical Analysis**: Conducts statistical tests to identify significant differences between groups
4. **Visualization**: Provides comprehensive visualizations of group differences and model performance

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- StatsModels
- OpenPyXL (for reading Excel files)

## Customization

You can modify the scripts to:

- Change the machine learning algorithms used
- Adjust feature selection parameters
- Add additional data sources
- Modify the regions of interest for analysis