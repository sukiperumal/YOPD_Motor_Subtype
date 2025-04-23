#!/usr/bin/env python3

"""
PD Subtype Classifier

This script implements machine learning models to classify Parkinson's Disease subtypes
(PIGD vs TDPD) using structural brain measurements and demographic features.
The script includes:
- Data loading and preparation
- Feature selection and engineering
- Model training (SVM and Random Forest)
- Leave-one-out cross-validation
- Performance evaluation and visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import openpyxl
import warnings

warnings.filterwarnings('ignore')

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
OUTPUT_DIR = PROJECT_DIR / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime%s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / f'pd_subtype_classifier_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pd_classifier')

def load_demographic_data():
    """Load demographic data from the Excel file"""
    logger.info("Loading demographic data")
    try:
        demo_file = PROJECT_DIR / "age_gender.xlsx"
        excel_data = pd.read_excel(demo_file, engine='openpyxl')
        
        # Check if the expected columns exist
        required_cols = ['sub', 'TDPD', 'PIGD', 'HC', 'age_assessment']
        missing_cols = [col for col in required_cols if col not in excel_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Could not identify required columns. Available columns: {excel_data.columns}")
        
        # Create a proper subject_id and group format from the data
        subject_groups = {}
        subject_to_age = {}
        subject_to_gender = {}
        
        # For each subject, determine their group (TDPD, PIGD, or HC) based on the flag columns
        for _, row in excel_data.iterrows():
            if pd.isna(row['sub']) or not isinstance(row['sub'], str):
                continue
                
            subject = row['sub']
            # Convert to the format used in other files (sub-YLOPD...)
            if not subject.startswith('sub-'):
                subject = f"sub-{subject.replace('_', '')}"
                
            # Determine group based on flag columns
            if row['TDPD'] == 1:
                group = 'TDPD'
            elif row['PIGD'] == 1:
                group = 'PIGD'
            elif row['HC'] == 1:
                group = 'HC'
            else:
                group = 'Unknown'
                
            subject_groups[subject] = group
            
            # Add age if available
            if 'age_assessment' in row and not pd.isna(row['age_assessment']):
                subject_to_age[subject] = row['age_assessment']
            
            # Add gender if available
            if 'gender' in row and not pd.isna(row['gender']):
                subject_to_gender[subject] = row['gender']
        
        # Create a new dataframe with subject_id and group
        new_demo_data = pd.DataFrame({
            'subject_id': list(subject_groups.keys()),
            'group': list(subject_groups.values())
        })
        
        # Add age column if we have age data
        if subject_to_age:
            new_demo_data['age'] = new_demo_data['subject_id'].map(subject_to_age)
        
        # Add gender column if we have gender data
        if subject_to_gender:
            new_demo_data['gender'] = new_demo_data['subject_id'].map(subject_to_gender)
        
        logger.info(f"Loaded demographic data with {len(new_demo_data)} subjects")
        return new_demo_data
    except Exception as e:
        logger.error(f"Failed to load demographic data: {e}")
        return pd.DataFrame(columns=['subject_id', 'group'])

def load_subcortical_data():
    """Load subcortical volume data"""
    logger.info("Loading subcortical volume data")
    try:
        subcort_file = PROJECT_DIR / "stats" / "all_subcortical_volumes.csv"
        subcort_data = pd.read_csv(subcort_file)
        
        # Pivot the data to get one row per subject
        pivot_data = subcort_data.pivot(index=['subject_id', 'group'], columns='structure', values='volume_mm3').reset_index()
        
        logger.info(f"Loaded subcortical volume data for {pivot_data['subject_id'].nunique()} subjects and {len(pivot_data.columns) - 2} structures")
        return pivot_data
    except Exception as e:
        logger.error(f"Failed to load subcortical data: {e}")
        return None

def load_cortical_data():
    """Load cortical thickness data"""
    logger.info("Loading cortical thickness data")
    try:
        thickness_file = PROJECT_DIR / "thickness_output" / "all_subjects_regional_thickness.csv"
        cortical_data = pd.read_csv(thickness_file)
        
        # Ensure subject_id column exists
        if 'subject_id' not in cortical_data.columns:
            cortical_data['subject_id'] = cortical_data['Subject']
            
        # Pivot the data to get one row per subject
        # First create a unique ID for each row by combining subject and region
        cortical_data['subject_region'] = cortical_data['subject_id'] + '_' + cortical_data['Region']
        
        # Then pivot the data
        pivot_data = cortical_data.pivot_table(
            index='subject_id',
            columns='Region', 
            values='Mean_Thickness'
        ).reset_index()
        
        logger.info(f"Loaded cortical thickness data for {pivot_data['subject_id'].nunique()} subjects and {len(pivot_data.columns) - 1} regions")
        return pivot_data
    except Exception as e:
        logger.error(f"Failed to load cortical thickness data: {e}")
        return None

def merge_datasets(demo_data, subcort_data, cortical_data):
    """Merge all datasets into one"""
    logger.info("Merging datasets")
    
    try:
        if demo_data is None or len(demo_data) == 0:
            logger.error("No demographic data available")
            return None
            
        # First create the pivot table from subcortical data if it's a detailed format
        if subcort_data is not None and 'structure' in subcort_data.columns:
            # Pivot to have one row per subject with structures as columns
            subcort_pivot = subcort_data.pivot_table(
                index=['subject_id', 'group'],
                columns='structure', 
                values='volume_mm3'
            ).reset_index()
            logger.info(f"Pivoted subcortical data with {len(subcort_pivot)} rows and {len(subcort_pivot.columns)} columns")
        else:
            subcort_pivot = subcort_data
        
        # Start with demographic data as the base
        merged_data = demo_data.copy()
        
        # Make sure subject_id column exists in all datasets
        for df, name in [(subcort_pivot, 'subcortical'), (cortical_data, 'cortical')]:
            if df is not None and 'subject_id' not in df.columns:
                if 'Subject' in df.columns:
                    df['subject_id'] = df['Subject']
                    logger.info(f"Renamed 'Subject' to 'subject_id' in {name} data")
                else:
                    logger.error(f"No subject ID column found in {name} data")
                    return None
        
        # Merge with subcortical data if available
        if subcort_pivot is not None and len(subcort_pivot) > 0:
            # Use outer merge to keep all subjects
            merged_data = merged_data.merge(
                subcort_pivot.drop('group', axis=1, errors='ignore'),
                on='subject_id',
                how='outer'
            )
            logger.info(f"Merged with subcortical data, now {len(merged_data)} rows")
        
        # Merge with cortical data if available
        if cortical_data is not None and len(cortical_data) > 0:
            merged_data = merged_data.merge(
                cortical_data,
                on='subject_id',
                how='outer'
            )
            logger.info(f"Merged with cortical data, now {len(merged_data)} rows")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        return None

def prepare_data_for_modeling(merged_data):
    """Prepare data for modeling, including filtering for PIGD and TDPD subtypes"""
    logger.info("Preparing data for modeling")
    try:
        # Filter for only PIGD and TDPD subjects
        model_data = merged_data[merged_data['group'].isin(['PIGD', 'TDPD'])].copy()
        logger.info(f"Selected {len(model_data)} subjects from PIGD and TDPD groups")
        
        # Count subjects in each group
        group_counts = model_data['group'].value_counts()
        logger.info(f"Group distribution: {dict(group_counts)}")
        
        # Check if we have both classes and at least 2 examples of each
        can_classify = True
        
        # First check if we have both PIGD and TDPD groups
        if 'PIGD' not in group_counts or 'TDPD' not in group_counts:
            logger.warning(f"Missing one or more required groups. Found groups: {list(group_counts.index)}")
            can_classify = False
        # Now check if we have at least 2 samples for both groups
        elif group_counts['PIGD'] < 2 or group_counts['TDPD'] < 2:
            logger.warning(f"Need at least 2 subjects per group. Found: PIGD={group_counts.get('PIGD', 0)}, TDPD={group_counts.get('TDPD', 0)}")
            can_classify = False
        
        if not can_classify:
            logger.warning("Classification requires at least two classes with multiple examples")
            logger.warning("Will run basic descriptive statistics instead of classification")
            
            # Just return the data with a flag indicating only one class
            return model_data, list(model_data.columns), False
        
        # Encode the target variable
        model_data['target'] = (model_data['group'] == 'PIGD').astype(int)
        
        # List of potential confounding columns to exclude from features
        exclude_cols = ['subject_id', 'group', 'Subject', 'target']
        feature_cols = [col for col in model_data.columns if col not in exclude_cols]
        
        # Handle missing values
        model_data[feature_cols] = model_data[feature_cols].fillna(model_data[feature_cols].mean())
        
        logger.info(f"Prepared data with {len(feature_cols)} features")
        return model_data, feature_cols, True
    except Exception as e:
        logger.error(f"Error preparing data for modeling: {e}")
        return None, None, False

def advanced_feature_engineering(model_data, feature_cols):
    """Create advanced features using domain-specific knowledge about PD subtypes"""
    logger.info("Performing advanced feature engineering for PD subtype classification")
    
    # Create a copy of the data to work with
    enhanced_data = model_data.copy()
    enhanced_features = feature_cols.copy()
    
    # 1. Create hemisphere asymmetry features for subcortical structures
    # Typically, asymmetry in structures like putamen, caudate, and thalamus is relevant in PD
    lateralized_pairs = {
        'L_Putamen': 'R_Putamen',
        'L_Caud': 'R_Caud',
        'L_Thal': 'R_Thal',
        'L_Pall': 'R_Pall',
        'L_Hipp': 'R_Hipp',
        'L_Amyg': 'R_Amyg',
        'L_Accu': 'R_Accu'
    }
    
    for left, right in lateralized_pairs.items():
        if left in feature_cols and right in feature_cols:
            # Create asymmetry index (normalized difference between hemispheres)
            asymmetry_name = f"Asymmetry_{left[2:]}"
            enhanced_data[asymmetry_name] = (enhanced_data[left] - enhanced_data[right]) / (enhanced_data[left] + enhanced_data[right])
            enhanced_features.append(asymmetry_name)
            
            # Create ratio between hemispheres
            ratio_name = f"Ratio_{left[2:]}"
            enhanced_data[ratio_name] = enhanced_data[left] / enhanced_data[right]
            enhanced_features.append(ratio_name)
            
            logger.info(f"Created asymmetry features for {left}/{right}")
    
    # 2. Create neuroanatomical network features
    # Motor circuit: includes putamen, caudate, thalamus, and motor cortices
    motor_regions = []
    sensorimotor_regions = []
    prefrontal_regions = []
    temporal_regions = []
    
    # Find motor cortex regions using partial name matching
    for col in feature_cols:
        if any(region in col.lower() for region in ['precentral', 'motor', 'sma', 'supplementary']):
            motor_regions.append(col)
        elif any(region in col.lower() for region in ['postcentral', 'sensory', 'parietal']):
            sensorimotor_regions.append(col)
        elif any(region in col.lower() for region in ['frontal', 'prefrontal', 'orbitofrontal']):
            prefrontal_regions.append(col)
        elif any(region in col.lower() for region in ['temporal', 'hippocampus', 'amygdala']):
            temporal_regions.append(col)
    
    # Add subcortical components of motor circuit
    subcortical_motor = ['L_Putamen', 'R_Putamen', 'L_Caud', 'R_Caud', 'L_Thal', 'R_Thal']
    motor_regions.extend([region for region in subcortical_motor if region in feature_cols])
    
    # Create network summary features
    for network_name, regions in [
        ('Motor_Network', motor_regions),
        ('Sensorimotor_Network', sensorimotor_regions),
        ('Prefrontal_Network', prefrontal_regions),
        ('Temporal_Network', temporal_regions)
    ]:
        if regions:
            # Calculate mean thickness/volume for the network
            if len(regions) > 0:
                enhanced_data[f"Mean_{network_name}"] = enhanced_data[regions].mean(axis=1)
                enhanced_features.append(f"Mean_{network_name}")
                
                # Calculate std dev (variability) within network
                if len(regions) > 1:
                    enhanced_data[f"Std_{network_name}"] = enhanced_data[regions].std(axis=1)
                    enhanced_features.append(f"Std_{network_name}")
                
                logger.info(f"Created {network_name} features from {len(regions)} regions")
    
    # 3. Create interaction terms between key regions
    # PIGD typically involves more diffuse brain involvement, while TDPD is more focal
    # Create interactions between cortical and subcortical structures
    
    # First select key cortical regions (limit to a reasonable number)
    key_cortical = []
    for col in feature_cols:
        if any(region in col.lower() for region in ['precentral', 'supplementary', 'putamen', 'caudate', 'thalamus']):
            key_cortical.append(col)
    
    # Add a few important subcortical structures
    key_subcortical = ['L_Putamen', 'R_Putamen', 'L_Thal', 'R_Thal', 'BrStem']
    key_subcortical = [s for s in key_subcortical if s in feature_cols]
    
    # Create interactions between key regions
    if key_cortical and key_subcortical:
        # Limit the number of interactions to avoid explosion
        limit_cortical = key_cortical[:5] if len(key_cortical) > 5 else key_cortical
        limit_subcortical = key_subcortical[:3] if len(key_subcortical) > 3 else key_subcortical
        
        for cortical in limit_cortical:
            for subcortical in limit_subcortical:
                interaction_name = f"Interact_{cortical}_{subcortical}".replace(" ", "")
                # Create standardized interaction terms to avoid scale issues
                c_std = (enhanced_data[cortical] - enhanced_data[cortical].mean()) / enhanced_data[cortical].std()
                s_std = (enhanced_data[subcortical] - enhanced_data[subcortical].mean()) / enhanced_data[subcortical].std()
                enhanced_data[interaction_name] = c_std * s_std
                enhanced_features.append(interaction_name)
        
        logger.info(f"Created {len(limit_cortical) * len(limit_subcortical)} interaction features")
    
    # 4. Create aggregate features across brain lobes
    lobe_map = {
        'frontal': [],
        'parietal': [],
        'temporal': [],
        'occipital': [],
        'limbic': []
    }
    
    # Map regions to lobes based on name patterns
    for col in feature_cols:
        col_lower = col.lower()
        if any(region in col_lower for region in ['frontal', 'precentral', 'opercularis', 'orbital']):
            lobe_map['frontal'].append(col)
        elif any(region in col_lower for region in ['parietal', 'postcentral', 'supramarginal', 'angular']):
            lobe_map['parietal'].append(col)
        elif any(region in col_lower for region in ['temporal', 'temporo']):
            lobe_map['temporal'].append(col)
        elif any(region in col_lower for region in ['occipital', 'calcarine', 'cuneal', 'lingual']):
            lobe_map['occipital'].append(col)
        elif any(region in col_lower for region in ['cingulate', 'hippocampus', 'amygdala', 'parahippocampal']):
            lobe_map['limbic'].append(col)
    
    # Create lobe-specific features
    for lobe, regions in lobe_map.items():
        if len(regions) > 0:
            # Calculate mean thickness/volume for the lobe
            enhanced_data[f"Mean_{lobe}_lobe"] = enhanced_data[regions].mean(axis=1)
            enhanced_features.append(f"Mean_{lobe}_lobe")
            
            # Calculate ratio of frontal to other lobes (important for PD subtypes)
            if lobe != 'frontal' and len(lobe_map['frontal']) > 0:
                ratio_name = f"Ratio_frontal_to_{lobe}"
                frontal_mean = enhanced_data[lobe_map['frontal']].mean(axis=1)
                lobe_mean = enhanced_data[regions].mean(axis=1)
                enhanced_data[ratio_name] = frontal_mean / lobe_mean
                enhanced_features.append(ratio_name)
            
            logger.info(f"Created {lobe} lobe features from {len(regions)} regions")
    
    # 5. Age interactions (age is often a confounding factor in PD)
    if 'age' in enhanced_data.columns:
        # Age interaction with key structures
        for region in key_subcortical[:3]:  # Limit to avoid feature explosion
            if region in feature_cols:
                age_interaction = f"Age_x_{region}"
                enhanced_data[age_interaction] = enhanced_data['age'] * enhanced_data[region]
                enhanced_features.append(age_interaction)
        
        logger.info("Created age interaction features")
    
    logger.info(f"Finished feature engineering. Original features: {len(feature_cols)}, Enhanced features: {len(enhanced_features)}")
    return enhanced_data, enhanced_features

def run_descriptive_analysis(model_data, feature_cols):
    """Run descriptive analysis when classification is not possible"""
    logger.info("Running descriptive analysis on the available data")
    
    group = model_data['group'].iloc[0]  # Get the only group present
    
    # Create output directory for descriptive statistics
    descriptive_dir = OUTPUT_DIR / "descriptive_statistics"
    descriptive_dir.mkdir(exist_ok=True, parents=True)
    
    # Basic descriptive statistics
    descriptive_stats = model_data[feature_cols].describe().transpose()
    descriptive_stats['coefficient_of_variation'] = descriptive_stats['std'] / descriptive_stats['mean'] * 100
    descriptive_stats.to_csv(descriptive_dir / f"{group}_descriptive_statistics.csv")
    
    # Visualize distributions of key features
    # Find features with highest variance
    feature_variance = model_data[feature_cols].var().sort_values(ascending=False)
    top_features = feature_variance.head(10).index.tolist()
    
    # Plot histograms for top features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(model_data[feature], kde=True)
        plt.title(f"Distribution of {feature} in {group} group")
        plt.savefig(descriptive_dir / f"{group}_{feature}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Correlation analysis
    plt.figure(figsize=(20, 16))
    corr_matrix = model_data[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title(f"Correlation Matrix for {group} group")
    plt.tight_layout()
    plt.savefig(descriptive_dir / f"{group}_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Descriptive analysis completed. Results saved to {descriptive_dir}")
    
    # Create a simple report
    with open(descriptive_dir / f"{group}_report.md", 'w') as report:
        report.write(f"# Descriptive Analysis for {group} Subjects\n\n")
        report.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.write(f"## Overview\n\n")
        report.write(f"Total subjects: {len(model_data)}\n\n")
        report.write(f"## Key Statistics\n\n")
        
        report.write("| Feature | Mean | Std | Min | Max | CV% |\n")
        report.write("|---------|------|-----|-----|-----|-----|\n")
        
        for feature in top_features:
            stats = descriptive_stats.loc[feature]
            report.write(f"| {feature} | {stats['mean']:.2f} | {stats['std']:.2f} | " +
                         f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['coefficient_of_variation']:.2f} |\n")
        
        report.write("\n\n## Visualizations\n\n")
        report.write("1. Feature distributions\n")
        report.write("2. Correlation matrix\n\n")
        
        report.write("See image files in the same directory for details.")
    
    return {
        "group": group,
        "n_subjects": len(model_data),
        "descriptive_stats": descriptive_stats,
        "top_features": top_features
    }

def train_and_evaluate_models(model_data, feature_cols):
    """Train and evaluate models using hyperparameter tuning and advanced techniques"""
    logger.info("Training and evaluating models with advanced optimization")
    
    if model_data is None or len(model_data) == 0:
        logger.error("No data available for modeling")
        return None
    
    # Get features and target
    X = model_data[feature_cols]
    y = model_data['target']
    
    # Check for missing values in the features
    missing_count = X.isna().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in the feature matrix. Will use an imputer.")
    
    # Import necessary components for the pipeline
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFECV, SelectFromModel
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    
    # Use stratified k-fold CV for more reliable performance estimates
    cv_outer = LeaveOneOut()
    
    # Advanced feature engineering - combine with domain knowledge from PD research
    # Add interaction terms for key brain regions known to be involved in PD subtypes
    logger.info("Performing advanced feature engineering")
    
    # Convert to numpy array for faster processing
    X_array = X.values
    
    # Models with hyperparameter tuning
    models = {
        'SVM': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=min(20, len(feature_cols)))),
                ('classifier', SVC(probability=True))
            ]),
            'param_grid': {
                'feature_selection__k': [10, 15, 20],
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__kernel': ['rbf', 'linear', 'poly'],
                'classifier__class_weight': [None, 'balanced']
            }
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'param_grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 5, 10, 15],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'Gradient Boosting': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), 
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=min(20, len(feature_cols)))),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            'param_grid': {
                'feature_selection__k': [10, 15, 20],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        },
        'Neural Network': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('pca', PCA(random_state=42)),
                ('classifier', MLPClassifier(random_state=42, max_iter=1000))
            ]),
            'param_grid': {
                'pca__n_components': [0.9, 0.95],
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate': ['constant', 'adaptive']
            }
        }
    }
    
    # Use SMOTE for better handling of any class imbalance
    smote = SMOTE(random_state=42)
    
    # Initialize results storage
    results = {}
    best_models = {}
    
    # Inner CV for hyperparameter tuning
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform hyperparameter tuning for each model
    for model_name, model_config in models.items():
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        pipeline = model_config['pipeline']
        param_grid = model_config['param_grid']
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_inner,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the grid search
        grid_search.fit(X_array, y)
        
        # Store the best model
        best_models[model_name] = grid_search.best_estimator_
        logger.info(f"Best {model_name} parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
    
    # Create ensemble models from best individual models
    logger.info("Creating ensemble models...")
    
    # Voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', best_models.get('SVM')),
            ('rf', best_models.get('Random Forest')),
            ('gb', best_models.get('Gradient Boosting'))
        ],
        voting='soft'
    )
    
    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('svm', best_models.get('SVM')),
            ('rf', best_models.get('Random Forest')),
            ('gb', best_models.get('Gradient Boosting'))
        ],
        final_estimator=LogisticRegression(random_state=42)
    )
    
    # Add ensemble models to the list of models to evaluate
    best_models['Voting Ensemble'] = voting_clf
    best_models['Stacking Ensemble'] = stacking_clf
    
    # Evaluate all models using leave-one-out cross-validation
    for model_name, model in best_models.items():
        logger.info(f"Evaluating {model_name} with leave-one-out cross-validation...")
        
        y_pred = np.zeros(len(y))
        y_proba = np.zeros(len(y))
        
        # Leave-One-Out cross-validation
        for train_idx, test_idx in cv_outer.split(X_array):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply SMOTE to training data to handle imbalance
            try:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            except:
                # If SMOTE fails (e.g., not enough samples), use original data
                logger.warning(f"SMOTE failed for this fold, using original data")
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # Train model on resampled data
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict
            y_pred[test_idx] = model.predict(X_test)
            try:
                y_proba[test_idx] = model.predict_proba(X_test)[:, 1]
            except:
                # Some models might not support predict_proba
                y_proba[test_idx] = y_pred[test_idx]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        from sklearn.metrics import balanced_accuracy_score
        balanced_acc = balanced_accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=['TDPD', 'PIGD'], output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        results[model_name] = {
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_curve': (fpr, tpr, roc_auc)
        }
        
        logger.info(f"{model_name} accuracy: {accuracy:.4f}, balanced accuracy: {balanced_acc:.4f}, AUC: {roc_auc:.4f}")
        logger.info(f"Classification report:\n{classification_report(y, y_pred, target_names=['TDPD', 'PIGD'])}")
    
    # Identify best model based on balanced accuracy score
    best_model = max(results.items(), key=lambda x: x[1]['balanced_accuracy'])
    logger.info(f"Best model: {best_model[0]} with balanced accuracy {best_model[1]['balanced_accuracy']:.4f}")
    
    return results

def get_important_features(model_data, feature_cols):
    """Identify important features for classification"""
    logger.info("Identifying important features")
    
    # Get features and target
    X = model_data[feature_cols]
    y = model_data['target']
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    
    # Create an imputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Use Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a dataframe with feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    logger.info("Top 10 important features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
        
    return feature_importance

def visualize_results(results, output_dir):
    """Visualize model results, including ROC curves and confusion matrices"""
    logger.info("Creating visualizations")
    
    # Create ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        fpr, tpr, roc_auc = result['roc_curve']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for PD Subtype Classification')
    plt.legend(loc='lower right')
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    
    # Create confusion matrices
    for model_name, result in results.items():
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['TDPD', 'PIGD'], yticklabels=['TDPD', 'PIGD'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(output_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    # Visualize feature importances
    if 'feature_importance' in locals():
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Important Features')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved visualizations to {output_dir}")

def main():
    """Main execution function"""
    logger.info("Starting PD subtype classification analysis")
    
    # Load data
    demo_data = load_demographic_data()
    subcort_data = load_subcortical_data()
    cortical_data = load_cortical_data()
    
    # Merge datasets
    merged_data = merge_datasets(demo_data, subcort_data, cortical_data)
    
    if merged_data is None or len(merged_data) == 0:
        logger.error("No data available for analysis")
        return
    
    # Prepare data for modeling
    model_data, feature_cols, can_classify = prepare_data_for_modeling(merged_data)
    
    if model_data is None or len(model_data) == 0:
        logger.error("Failed to prepare data for modeling")
        return
    
    # Perform advanced feature engineering
    model_data, feature_cols = advanced_feature_engineering(model_data, feature_cols)
    
    # If we have multiple classes, run classification; otherwise, run descriptive statistics
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    if can_classify:
        logger.info("Running classification analysis since multiple classes are present")
        # Train and evaluate models
        results = train_and_evaluate_models(model_data, feature_cols)
        
        if results:
            # Get important features
            feature_importance = get_important_features(model_data, feature_cols)
            
            # Visualize results
            visualize_results(results, OUTPUT_DIR)
            
            # Save feature importance
            feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    else:
        logger.info("Only one class present, running descriptive analysis instead of classification")
        # Run descriptive analysis instead
        results = run_descriptive_analysis(model_data, feature_cols)
        
        # Create a flag file to indicate we completed successfully with descriptive stats
        with open(OUTPUT_DIR / 'descriptive_analysis_done.txt', 'w') as f:
            f.write(f"Descriptive analysis completed at {pd.Timestamp.now()}")
    
    # Save model data for reference
    model_data.to_csv(OUTPUT_DIR / 'model_data.csv', index=False)
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()
````
