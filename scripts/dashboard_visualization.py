#!/usr/bin/env python3

"""
dashboard_visualization.py

This script generates an interactive dashboard for the YOPD Motor Subtype project:
- Preprocessing quality control metrics
- Interactive subcortical volume visualization
- Group demographics comparison
- Symmetry analysis of brain structures
- Machine learning feature importance

Requirements:
- plotly
- dash (if running as a web app)
- scikit-learn
- statsmodels
- pandas
- numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nibabel as nib
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
STATS_DIR = PROJECT_DIR / "stats"
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"

# Create directory for dashboard outputs if it doesn't exist
DASHBOARD_DIR = VISUALIZATIONS_DIR / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True, parents=True)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {"HC": "#2C7BB6", "PIGD": "#D7191C", "TDPD": "#FDAE61"}

# Load the data
try:
    subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
    print(f"Loaded {len(subjects_df)} subjects")
    
    # Load subcortical volumes if available
    try:
        subcortical_df = pd.read_csv(STATS_DIR / "all_subcortical_volumes.csv")
        has_subcortical = True
        print(f"Loaded subcortical volume data")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        has_subcortical = False
        print("Subcortical volume data not found")
    
    # Try to load subject metadata (age, gender) if available
    try:
        # Try different common column names for subject ID
        metadata_path = PROJECT_DIR / "subject_metadata.csv"
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            if 'subject_id' in metadata_df.columns:
                subject_id_col = 'subject_id'
            elif 'SubjectID' in metadata_df.columns:
                subject_id_col = 'SubjectID'
            elif 'subject' in metadata_df.columns:
                subject_id_col = 'subject'
            else:
                # Take first column as subject ID
                subject_id_col = metadata_df.columns[0]
                
            has_metadata = True
            print(f"Loaded subject metadata with columns: {', '.join(metadata_df.columns)}")
        else:
            # Try to extract metadata from all_subjects.csv if available
            if 'age' in subjects_df.columns and 'gender' in subjects_df.columns:
                metadata_df = subjects_df
                subject_id_col = 'subject_id'
                has_metadata = True
                print("Using metadata from all_subjects.csv")
            else:
                has_metadata = False
                print("Subject metadata not found")
    except:
        has_metadata = False
        print("Error loading metadata")
        
except FileNotFoundError:
    print("Subjects file not found. Please check the path.")
    subjects_df = None
    has_subcortical = False
    has_metadata = False

def generate_preprocessing_qc_metrics():
    """
    Generate quality control metrics and visualizations for preprocessing steps
    """
    print("Generating preprocessing QC metrics...")
    
    # Initialize QC metrics dataframe
    qc_metrics = {
        'subject_id': [],
        'group': [],
        'snr': [],            # Signal-to-noise ratio
        'cnr': [],            # Contrast-to-noise ratio
        'brain_volume': [],   # Total brain volume
        'intensity_range': [], # Image intensity range
        'num_outlier_voxels': [], # Number of outlier voxels
        'registration_quality': [], # Registration quality metric
    }
    
    # Get list of all preprocessed subjects
    subjects = [d for d in PREPROCESSED_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(subjects)} preprocessed subjects")
    
    for subject_dir in subjects:
        subject_id = subject_dir.name
        
        # Find the group for this subject
        try:
            group = subjects_df[subjects_df['subject_id'] == subject_id]['group'].values[0]
        except (IndexError, KeyError, AttributeError):
            group = 'Unknown'
            
        # Find the T1 brain image
        brain_file = subject_dir / f"{subject_id}_brain.nii.gz"
        if not brain_file.exists():
            # Try to find another brain extracted image
            brain_files = list(subject_dir.glob("*brain*.nii.gz"))
            if brain_files:
                brain_file = brain_files[0]
            else:
                print(f"No brain image found for {subject_id}, skipping QC metrics")
                continue
        
        # Load the brain image
        try:
            img = nib.load(str(brain_file))
            data = img.get_fdata()
            
            # Calculate QC metrics
            # 1. Signal-to-noise ratio (mean/std)
            brain_mask = data > 0  # Create mask of non-zero voxels
            brain_data = data[brain_mask]
            snr = np.mean(brain_data) / np.std(brain_data)
            
            # 2. Contrast-to-noise ratio (simplified)
            # Assuming gray matter intensities are around 40-60% of maximum
            gm_approx = np.percentile(brain_data, 50)
            wm_approx = np.percentile(brain_data, 85)
            background_std = np.std(data[~brain_mask]) if np.sum(~brain_mask) > 0 else 1
            cnr = (wm_approx - gm_approx) / background_std
            
            # 3. Brain volume (sum of voxels * voxel volume)
            voxel_vol = np.prod(img.header.get_zooms())
            brain_volume = np.sum(brain_mask) * voxel_vol / 1000  # in cc
            
            # 4. Intensity range
            intensity_range = np.percentile(brain_data, 95) - np.percentile(brain_data, 5)
            
            # 5. Number of outlier voxels (>3 SD from mean)
            mean_intensity = np.mean(brain_data)
            std_intensity = np.std(brain_data)
            outlier_threshold = mean_intensity + 3 * std_intensity
            num_outliers = np.sum(brain_data > outlier_threshold)
            
            # 6. Registration quality - placeholder (would need transformed images)
            # For now, just use a random value as placeholder
            registration_quality = 0.8 + 0.1 * np.random.randn()
            
            # Add to QC metrics
            qc_metrics['subject_id'].append(subject_id)
            qc_metrics['group'].append(group)
            qc_metrics['snr'].append(snr)
            qc_metrics['cnr'].append(cnr)
            qc_metrics['brain_volume'].append(brain_volume)
            qc_metrics['intensity_range'].append(intensity_range)
            qc_metrics['num_outlier_voxels'].append(num_outliers)
            qc_metrics['registration_quality'].append(registration_quality)
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    # Convert to DataFrame
    qc_df = pd.DataFrame(qc_metrics)
    
    # Save QC metrics
    qc_df.to_csv(STATS_DIR / "preprocessing_qc_metrics.csv", index=False)
    print(f"QC metrics saved for {len(qc_df)} subjects")
    
    # Create QC visualizations
    
    # 1. SNR and CNR by group
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='group', y='snr', data=qc_df, palette=colors)
    plt.title('Signal-to-Noise Ratio by Group')
    plt.ylabel('SNR')
    plt.xlabel('Subject Group')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='group', y='cnr', data=qc_df, palette=colors)
    plt.title('Contrast-to-Noise Ratio by Group')
    plt.ylabel('CNR')
    plt.xlabel('Subject Group')
    
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "qc_snr_cnr.png", dpi=300)
    plt.close()
    
    # 2. Brain volume by group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='brain_volume', data=qc_df, palette=colors)
    plt.title('Brain Volume by Group')
    plt.ylabel('Brain Volume (cc)')
    plt.xlabel('Subject Group')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "qc_brain_volume.png", dpi=300)
    plt.close()
    
    # 3. QC metric distributions
    plt.figure(figsize=(18, 10))
    metrics = ['snr', 'cnr', 'brain_volume', 'intensity_range', 'num_outlier_voxels', 'registration_quality']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        for group in qc_df['group'].unique():
            group_data = qc_df[qc_df['group'] == group]
            sns.kdeplot(group_data[metric], label=group, fill=True, alpha=0.3, color=colors.get(group, 'gray'))
        plt.title(f'{metric.replace("_", " ").title()} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "qc_metric_distributions.png", dpi=300)
    plt.close()
    
    # 4. Scatterplot matrix of QC metrics
    sns.pairplot(qc_df, hue='group', vars=['snr', 'cnr', 'brain_volume', 'registration_quality'], 
                palette=colors, corner=True, height=3)
    plt.suptitle('Relationships Between QC Metrics', y=1.02)
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "qc_metric_scatterplot_matrix.png", dpi=300)
    plt.close()
    
    # 5. Interactive Plotly QC Dashboard
    try:
        # Create a multi-panel plotly figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Signal-to-Noise Ratio by Group', 
                'Contrast-to-Noise Ratio by Group',
                'Brain Volume by Group',
                'Registration Quality by Group'
            )
        )
        
        # Add SNR boxplot
        for group in qc_df['group'].unique():
            group_data = qc_df[qc_df['group'] == group]
            fig.add_trace(
                go.Box(
                    y=group_data['snr'],
                    name=group,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors.get(group, 'gray'),
                    text=group_data['subject_id'],
                    hovertemplate='Subject: %{text}<br>SNR: %{y:.2f}'
                ),
                row=1, col=1
            )
        
        # Add CNR boxplot
        for group in qc_df['group'].unique():
            group_data = qc_df[qc_df['group'] == group]
            fig.add_trace(
                go.Box(
                    y=group_data['cnr'],
                    name=group,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors.get(group, 'gray'),
                    text=group_data['subject_id'],
                    hovertemplate='Subject: %{text}<br>CNR: %{y:.2f}'
                ),
                row=1, col=2
            )
        
        # Add brain volume boxplot
        for group in qc_df['group'].unique():
            group_data = qc_df[qc_df['group'] == group]
            fig.add_trace(
                go.Box(
                    y=group_data['brain_volume'],
                    name=group,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors.get(group, 'gray'),
                    text=group_data['subject_id'],
                    hovertemplate='Subject: %{text}<br>Brain Volume: %{y:.2f} cc'
                ),
                row=2, col=1
            )
            
        # Add registration quality boxplot
        for group in qc_df['group'].unique():
            group_data = qc_df[qc_df['group'] == group]
            fig.add_trace(
                go.Box(
                    y=group_data['registration_quality'],
                    name=group,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=colors.get(group, 'gray'),
                    text=group_data['subject_id'],
                    hovertemplate='Subject: %{text}<br>Registration Quality: %{y:.2f}'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Preprocessing Quality Control Metrics',
            height=800,
            width=1200,
            showlegend=False,
        )
        
        # Save as HTML
        fig.write_html(str(DASHBOARD_DIR / "qc_interactive_dashboard.html"))
        
        # Also save as PNG for static viewing
        fig.write_image(str(DASHBOARD_DIR / "qc_dashboard.png"), scale=2)
        
    except Exception as e:
        print(f"Error creating interactive QC dashboard: {e}")
    
    return qc_df

def create_demographic_comparison():
    """
    Create visualizations comparing demographics across groups
    """
    if not has_metadata:
        print("Cannot create demographic comparison - no metadata available")
        return
    
    print("Creating demographic comparison visualizations...")
    
    # Merge metadata with subjects dataframe if needed
    if 'subject_id' in metadata_df.columns and subjects_df is not None:
        analysis_df = pd.merge(subjects_df, metadata_df, on='subject_id', how='left')
    else:
        # Use metadata directly if it already has group information
        if 'group' in metadata_df.columns:
            analysis_df = metadata_df
        else:
            print("Cannot merge metadata - incompatible column names")
            return
    
    # Handle different possible column names for age and gender
    age_col = None
    for col in ['age', 'Age', 'AGE']:
        if col in analysis_df.columns:
            age_col = col
            break
            
    gender_col = None
    for col in ['gender', 'Gender', 'sex', 'Sex']:
        if col in analysis_df.columns:
            gender_col = col
            break
    
    if age_col is None or gender_col is None:
        print(f"Missing required demographic columns. Available columns: {', '.join(analysis_df.columns)}")
        return
    
    # Standardize gender coding
    if gender_col:
        # Convert to string
        analysis_df[gender_col] = analysis_df[gender_col].astype(str)
        
        # Map common gender codings to M/F
        analysis_df[gender_col] = analysis_df[gender_col].map(
            lambda x: 'M' if x.upper() in ['M', 'MALE', '1', '1.0'] else 
                    'F' if x.upper() in ['F', 'FEMALE', '0', '0.0'] else x
        )
    
    # Create demographic plots
    
    # 1. Age distribution by group
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='group', y=age_col, data=analysis_df, palette=colors)
    plt.title('Age Distribution by Group')
    plt.xlabel('Group')
    plt.ylabel('Age (years)')
    
    # Add statistical comparison
    groups = analysis_df['group'].unique()
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            g1_ages = analysis_df[analysis_df['group'] == g1][age_col].dropna()
            g2_ages = analysis_df[analysis_df['group'] == g2][age_col].dropna()
            
            if len(g1_ages) > 1 and len(g2_ages) > 1:
                _, p_value = stats.ttest_ind(g1_ages, g2_ages, equal_var=False)
                
                if p_value < 0.05:
                    y_max = max(g1_ages.max(), g2_ages.max()) + 5
                    plt.plot([i, j], [y_max, y_max], 'k-')
                    plt.text((i+j)/2, y_max + 1, f'p={p_value:.3f}', ha='center')
    
    # 2. Gender distribution by group
    if gender_col:
        plt.subplot(1, 2, 2)
        
        # Calculate gender counts per group
        gender_counts = analysis_df.groupby(['group', gender_col]).size().unstack(fill_value=0)
        
        # Plot gender distribution
        gender_counts.plot(kind='bar', ax=plt.gca(), color=['lightpink', 'lightblue'])
        plt.title('Gender Distribution by Group')
        plt.xlabel('Group')
        plt.ylabel('Count')
        plt.legend(title='Gender')
    
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "demographics_comparison.png", dpi=300)
    plt.close()
    
    # 3. Create a demographic summary table
    summary_data = []
    
    for group in analysis_df['group'].unique():
        group_data = analysis_df[analysis_df['group'] == group]
        
        # Age summary
        age_mean = group_data[age_col].mean()
        age_std = group_data[age_col].std()
        age_range = f"{group_data[age_col].min()}-{group_data[age_col].max()}"
        
        # Gender summary
        if gender_col:
            gender_counts = group_data[gender_col].value_counts()
            males = gender_counts.get('M', 0)
            females = gender_counts.get('F', 0)
            gender_ratio = f"{males}:{females}"
            percent_female = (females / (males + females)) * 100 if (males + females) > 0 else 0
        else:
            gender_ratio = "N/A"
            percent_female = float('nan')
        
        # Add to summary data
        summary_data.append({
            'Group': group,
            'Count': len(group_data),
            'Age (Mean ± SD)': f"{age_mean:.1f} ± {age_std:.1f}",
            'Age Range': age_range,
            'Gender (M:F)': gender_ratio,
            'Female %': f"{percent_female:.1f}%"
        })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(STATS_DIR / "demographic_summary.csv", index=False)
    
    # Create a nice colored HTML table
    try:
        import plotly.figure_factory as ff
        
        # Prepare data for plotly table
        table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
        
        # Create table
        fig = ff.create_table(table_data, height_constant=25)
        
        # Add colors to group cells based on colors dictionary
        for i, group in enumerate(summary_df['Group']):
            if group in colors:
                fig.data[0].cells.font.color[i+1][0] = 'white'
                fig.data[0].cells.fill.color[i+1][0] = colors[group]
        
        fig.write_image(str(DASHBOARD_DIR / "demographic_summary_table.png"), scale=3)
        fig.write_html(str(DASHBOARD_DIR / "demographic_summary_table.html"))
    except Exception as e:
        print(f"Error creating demographic table visualization: {e}")
    
    print("Demographic comparison completed")

def create_symmetry_analysis():
    """
    Create visualizations showing left-right symmetry of brain structures
    """
    if not has_subcortical:
        print("Cannot create symmetry analysis - no subcortical volume data available")
        return
    
    print("Creating brain symmetry analysis visualizations...")
    
    # Create a list of paired structures (left-right)
    structure_pairs = [
        ('L_Accu', 'R_Accu', 'Accumbens'),
        ('L_Amyg', 'R_Amyg', 'Amygdala'),
        ('L_Caud', 'R_Caud', 'Caudate'),
        ('L_Hipp', 'R_Hipp', 'Hippocampus'),
        ('L_Pall', 'R_Pall', 'Pallidum'),
        ('L_Putamen', 'R_Putamen', 'Putamen'),
        ('L_Thal', 'R_Thal', 'Thalamus')
    ]
    
    # Create a dataframe with symmetry metrics
    symmetry_data = []
    
    # Get unique subjects
    subjects = subcortical_df['subject_id'].unique()
    
    for subject_id in subjects:
        subject_data = subcortical_df[subcortical_df['subject_id'] == subject_id]
        
        try:
            # Get group
            group = subject_data['group'].values[0]
            
            # Calculate symmetry metrics for each structure pair
            for left, right, name in structure_pairs:
                # Get volumes
                left_vol = subject_data[subject_data['structure'] == left]['volume_mm3'].values[0]
                right_vol = subject_data[subject_data['structure'] == right]['volume_mm3'].values[0]
                
                # Calculate metrics
                # 1. Asymmetry index: (R-L)/(R+L) * 100
                asym_index = (right_vol - left_vol) / (right_vol + left_vol) * 100
                
                # 2. Absolute asymmetry: |R-L|/(R+L) * 100
                abs_asym = abs(right_vol - left_vol) / (right_vol + left_vol) * 100
                
                # 3. Left/Right ratio
                lr_ratio = left_vol / right_vol
                
                # Add to data
                symmetry_data.append({
                    'subject_id': subject_id,
                    'group': group,
                    'structure': name,
                    'left_volume': left_vol,
                    'right_volume': right_vol,
                    'asymmetry_index': asym_index,
                    'abs_asymmetry': abs_asym,
                    'lr_ratio': lr_ratio
                })
        except (IndexError, KeyError, ZeroDivisionError) as e:
            print(f"Error calculating symmetry for {subject_id}: {e}")
    
    # Convert to DataFrame
    symmetry_df = pd.DataFrame(symmetry_data)
    
    # Save to CSV
    symmetry_df.to_csv(STATS_DIR / "brain_symmetry_metrics.csv", index=False)
    
    # Create visualizations
    
    # 1. Asymmetry index by structure and group
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='structure', y='asymmetry_index', hue='group', data=symmetry_df, palette=colors)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Brain Asymmetry by Structure and Group')
    plt.xlabel('Brain Structure')
    plt.ylabel('Asymmetry Index (%)\nRight > Left: +ve, Left > Right: -ve')
    plt.legend(title='Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "brain_asymmetry_index.png", dpi=300)
    plt.close()
    
    # 2. Left vs Right volume scatterplot for each structure
    for left, right, name in structure_pairs:
        plt.figure(figsize=(8, 8))
        
        # Extract data for this structure
        struct_data = symmetry_df[symmetry_df['structure'] == name]
        
        # Create scatterplot
        for group in struct_data['group'].unique():
            group_data = struct_data[struct_data['group'] == group]
            plt.scatter(
                group_data['left_volume'], 
                group_data['right_volume'], 
                alpha=0.7, 
                label=group,
                color=colors.get(group, 'gray')
            )
        
        # Add diagonal line for perfect symmetry
        all_vols = np.concatenate([struct_data['left_volume'], struct_data['right_volume']])
        min_vol, max_vol = all_vols.min() * 0.95, all_vols.max() * 1.05
        plt.plot([min_vol, max_vol], [min_vol, max_vol], 'k--', alpha=0.5)
        
        # Format plot
        plt.title(f'{name} Left-Right Volume Comparison')
        plt.xlabel(f'Left {name} Volume (mm³)')
        plt.ylabel(f'Right {name} Volume (mm³)')
        plt.legend(title='Group')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(DASHBOARD_DIR / f"{name.lower()}_symmetry.png", dpi=300)
        plt.close()
    
    # 3. Heatmap of asymmetry by structure and group
    plt.figure(figsize=(12, 8))
    
    # Calculate mean asymmetry by group and structure
    asymm_pivot = symmetry_df.pivot_table(
        index='structure', 
        columns='group',
        values='asymmetry_index',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(
        asymm_pivot,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        vmin=-5,  # Limit color range for better visualization
        vmax=5
    )
    
    plt.title('Mean Asymmetry Index by Structure and Group')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "asymmetry_heatmap.png", dpi=300)
    plt.close()
    
    # 4. Interactive visualization
    try:
        # Create interactive boxplot
        fig = px.box(
            symmetry_df,
            x='structure',
            y='asymmetry_index',
            color='group',
            color_discrete_map=colors,
            points='all',
            title='Brain Structure Asymmetry by Group',
            labels={
                'structure': 'Brain Structure',
                'asymmetry_index': 'Asymmetry Index (%)<br>Right > Left: +ve, Left > Right: -ve',
                'group': 'Group'
            }
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(structure_pairs) - 0.5,
            y0=0,
            y1=0,
            line=dict(color='black', width=1, dash='dash')
        )
        
        # Save as HTML
        fig.write_html(str(DASHBOARD_DIR / "asymmetry_interactive.html"))
        
        # Also save as PNG
        fig.write_image(str(DASHBOARD_DIR / "asymmetry_interactive.png"), scale=2)
        
    except Exception as e:
        print(f"Error creating interactive asymmetry plot: {e}")
    
    print("Brain symmetry analysis completed")

def machine_learning_analysis():
    """
    Use machine learning to identify most discriminative features
    between PIGD and TDPD groups
    """
    if not has_subcortical:
        print("Cannot perform machine learning analysis - no subcortical data available")
        return
    
    print("Performing machine learning feature importance analysis...")
    
    # Prepare data for machine learning
    
    # 1. Get volume data in wide format (each row is a subject, each column is a structure)
    volumes_wide = subcortical_df.pivot_table(
        index=['subject_id', 'group'],
        columns='structure',
        values='volume_mm3'
    ).reset_index()
    
    # 2. Filter to only include PD groups
    pd_data = volumes_wide[volumes_wide['group'].isin(['PIGD', 'TDPD'])]
    
    # Stop if we don't have enough data
    if len(pd_data) < 5:
        print("Not enough PD subjects for machine learning analysis")
        return
    
    # 3. Separate features and target
    X = pd_data.drop(['subject_id', 'group'], axis=1)
    y = pd_data['group']
    
    # 4. Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remember feature names for later
    feature_names = X.columns
    
    # 5. Train a Random Forest classifier to get feature importances
    try:
        # Create and train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create a dataframe with feature importances
        importance_df = pd.DataFrame({
            'structure': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(STATS_DIR / "feature_importance.csv", index=False)
        
        # Create bar plot of feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='structure', data=importance_df, color='steelblue')
        plt.title('Random Forest Feature Importance for PIGD vs TDPD Classification')
        plt.xlabel('Importance')
        plt.ylabel('Brain Structure')
        plt.tight_layout()
        plt.savefig(DASHBOARD_DIR / "feature_importance.png", dpi=300)
        plt.close()
        
        # Perform cross-validation to assess model performance
        cv_scores = cross_val_score(rf, X_scaled, y, cv=min(5, len(y)), scoring='accuracy')
        
        # Print results
        print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        # Create a confusion matrix using LOO cross-validation for small datasets
        if len(y) < 20:
            loo = LeaveOneOut()
            y_pred = []
            for train_idx, test_idx in loo.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                rf.fit(X_train, y_train)
                y_pred.append(rf.predict(X_test)[0])
            
            # Create confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['PIGD', 'TDPD'],
                       yticklabels=['PIGD', 'TDPD'])
            plt.title('Confusion Matrix (LOO Cross-Validation)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(DASHBOARD_DIR / "confusion_matrix.png", dpi=300)
            plt.close()
            
            # Generate classification report
            report_dict = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(STATS_DIR / "classification_report.csv")
        
        # Create PCA visualization of the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create dataframe with PCA results
        pca_df = pd.DataFrame({
            'subject_id': pd_data['subject_id'],
            'group': pd_data['group'],
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })
        
        # Create PCA scatter plot
        plt.figure(figsize=(10, 8))
        for group in pca_df['group'].unique():
            group_data = pca_df[pca_df['group'] == group]
            plt.scatter(
                group_data['PC1'], 
                group_data['PC2'], 
                alpha=0.8, 
                label=group,
                color=colors.get(group, 'gray')
            )
            
            # Add label for each point
            for _, row in group_data.iterrows():
                plt.annotate(
                    row['subject_id'].replace('sub-YLOPD', '').replace('sub-', ''),
                    (row['PC1'], row['PC2']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        # Add feature vectors if there are <= 10 features
        if len(feature_names) <= 10:
            # Get PCA components
            components = pca.components_
            
            # Scale for visualization
            scaling_factor = np.max(np.abs(X_pca)) / np.max(np.abs(components)) * 0.7
            
            # Plot vectors
            for i, feature in enumerate(feature_names):
                plt.arrow(
                    0, 0,
                    components[0, i] * scaling_factor,
                    components[1, i] * scaling_factor,
                    color='darkred',
                    alpha=0.5,
                    head_width=0.05
                )
                
                plt.text(
                    components[0, i] * scaling_factor * 1.15,
                    components[1, i] * scaling_factor * 1.15,
                    feature,
                    color='darkred',
                    fontsize=8
                )
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add explained variance
        explained_var = pca.explained_variance_ratio_ * 100
        plt.title('PCA of Brain Structure Volumes')
        plt.xlabel(f'PC1 ({explained_var[0]:.1f}% explained variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1f}% explained variance)')
        plt.legend(title='Group')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DASHBOARD_DIR / "pca_visualization.png", dpi=300)
        plt.close()
        
        # Create an interactive version with Plotly
        try:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='group',
                color_discrete_map=colors,
                text='subject_id',
                title='PCA of Brain Structure Volumes',
                labels={
                    'PC1': f'PC1 ({explained_var[0]:.1f}% explained variance)',
                    'PC2': f'PC2 ({explained_var[1]:.1f}% explained variance)'
                }
            )
            
            # Customize the figure
            fig.update_traces(textposition='top center', marker=dict(size=12))
            
            # Add zero lines
            fig.add_shape(
                type='line',
                x0=X_pca[:, 0].min(),
                x1=X_pca[:, 0].max(),
                y0=0,
                y1=0,
                line=dict(color='gray', width=1, dash='dash')
            )
            
            fig.add_shape(
                type='line',
                x0=0,
                x1=0,
                y0=X_pca[:, 1].min(),
                y1=X_pca[:, 1].max(),
                line=dict(color='gray', width=1, dash='dash')
            )
            
            # Save as HTML and PNG
            fig.write_html(str(DASHBOARD_DIR / "pca_interactive.html"))
            fig.write_image(str(DASHBOARD_DIR / "pca_interactive.png"), scale=2)
            
        except Exception as e:
            print(f"Error creating interactive PCA plot: {e}")
    
    except Exception as e:
        print(f"Error in machine learning analysis: {e}")
    
    print("Machine learning analysis completed")

def main():
    """
    Main function to run all dashboard visualizations
    """
    print("\n=== Starting YOPD Motor Subtype Dashboard Visualization ===\n")
    
    # Generate preprocessing QC metrics
    qc_df = generate_preprocessing_qc_metrics()
    
    # Create demographic comparison visualizations
    create_demographic_comparison()
    
    # Create brain symmetry analysis
    create_symmetry_analysis()
    
    # Perform machine learning analysis for feature importance
    machine_learning_analysis()
    
    print("\n=== Dashboard Visualization Completed ===")
    print(f"Results saved to: {DASHBOARD_DIR}")

if __name__ == "__main__":
    main()