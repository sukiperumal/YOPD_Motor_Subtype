#!/usr/bin/env python3

"""
visualize_results.py

This script generates visualizations and preliminary results from preprocessed T1 MRI data,
comparing brain structures across three subject groups:
- HC (Healthy Controls)
- PIGD (Postural Instability Gait Disorder Parkinson's Disease patients)
- TDPD (Tremor Dominant Parkinson's Disease patients)

The script provides:
1. Visualization of sample preprocessed T1 images from each group
2. Subcortical volume analysis with comparative statistics
3. Cortical thickness analysis and visualization
4. Brain structure differences visualization across groups
5. Summary statistics and tables
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
FIRST_RESULTS_DIR = PROJECT_DIR / "first_results"
FREESURFER_RESULTS_DIR = PROJECT_DIR / "freesurfer_results"
NILEARN_DIR = PROJECT_DIR / "nilearn_segmentation"
STATS_DIR = PROJECT_DIR / "stats"
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# Set up styles for consistent visualizations
plt.style.use('seaborn-v0_8-whitegrid')
colors = {"HC": "#2C7BB6", "PIGD": "#D7191C", "TDPD": "#FDAE61"}
color_palette = sns.color_palette([colors["HC"], colors["PIGD"], colors["TDPD"]])
sns.set_palette(color_palette)

# Read subject data
subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
print(f"Total subjects: {len(subjects_df)}")
print(f"Groups: {subjects_df['group'].value_counts().to_dict()}")

# Load subcortical volumes from nilearn_segmentation folder instead of stats folder
try:
    # Read the data from nilearn_segmentation
    nilearn_subcortical_df = pd.read_csv(NILEARN_DIR / "subcortical_volumes.csv")
    
    # Define mapping from nilearn column names to our structure naming convention
    structure_mapping = {
        'Left Accumbens': 'L_Accu',
        'Right Accumbens': 'R_Accu',
        'Left Amygdala': 'L_Amyg',
        'Right Amygdala': 'R_Amyg',
        'Left Caudate': 'L_Caud',
        'Right Caudate': 'R_Caud',
        'Left Hippocampus': 'L_Hipp',
        'Right Hippocampus': 'R_Hipp',
        'Left Pallidum': 'L_Pall',
        'Right Pallidum': 'R_Pall', 
        'Left Putamen': 'L_Putamen',
        'Right Putamen': 'R_Putamen',
        'Left Thalamus': 'L_Thal',
        'Right Thalamus': 'R_Thal',
        'Brain-Stem': 'BrStem'
    }
    
    # Transform the data into the format we need (long format)
    subcortical_data = []
    for _, row in nilearn_subcortical_df.iterrows():
        subject_id = row['subject_id']
        group = row['group']
        
        # Add 'sub-' prefix if not present to match the filenames in preprocessed directory
        if not subject_id.startswith('sub-'):
            subject_id = f"sub-{subject_id}"
        
        for orig_name, new_name in structure_mapping.items():
            if orig_name in row:
                volume = row[orig_name]
                subcortical_data.append({
                    'subject_id': subject_id,
                    'group': group,
                    'structure': new_name,
                    'volume_mm3': volume
                })
    
    # Create DataFrame in required format
    subcortical_df = pd.DataFrame(subcortical_data)
    has_subcortical_data = True
    
    print(f"Loaded subcortical volume data for {len(nilearn_subcortical_df)} subjects")
    print(f"Structures identified: {len(subcortical_df['structure'].unique())}")
    
    # Save the transformed data to the stats directory for future use
    subcortical_df.to_csv(STATS_DIR / "all_subcortical_volumes.csv", index=False)
    
except (FileNotFoundError, pd.errors.EmptyDataError) as e:
    has_subcortical_data = False
    subcortical_df = None
    print(f"Error loading subcortical volume data: {e}")
    print("Some visualizations will be skipped.")

# Define key brain structures for analysis
structures = {
    "subcortical": [
        "L_Thal", "R_Thal",     # Thalamus
        "L_Caud", "R_Caud",     # Caudate nucleus
        "L_Pall", "R_Pall",     # Pallidum
        "L_Putamen", "R_Putamen", # Putamen
        "L_Hipp", "R_Hipp",     # Hippocampus
        "L_Amyg", "R_Amyg",     # Amygdala
        "L_Accu", "R_Accu",     # Nucleus accumbens
        "BrStem"                # Brain stem
    ],
    "pigd_focus": ["L_Pall", "R_Pall", "BrStem"],  # Typically affected in PIGD
    "tdpd_focus": ["L_Thal", "R_Thal", "L_Caud", "R_Caud"]  # Typically affected in TDPD
}

def create_sample_t1_visualization():
    """
    Create a figure showing sample preprocessed T1 images from each subject group
    """
    print("Generating T1 visualization samples...")
    
    # Select one random subject from each group
    sample_subjects = {}
    for group in ["HC", "PIGD", "TDPD"]:
        group_subjects = subjects_df[subjects_df['group'] == group]['subject_id'].tolist()
        # Find a subject with preprocessed brain.nii.gz file
        for subject_id in group_subjects:
            brain_file = PREPROCESSED_DIR / subject_id / f"{subject_id}_brain.nii.gz"
            if brain_file.exists():
                sample_subjects[group] = (subject_id, brain_file)
                break
    
    if not sample_subjects:
        print("No sample T1 images found. Skipping T1 visualization.")
        return
    
    # Create figure with 3 rows (one for each group) and 3 columns (axial, sagittal, coronal views)
    fig = plt.figure(figsize=(15, 12))
    grid = gridspec.GridSpec(3, 3, figure=fig)
    
    row_idx = 0
    for group, (subject_id, brain_file) in sample_subjects.items():
        img = nib.load(str(brain_file))
        
        # Axial view
        ax1 = fig.add_subplot(grid[row_idx, 0])
        plotting.plot_anat(img, axes=ax1, display_mode='z', cut_coords=1, 
                          title=f"{group}: {subject_id} (Axial)", 
                          draw_cross=False)
        
        # Sagittal view - Changed cut_coords from 0 to 1 to avoid ValueError
        ax2 = fig.add_subplot(grid[row_idx, 1])
        plotting.plot_anat(img, axes=ax2, display_mode='x', cut_coords=1, 
                          title=f"{group}: {subject_id} (Sagittal)", 
                          draw_cross=False)
        
        # Coronal view - Changed cut_coords from 0 to 1 to avoid ValueError
        ax3 = fig.add_subplot(grid[row_idx, 2])
        plotting.plot_anat(img, axes=ax3, display_mode='y', cut_coords=1, 
                          title=f"{group}: {subject_id} (Coronal)",
                          draw_cross=False)
        
        row_idx += 1
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "sample_t1_images.png", dpi=300)
    plt.close()
    print("Sample T1 visualization saved.")

def visualize_subcortical_volumes():
    """
    Create visualizations of subcortical volumes across groups
    """
    if not has_subcortical_data:
        print("No subcortical volume data available. Skipping subcortical volume visualization.")
        return
    
    print("Generating subcortical volume visualizations...")
    
    # 1. Overall subcortical volume comparison across groups (boxplot)
    plt.figure(figsize=(15, 10))
    
    # Create a grouped boxplot for all structures
    ax = sns.boxplot(x='structure', y='volume_mm3', hue='group', data=subcortical_df,
                     palette=colors, hue_order=['HC', 'PIGD', 'TDPD'])
    
    # Add individual data points for transparency
    sns.stripplot(x='structure', y='volume_mm3', hue='group', data=subcortical_df,
                 palette=colors, dodge=True, size=4, alpha=0.3, 
                 hue_order=['HC', 'PIGD', 'TDPD'])
    
    plt.title("Subcortical Structure Volumes Across Subject Groups", fontsize=16)
    plt.xlabel("Brain Structure", fontsize=14)
    plt.ylabel("Volume (mm³)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Group", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "all_subcortical_volumes.png", dpi=300)
    plt.close()
    
    # 2. Focus on PIGD-specific structures
    plt.figure(figsize=(12, 7))
    pigd_data = subcortical_df[subcortical_df['structure'].isin(structures['pigd_focus'])]
    
    sns.boxplot(x='structure', y='volume_mm3', hue='group', data=pigd_data,
               palette=colors, hue_order=['HC', 'PIGD', 'TDPD'])
    
    sns.stripplot(x='structure', y='volume_mm3', hue='group', data=pigd_data,
                 palette=colors, dodge=True, size=4, alpha=0.3, 
                 hue_order=['HC', 'PIGD', 'TDPD'])
    
    plt.title("PIGD-Associated Subcortical Structure Volumes", fontsize=16)
    plt.xlabel("Brain Structure", fontsize=14)
    plt.ylabel("Volume (mm³)", fontsize=14)
    plt.legend(title="Group", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "pigd_subcortical_volumes.png", dpi=300)
    plt.close()
    
    # 3. Focus on TDPD-specific structures
    plt.figure(figsize=(12, 7))
    tdpd_data = subcortical_df[subcortical_df['structure'].isin(structures['tdpd_focus'])]
    
    sns.boxplot(x='structure', y='volume_mm3', hue='group', data=tdpd_data,
               palette=colors, hue_order=['HC', 'PIGD', 'TDPD'])
    
    sns.stripplot(x='structure', y='volume_mm3', hue='group', data=tdpd_data,
                 palette=colors, dodge=True, size=4, alpha=0.3,
                 hue_order=['HC', 'PIGD', 'TDPD'])
    
    plt.title("TDPD-Associated Subcortical Structure Volumes", fontsize=16)
    plt.xlabel("Brain Structure", fontsize=14)
    plt.ylabel("Volume (mm³)", fontsize=14)
    plt.legend(title="Group", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "tdpd_subcortical_volumes.png", dpi=300)
    plt.close()
    
    # 4. Statistical analysis and heatmap visualization
    print("Performing statistical analysis of subcortical volumes...")
    
    # Create a function to run statistical tests
    def run_volume_stats(data, structure_list, group1, group2):
        results = []
        
        for structure in structure_list:
            structure_data = data[data['structure'] == structure]
            group1_data = structure_data[structure_data['group'] == group1]['volume_mm3'].dropna()
            group2_data = structure_data[structure_data['group'] == group2]['volume_mm3'].dropna()
            
            # Skip if not enough data
            if len(group1_data) < 2 or len(group2_data) < 2:
                continue
                
            # Run Mann-Whitney U test (non-parametric)
            try:
                stat, pvalue = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                
                # Calculate effect size (Cohen's d)
                effect_size = (group1_data.mean() - group2_data.mean()) / np.sqrt(
                    ((len(group1_data) - 1) * group1_data.var() + 
                     (len(group2_data) - 1) * group2_data.var()) / 
                    (len(group1_data) + len(group2_data) - 2))
                
                results.append({
                    "structure": structure,
                    "group1": group1,
                    "group2": group2,
                    "group1_mean": group1_data.mean(),
                    "group2_mean": group2_data.mean(),
                    "percent_diff": ((group1_data.mean() - group2_data.mean()) / group2_data.mean()) * 100,
                    "effect_size": effect_size,
                    "p_value": pvalue
                })
            except:
                print(f"Statistical test failed for {structure} between {group1} and {group2}")
                continue
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Apply multiple comparison correction
            results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
            
            # Add significance markers
            results_df['significance'] = 'ns'
            results_df.loc[results_df['p_adjusted'] < 0.05, 'significance'] = '*'
            results_df.loc[results_df['p_adjusted'] < 0.01, 'significance'] = '**'
            results_df.loc[results_df['p_adjusted'] < 0.001, 'significance'] = '***'
            
            return results_df
        else:
            return None
    
    # Run statistics for each comparison
    all_structures = structures['subcortical']
    
    pigd_vs_hc_stats = run_volume_stats(subcortical_df, all_structures, "PIGD", "HC")
    tdpd_vs_hc_stats = run_volume_stats(subcortical_df, all_structures, "TDPD", "HC")
    pigd_vs_tdpd_stats = run_volume_stats(subcortical_df, all_structures, "PIGD", "TDPD")
    
    # Combine all stats for heatmap
    if pigd_vs_hc_stats is not None and tdpd_vs_hc_stats is not None and pigd_vs_tdpd_stats is not None:
        # Create heatmap of effect sizes
        effect_size_df = pd.DataFrame(index=all_structures)
        
        if pigd_vs_hc_stats is not None:
            pigd_vs_hc = pigd_vs_hc_stats.set_index('structure')['effect_size']
            effect_size_df['PIGD vs HC'] = pigd_vs_hc
        
        if tdpd_vs_hc_stats is not None:
            tdpd_vs_hc = tdpd_vs_hc_stats.set_index('structure')['effect_size']
            effect_size_df['TDPD vs HC'] = tdpd_vs_hc
        
        if pigd_vs_tdpd_stats is not None:
            pigd_vs_tdpd = pigd_vs_tdpd_stats.set_index('structure')['effect_size']
            effect_size_df['PIGD vs TDPD'] = pigd_vs_tdpd
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        sns.heatmap(effect_size_df, cmap='RdBu_r', center=0, 
                   vmin=-1.5, vmax=1.5, annot=True, fmt='.2f')
        plt.title("Effect Sizes for Group Comparisons\n(Cohen's d)", fontsize=16)
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_effect_sizes_heatmap.png", dpi=300)
        plt.close()
        
        # Save stats to CSV for future reference
        if pigd_vs_hc_stats is not None:
            pigd_vs_hc_stats.to_csv(STATS_DIR / "pigd_vs_hc_volume_stats.csv", index=False)
        
        if tdpd_vs_hc_stats is not None:
            tdpd_vs_hc_stats.to_csv(STATS_DIR / "tdpd_vs_hc_volume_stats.csv", index=False)
        
        if pigd_vs_tdpd_stats is not None:
            pigd_vs_tdpd_stats.to_csv(STATS_DIR / "pigd_vs_tdpd_volume_stats.csv", index=False)
    
    print("Subcortical volume visualizations and statistics completed.")

def visualize_group_differences():
    """
    Create radar/spider plots to visualize differences between groups across structures
    """
    if not has_subcortical_data:
        print("No subcortical volume data available. Skipping group differences visualization.")
        return
    
    print("Creating group difference visualizations...")
    
    # Calculate mean volumes for each structure by group
    structure_means = subcortical_df.groupby(['group', 'structure'])['volume_mm3'].mean().reset_index()
    
    # Create a pivot table for easier plotting
    pivot_means = structure_means.pivot(index='structure', columns='group', values='volume_mm3')
    
    # Select a subset of important structures for clarity
    key_structures = [
        'L_Thal', 'R_Thal',
        'L_Putamen', 'R_Putamen', 
        'L_Caud', 'R_Caud',
        'L_Pall', 'R_Pall',
        'BrStem'
    ]
    
    plot_data = pivot_means.loc[pivot_means.index.isin(key_structures)]
    
    # Create radar plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    categories = plot_data.index.tolist()
    N = len(categories)
    
    # Set angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot each group
    for group in ['HC', 'PIGD', 'TDPD']:
        if group in plot_data.columns:
            values = plot_data[group].tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=group, color=colors[group])
            ax.fill(angles, values, alpha=0.1, color=colors[group])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Label spacing
    plt.title('Subcortical Structure Volumes Across Groups', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "subcortical_radar_plot.png", dpi=300)
    plt.close()
    
    # Create percent difference plot (PIGD and TDPD compared to HC)
    plt.figure(figsize=(14, 8))
    
    # Calculate percent differences
    if 'HC' in plot_data.columns:
        hc_values = plot_data['HC']
        
        diff_data = pd.DataFrame(index=plot_data.index)
        
        if 'PIGD' in plot_data.columns:
            diff_data['PIGD vs HC'] = ((plot_data['PIGD'] - hc_values) / hc_values) * 100
            
        if 'TDPD' in plot_data.columns:
            diff_data['TDPD vs HC'] = ((plot_data['TDPD'] - hc_values) / hc_values) * 100
        
        # Horizontal bar chart for percent differences
        diff_data_melted = diff_data.reset_index().melt(id_vars='structure', 
                                                      var_name='Comparison', 
                                                      value_name='Percent Difference')
        
        # Create staggered bar chart
        g = sns.catplot(data=diff_data_melted, 
                      kind='bar',
                      x='Percent Difference', 
                      y='structure',
                      hue='Comparison',
                      palette=['#D7191C', '#FDAE61'],
                      height=8, aspect=1.5)
        
        # Add vertical line at 0%
        g.refline(x=0, color='black', linestyle='--')
        
        # Customize
        g.set_axis_labels("Percent Difference from HC (%)", "Brain Structure")
        g.legend.set_title("")
        plt.title('Volume Differences Compared to Healthy Controls', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(VISUALIZATIONS_DIR / "volume_percent_differences.png", dpi=300)
        plt.close()
    
    print("Group difference visualizations completed.")

def create_summary_dashboard():
    """
    Create a summary dashboard combining key visualizations
    """
    print("Creating summary dashboard...")
    
    plt.figure(figsize=(20, 16))
    
    gs = gridspec.GridSpec(2, 2)
    
    # 1. Top left: Sample T1 image from each group
    sample_subjects = {}
    for group in ["HC", "PIGD", "TDPD"]:
        group_subjects = subjects_df[subjects_df['group'] == group]['subject_id'].tolist()
        for subject_id in group_subjects:
            brain_file = PREPROCESSED_DIR / subject_id / f"{subject_id}_brain.nii.gz"
            if brain_file.exists():
                sample_subjects[group] = (subject_id, brain_file)
                break
    
    if sample_subjects:
        # Take one example subject
        example_group = list(sample_subjects.keys())[0]
        example_subject, example_file = sample_subjects[example_group]
        
        ax1 = plt.subplot(gs[0, 0])
        img = nib.load(str(example_file))
        plotting.plot_anat(img, axes=ax1, title=f"Sample Preprocessed T1 Image\n({example_group}: {example_subject})",
                         display_mode='z', cut_coords=1, draw_cross=False)
    
    # 2. Top right: Group sizes
    ax2 = plt.subplot(gs[0, 1])
    group_counts = subjects_df['group'].value_counts()
    sns.barplot(x=group_counts.index, y=group_counts.values, palette=colors, ax=ax2)
    ax2.set_title('Sample Size by Group', fontsize=16)
    ax2.set_xlabel('Group', fontsize=14)
    ax2.set_ylabel('Number of Subjects', fontsize=14)
    
    # 3. Bottom left: Key structures comparison (if data available)
    ax3 = plt.subplot(gs[1, 0])
    if has_subcortical_data:
        key_structures = ['L_Pall', 'R_Pall', 'L_Thal', 'R_Thal']
        key_data = subcortical_df[subcortical_df['structure'].isin(key_structures)]
        
        sns.boxplot(x='structure', y='volume_mm3', hue='group', data=key_data,
                   palette=colors, hue_order=['HC', 'PIGD', 'TDPD'], ax=ax3)
        
        ax3.set_title('Key Subcortical Structures Volume Comparison', fontsize=16)
        ax3.set_xlabel('Brain Structure', fontsize=14)
        ax3.set_ylabel('Volume (mm³)', fontsize=14)
        ax3.legend(title="Group")
    else:
        ax3.text(0.5, 0.5, 'Subcortical volume data not available', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14)
        ax3.axis('off')
    
    # 4. Bottom right: Study information
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    
    info_text = (
        "YOPD Motor Subtype Neuroimaging Study\n\n"
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"Total Subjects: {len(subjects_df)}\n"
        f"   - HC: {len(subjects_df[subjects_df['group'] == 'HC'])}\n"
        f"   - PIGD: {len(subjects_df[subjects_df['group'] == 'PIGD'])}\n"
        f"   - TDPD: {len(subjects_df[subjects_df['group'] == 'TDPD'])}\n\n"
        "Preprocessing Applied:\n"
        "   - Bias Field Correction\n"
        "   - Brain Extraction\n"
        "   - Registration to MNI Space\n\n"
        "Key Structures Examined:\n"
        "   - PIGD Focus: Pallidum, Brainstem\n"
        "   - TDPD Focus: Thalamus, Caudate\n"
    )
    
    ax4.text(0.1, 0.95, info_text, verticalalignment='top', 
             fontfamily='monospace', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "summary_dashboard.png", dpi=300)
    plt.close()
    
    print("Summary dashboard created.")

def main():
    """
    Main execution function
    """
    print("Starting visualization script...")
    print(f"Output directory: {VISUALIZATIONS_DIR}")
    
    # Create sample T1 image visualization
    create_sample_t1_visualization()
    
    # Visualize subcortical volumes if data available
    visualize_subcortical_volumes()
    
    # Create group difference visualizations
    visualize_group_differences()
    
    # Create summary dashboard
    create_summary_dashboard()
    
    print("All visualizations created successfully!")
    print(f"Results saved to: {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    main()