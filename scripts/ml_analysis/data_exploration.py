#!/usr/bin/env python3

"""
PD Subtype Data Exploration

This script provides exploratory data analysis and visualization for the YOPD_Motor_Subtype project.
It focuses on examining subcortical volumes, cortical thickness, and demographic differences between
HC, PIGD, and TDPD groups.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
OUTPUT_DIR = PROJECT_DIR / "ml_results" / "exploration"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / f'data_exploration_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pd_data_exploration')

def load_all_data():
    """Load all datasets for exploration"""
    logger.info("Loading all data sources")
    
    # Load subcortical volumes
    try:
        subcort_file = PROJECT_DIR / "stats" / "all_subcortical_volumes.csv"
        subcort_data = pd.read_csv(subcort_file)
        logger.info(f"Loaded subcortical data with {subcort_data['subject_id'].nunique()} subjects")
    except Exception as e:
        logger.error(f"Failed to load subcortical data: {e}")
        subcort_data = None
    
    # Try to load cortical thickness data
    try:
        thickness_file = PROJECT_DIR / "thickness_output" / "all_subjects_regional_thickness.csv"
        if thickness_file.exists():
            cortical_data = pd.read_csv(thickness_file)
            logger.info(f"Loaded cortical thickness data with {cortical_data['Subject'].nunique() if 'Subject' in cortical_data.columns else 'unknown'} subjects")
            
            # Ensure subject_id column exists
            if 'subject_id' not in cortical_data.columns and 'Subject' in cortical_data.columns:
                cortical_data['subject_id'] = cortical_data['Subject']
        else:
            logger.warning("Cortical thickness file not found")
            cortical_data = None
    except Exception as e:
        logger.error(f"Failed to load cortical thickness data: {e}")
        cortical_data = None
    
    # Try to load demographic data
    try:
        demo_file = PROJECT_DIR / "age_gender.xlsx"
        try:
            excel_data = pd.read_excel(demo_file, engine='openpyxl')
            logger.info(f"Loaded raw demographic data with {len(excel_data)} subjects")
            
            # Process the demographic data to create subject_id and group columns
            demo_data = process_demographic_data(excel_data)
            
            # If we have cortical thickness data but no group info, add it using the mapping from demo_data
            if cortical_data is not None and 'group' not in cortical_data.columns:
                # Create subject-to-group mapping dictionary
                subject_to_group = dict(zip(demo_data['subject_id'], demo_data['group']))
                
                # Add group information to cortical data
                cortical_data['group'] = cortical_data['subject_id'].map(subject_to_group)
                
                # Check if group was successfully added
                group_null_count = cortical_data['group'].isna().sum()
                if group_null_count > 0:
                    logger.warning(f"Could not assign group to {group_null_count} cortical thickness records")
                else:
                    logger.info("Successfully added group information to all cortical thickness data")
            
        except Exception as e:
            logger.warning(f"Failed to process Excel data: {e}")
            # If direct load fails, create demographic data from subcortical file
            if subcort_data is not None:
                logger.info("Creating demographic data from subcortical data")
                demo_data = subcort_data[['subject_id', 'group']].drop_duplicates().reset_index(drop=True)
                demo_data['age'] = np.nan
                demo_data['gender'] = np.nan
                
                # Also add group to cortical data if needed
                if cortical_data is not None and 'group' not in cortical_data.columns:
                    subject_to_group = dict(zip(demo_data['subject_id'], demo_data['group']))
                    cortical_data['group'] = cortical_data['subject_id'].map(subject_to_group)
            else:
                demo_data = None
    except Exception as e:
        logger.error(f"Failed to load demographic data: {e}")
        demo_data = None
    
    return subcort_data, cortical_data, demo_data

def process_demographic_data(excel_data):
    """Process the demographic Excel data to create a standardized format"""
    logger.info("Processing demographic data")
    
    # Create a proper subject_id and group format from the data
    subject_groups = {}
    
    # For each subject, determine their group (TDPD, PIGD, or HC) based on the flag columns
    for _, row in excel_data.iterrows():
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
    
    # Create a new dataframe with subject_id and group
    new_demo_data = pd.DataFrame({
        'subject_id': list(subject_groups.keys()),
        'group': list(subject_groups.values()),
        'age': excel_data['age_assessment'].values,
        'gender': excel_data['gender'].values
    })
    
    logger.info(f"Processed demographic data with {len(new_demo_data)} subjects")
    return new_demo_data

def explore_subcortical_volumes(subcort_data):
    """Explore subcortical volume data"""
    logger.info("Exploring subcortical volume data")
    
    if subcort_data is None:
        logger.error("No subcortical data available")
        return
    
    # Get basic stats by group and structure
    stats_by_group = subcort_data.groupby(['group', 'structure'])['volume_mm3'].agg(['mean', 'std', 'count']).reset_index()
    stats_by_group.to_csv(OUTPUT_DIR / 'subcortical_stats_by_group.csv', index=False)
    logger.info(f"Saved group statistics to {OUTPUT_DIR / 'subcortical_stats_by_group.csv'}")
    
    # Create visualizations
    # Group bar plot for key structures
    key_structures = ['L_Thal', 'R_Thal', 'L_Putamen', 'R_Putamen', 'L_Caud', 'R_Caud', 'BrStem']
    key_data = subcort_data[subcort_data['structure'].isin(key_structures)]
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='structure', y='volume_mm3', hue='group', data=key_data)
    plt.title('Subcortical Volumes by Group')
    plt.xlabel('Brain Structure')
    plt.ylabel('Volume (mm³)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subcortical_volumes_by_group.png', dpi=300)
    
    # Create a heatmap of volumes by structure and group
    pivot_data = subcort_data.pivot_table(index='structure', columns='group', values='volume_mm3', aggfunc='mean')
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.1f')
    plt.title('Mean Subcortical Volumes by Group')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subcortical_volumes_heatmap.png', dpi=300)
    
    # Statistical testing for group differences
    stat_results = []
    for structure in subcort_data['structure'].unique():
        structure_data = subcort_data[subcort_data['structure'] == structure]
        
        # PIGD vs HC
        pigd_data = structure_data[structure_data['group'] == 'PIGD']['volume_mm3']
        hc_data = structure_data[structure_data['group'] == 'HC']['volume_mm3']
        
        if len(pigd_data) > 0 and len(hc_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(pigd_data, hc_data, equal_var=False)
                effect_size = (pigd_data.mean() - hc_data.mean()) / np.sqrt((pigd_data.std()**2 + hc_data.std()**2) / 2)
                
                stat_results.append({
                    'structure': structure,
                    'comparison': 'PIGD_vs_HC',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {structure} PIGD vs HC: {e}")
        
        # TDPD vs HC
        tdpd_data = structure_data[structure_data['group'] == 'TDPD']['volume_mm3']
        
        if len(tdpd_data) > 0 and len(hc_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(tdpd_data, hc_data, equal_var=False)
                effect_size = (tdpd_data.mean() - hc_data.mean()) / np.sqrt((tdpd_data.std()**2 + hc_data.std()**2) / 2)
                
                stat_results.append({
                    'structure': structure,
                    'comparison': 'TDPD_vs_HC',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {structure} TDPD vs HC: {e}")
        
        # PIGD vs TDPD
        if len(pigd_data) > 0 and len(tdpd_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(pigd_data, tdpd_data, equal_var=False)
                effect_size = (pigd_data.mean() - tdpd_data.mean()) / np.sqrt((pigd_data.std()**2 + tdpd_data.std()**2) / 2)
                
                stat_results.append({
                    'structure': structure,
                    'comparison': 'PIGD_vs_TDPD',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {structure} PIGD vs TDPD: {e}")
    
    # Save statistical results
    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        stat_df.to_csv(OUTPUT_DIR / 'subcortical_statistical_tests.csv', index=False)
        
        # Visualize significant results
        sig_results = stat_df[stat_df['significant']]
        if len(sig_results) > 0:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='structure', y='effect_size', hue='comparison', data=sig_results)
            plt.title('Significant Group Differences in Subcortical Volumes')
            plt.xlabel('Brain Structure')
            plt.ylabel('Effect Size (Cohen\'s d)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'significant_subcortical_differences.png', dpi=300)
    
    logger.info("Completed subcortical volume exploration")

def explore_cortical_thickness(cortical_data):
    """Explore cortical thickness data"""
    logger.info("Exploring cortical thickness data")
    
    if cortical_data is None:
        logger.error("No cortical thickness data available")
        return
    
    # Check column names and standardize
    if 'Subject' in cortical_data.columns and 'subject_id' not in cortical_data.columns:
        cortical_data['subject_id'] = cortical_data['Subject']
    
    if 'Group' in cortical_data.columns and 'group' not in cortical_data.columns:
        cortical_data['group'] = cortical_data['Group']
    
    # Check required columns
    required_cols = ['subject_id', 'group', 'Region', 'Mean_Thickness']
    for col in required_cols:
        if col not in cortical_data.columns:
            logger.error(f"Required column {col} not found in cortical data")
            logger.info(f"Available columns: {cortical_data.columns}")
            return
    
    # Get basic stats by group and region
    stats_by_group = cortical_data.groupby(['group', 'Region'])['Mean_Thickness'].agg(['mean', 'std', 'count']).reset_index()
    stats_by_group.to_csv(OUTPUT_DIR / 'cortical_stats_by_group.csv', index=False)
    logger.info(f"Saved group statistics to {OUTPUT_DIR / 'cortical_stats_by_group.csv'}")
    
    # Key regions of interest for PD
    roi_regions = [
        'precentral', 'postcentral', 'superiorfrontal', 'caudalmiddlefrontal',
        'parsopercularis', 'parstriangularis', 'parsorbitalis', 'lateralorbitofrontal',
        'supramarginal', 'superiorparietal', 'inferiorparietal'
    ]
    
    # Find regions that match our regions of interest (case insensitive substring match)
    matching_regions = []
    for region in cortical_data['Region'].unique():
        for roi in roi_regions:
            if roi.lower() in region.lower():
                matching_regions.append(region)
                break
    
    # If no exact matches, use top regions by variance
    if not matching_regions:
        logger.warning("No matching ROI regions found, using top regions by variance")
        region_var = cortical_data.groupby('Region')['Mean_Thickness'].var().sort_values(ascending=False)
        matching_regions = region_var.head(10).index.tolist()
    
    # Create visualizations for regions of interest
    roi_data = cortical_data[cortical_data['Region'].isin(matching_regions)]
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Region', y='Mean_Thickness', hue='group', data=roi_data)
    plt.title('Cortical Thickness by Group for Regions of Interest')
    plt.xlabel('Brain Region')
    plt.ylabel('Thickness (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cortical_thickness_boxplot.png', dpi=300)
    
    # Create a heatmap of cortical thickness
    pivot_data = cortical_data.pivot_table(index='Region', columns='group', values='Mean_Thickness', aggfunc='mean')
    
    # Select a subset of regions for better visualization
    top_regions = pivot_data.std(axis=1).sort_values(ascending=False).head(15).index
    pivot_subset = pivot_data.loc[top_regions]
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_subset, cmap='RdBu_r', center=pivot_subset.values.mean(), annot=True, fmt='.3f')
    plt.title('Mean Cortical Thickness by Group (Top Variable Regions)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cortical_thickness_heatmap.png', dpi=300)
    
    # Statistical testing for group differences
    stat_results = []
    for region in cortical_data['Region'].unique():
        region_data = cortical_data[cortical_data['Region'] == region]
        
        # PIGD vs HC
        pigd_data = region_data[region_data['group'] == 'PIGD']['Mean_Thickness']
        hc_data = region_data[region_data['group'] == 'HC']['Mean_Thickness']
        
        if len(pigd_data) > 0 and len(hc_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(pigd_data, hc_data, equal_var=False)
                effect_size = (pigd_data.mean() - hc_data.mean()) / np.sqrt((pigd_data.std()**2 + hc_data.std()**2) / 2)
                
                stat_results.append({
                    'region': region,
                    'comparison': 'PIGD_vs_HC',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {region} PIGD vs HC: {e}")
        
        # TDPD vs HC
        tdpd_data = region_data[region_data['group'] == 'TDPD']['Mean_Thickness']
        
        if len(tdpd_data) > 0 and len(hc_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(tdpd_data, hc_data, equal_var=False)
                effect_size = (tdpd_data.mean() - hc_data.mean()) / np.sqrt((tdpd_data.std()**2 + hc_data.std()**2) / 2)
                
                stat_results.append({
                    'region': region,
                    'comparison': 'TDPD_vs_HC',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {region} TDPD vs HC: {e}")
        
        # PIGD vs TDPD
        if len(pigd_data) > 0 and len(tdpd_data) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(pigd_data, tdpd_data, equal_var=False)
                effect_size = (pigd_data.mean() - tdpd_data.mean()) / np.sqrt((pigd_data.std()**2 + tdpd_data.std()**2) / 2)
                
                stat_results.append({
                    'region': region,
                    'comparison': 'PIGD_vs_TDPD',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
            except Exception as e:
                logger.warning(f"Error in t-test for {region} PIGD vs TDPD: {e}")
    
    # Save statistical results
    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        stat_df.to_csv(OUTPUT_DIR / 'cortical_statistical_tests.csv', index=False)
        
        # Apply FDR correction
        from statsmodels.stats.multitest import fdrcorrection
        
        for comparison in stat_df['comparison'].unique():
            comp_results = stat_df[stat_df['comparison'] == comparison]
            reject, p_corrected = fdrcorrection(comp_results['p_value'].values)
            
            # Update the original dataframe
            stat_df.loc[comp_results.index, 'p_corrected'] = p_corrected
            stat_df.loc[comp_results.index, 'significant_corrected'] = reject
        
        # Save updated results
        stat_df.to_csv(OUTPUT_DIR / 'cortical_statistical_tests_corrected.csv', index=False)
        
        # Visualize significant results after correction
        sig_results = stat_df[stat_df['significant_corrected'] == True]
        if len(sig_results) > 0:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='region', y='effect_size', hue='comparison', data=sig_results)
            plt.title('Significant Group Differences in Cortical Thickness (FDR-corrected)')
            plt.xlabel('Brain Region')
            plt.ylabel('Effect Size (Cohen\'s d)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'significant_cortical_differences.png', dpi=300)
        else:
            logger.info("No significant differences after FDR correction")
    
    logger.info("Completed cortical thickness exploration")

def explore_demographics(demo_data):
    """Explore demographic data"""
    logger.info("Exploring demographic data")
    
    if demo_data is None:
        logger.error("No demographic data available")
        return
    
    # Check required columns
    if 'subject_id' not in demo_data.columns:
        subject_cols = [col for col in demo_data.columns if 'subject' in col.lower() or 'id' in col.lower()]
        if subject_cols:
            demo_data = demo_data.rename(columns={subject_cols[0]: 'subject_id'})
        else:
            logger.error("No subject ID column found in demographic data")
            return
    
    if 'group' not in demo_data.columns:
        group_cols = [col for col in demo_data.columns if 'group' in col.lower() or 'type' in col.lower()]
        if group_cols:
            demo_data = demo_data.rename(columns={group_cols[0]: 'group'})
        else:
            logger.error("No group column found in demographic data")
            return
    
    # Check if age and gender columns exist
    age_col = None
    for col in demo_data.columns:
        if 'age' in col.lower():
            age_col = col
            break
    
    gender_col = None
    for col in demo_data.columns:
        if 'gender' in col.lower() or 'sex' in col.lower():
            gender_col = col
            break
    
    # Basic statistics by group
    group_stats = demo_data.groupby('group').size().reset_index(name='count')
    logger.info(f"Group counts: {group_stats.to_dict()}")
    
    # Visualize subject counts by group
    plt.figure(figsize=(8, 6))
    sns.countplot(x='group', data=demo_data, palette='viridis')
    plt.title('Subject Count by Group')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.savefig(OUTPUT_DIR / 'subject_count_by_group.png', dpi=300)
    
    # If age column exists, analyze age
    if age_col:
        # Check if there are non-NaN values before plotting
        if demo_data[age_col].notna().any():
            # Descriptive statistics
            try:
                age_stats = demo_data.groupby('group')[age_col].agg(['mean', 'std', 'min', 'max']).reset_index()
                age_stats.to_csv(OUTPUT_DIR / 'age_statistics_by_group.csv', index=False)
                
                # Visualize age distribution - filter out NaN values
                valid_age_data = demo_data.dropna(subset=[age_col])
                
                if len(valid_age_data) > 0:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='group', y=age_col, data=valid_age_data)
                    plt.title('Age Distribution by Group')
                    plt.xlabel('Group')
                    plt.ylabel('Age (years)')
                    plt.savefig(OUTPUT_DIR / 'age_distribution_by_group.png', dpi=300)
                else:
                    logger.warning("No valid age data for plotting")
                
                # Statistical testing for age differences - only if we have sufficient data
                if len(valid_age_data) > 2:
                    groups_with_data = []
                    group_data = []
                    
                    for group in valid_age_data['group'].unique():
                        data = valid_age_data[valid_age_data['group'] == group][age_col].dropna()
                        if len(data) > 1:  # Need at least 2 samples for t-test
                            groups_with_data.append(group)
                            group_data.append(data)
                    
                    if len(groups_with_data) >= 2 and len(group_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        logger.info(f"Age ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
                        
                        # Pairwise t-tests if we have multiple groups
                        if len(groups_with_data) > 1:
                            for i in range(len(groups_with_data)):
                                for j in range(i+1, len(groups_with_data)):
                                    g1, g2 = groups_with_data[i], groups_with_data[j]
                                    d1, d2 = group_data[i], group_data[j]
                                    
                                    t_stat, t_p = stats.ttest_ind(d1, d2, equal_var=False)
                                    logger.info(f"Age t-test {g1} vs {g2}: t={t_stat:.4f}, p={t_p:.4f}")
            except Exception as e:
                logger.warning(f"Error in age statistical tests: {e}")
    
    # If gender column exists, analyze gender
    if gender_col and demo_data[gender_col].notna().any():
        try:
            # Filter out NaN values for gender analysis
            valid_gender_data = demo_data.dropna(subset=[gender_col])
            
            if len(valid_gender_data) > 0:
                # Count by gender and group
                gender_counts = valid_gender_data.groupby(['group', gender_col]).size().reset_index(name='count')
                gender_counts.to_csv(OUTPUT_DIR / 'gender_counts_by_group.csv', index=False)
                
                # Visualize gender distribution
                plt.figure(figsize=(10, 6))
                sns.countplot(x='group', hue=gender_col, data=valid_gender_data)
                plt.title('Gender Distribution by Group')
                plt.xlabel('Group')
                plt.ylabel('Count')
                plt.savefig(OUTPUT_DIR / 'gender_distribution_by_group.png', dpi=300)
                
                # Chi-square test for gender differences - only if we have sufficient data
                if len(valid_gender_data['group'].unique()) >= 2:
                    # Create contingency table
                    contingency = pd.crosstab(valid_gender_data['group'], valid_gender_data[gender_col])
                    
                    # Only perform test if we have data in at least 2 cells
                    if (contingency > 0).sum().sum() >= 2:
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        logger.info(f"Gender Chi-square: χ²={chi2:.4f}, p={p:.4f}, dof={dof}")
        except Exception as e:
            logger.warning(f"Error in gender statistical tests: {e}")
    
    logger.info("Completed demographic exploration")

def main():
    """Main function to run all data exploration"""
    logger.info("Starting data exploration for YOPD Motor Subtype project")
    
    # Load data
    subcort_data, cortical_data, demo_data = load_all_data()
    
    # Explore each dataset
    explore_subcortical_volumes(subcort_data)
    explore_cortical_thickness(cortical_data)
    explore_demographics(demo_data)
    
    # Create summary report
    with open(OUTPUT_DIR / 'exploration_summary.md', 'w') as report:
        report.write("# YOPD Motor Subtype Data Exploration Summary\n\n")
        report.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.write("## Dataset Overview\n\n")
        
        if subcort_data is not None:
            report.write(f"- Subcortical volume data: {subcort_data['subject_id'].nunique()} subjects, ")
            report.write(f"{subcort_data['structure'].nunique()} structures\n")
            
            # Group counts
            group_counts = subcort_data.groupby('group')['subject_id'].nunique()
            for group, count in group_counts.items():
                report.write(f"  - {group}: {count} subjects\n")
        else:
            report.write("- Subcortical volume data: Not available\n")
            
        if cortical_data is not None:
            subject_col = 'Subject' if 'Subject' in cortical_data.columns else 'subject_id'
            region_col = 'Region' if 'Region' in cortical_data.columns else 'region'
            
            report.write(f"- Cortical thickness data: {cortical_data[subject_col].nunique()} subjects, ")
            report.write(f"{cortical_data[region_col].nunique()} regions\n")
        else:
            report.write("- Cortical thickness data: Not available\n")
            
        if demo_data is not None:
            report.write(f"- Demographic data: {len(demo_data)} subjects\n")
        else:
            report.write("- Demographic data: Not available\n")
            
        report.write("\n## Key Findings\n\n")
        
        report.write("### Subcortical Volumes\n\n")
        subcort_test_file = OUTPUT_DIR / 'subcortical_statistical_tests.csv'
        if subcort_test_file.exists():
            subcort_tests = pd.read_csv(subcort_test_file)
            sig_tests = subcort_tests[subcort_tests['significant']]
            
            if len(sig_tests) > 0:
                report.write(f"Found {len(sig_tests)} significant differences in subcortical volumes:\n\n")
                for _, test in sig_tests.iterrows():
                    report.write(f"- {test['structure']}: {test['comparison']}, p={test['p_value']:.4f}, ")
                    report.write(f"effect size={test['effect_size']:.4f}\n")
            else:
                report.write("No significant differences in subcortical volumes between groups\n")
        else:
            report.write("No statistical tests performed on subcortical volumes\n")
            
        report.write("\n### Cortical Thickness\n\n")
        cort_test_file = OUTPUT_DIR / 'cortical_statistical_tests_corrected.csv'
        if cort_test_file.exists():
            cort_tests = pd.read_csv(cort_test_file)
            if 'significant_corrected' in cort_tests.columns:
                sig_tests = cort_tests[cort_tests['significant_corrected'] == True]
                
                if len(sig_tests) > 0:
                    report.write(f"Found {len(sig_tests)} significant differences in cortical thickness after FDR correction:\n\n")
                    for _, test in sig_tests.iterrows():
                        report.write(f"- {test['region']}: {test['comparison']}, p={test['p_corrected']:.4f}, ")
                        report.write(f"effect size={test['effect_size']:.4f}\n")
                else:
                    report.write("No significant differences in cortical thickness between groups after FDR correction\n")
            else:
                report.write("No FDR-corrected tests available for cortical thickness\n")
        else:
            report.write("No statistical tests performed on cortical thickness\n")
            
        report.write("\n### Demographics\n\n")
        if demo_data is not None and 'group' in demo_data.columns:
            age_col = [col for col in demo_data.columns if 'age' in col.lower()]
            if age_col:
                try:
                    age_stats = demo_data.groupby('group')[age_col[0]].agg(['mean', 'std']).reset_index()
                    report.write("Age statistics by group:\n\n")
                    for _, row in age_stats.iterrows():
                        report.write(f"- {row['group']}: {row['mean']:.1f} ± {row['std']:.1f} years\n")
                except Exception as e:
                    logger.warning(f"Error generating age statistics: {e}")
                    report.write("Age statistics could not be calculated\n")
            
            gender_col = [col for col in demo_data.columns if 'gender' in col.lower() or 'sex' in col.lower()]
            if gender_col:
                try:
                    gender_counts = demo_data.groupby(['group', gender_col[0]]).size().reset_index(name='count')
                    report.write("\nGender distribution by group:\n\n")
                    for group in demo_data['group'].unique():
                        group_data = gender_counts[gender_counts['group'] == group]
                        report.write(f"- {group}: ")
                        report.write(", ".join([f"{row[gender_col[0]]}: {row['count']}" for _, row in group_data.iterrows()]))
                        report.write("\n")
                except Exception as e:
                    logger.warning(f"Error generating gender statistics: {e}")
                    report.write("Gender distribution could not be calculated\n")
        else:
            report.write("No demographic group information available\n")
            
        report.write("\n## Visualizations\n\n")
        report.write("The following visualizations were created:\n\n")
        report.write("1. Subcortical volumes by group\n")
        report.write("2. Cortical thickness by group\n")
        report.write("3. Age and gender distributions\n")
        report.write("4. Statistical test results\n\n")
        
        report.write("See image files in the same directory for details.\n")
    
    logger.info(f"Saved exploration summary to {OUTPUT_DIR / 'exploration_summary.md'}")
    logger.info("Data exploration complete")

if __name__ == "__main__":
    main()
