#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from datetime import date

# Set working directory
base_dir = os.path.expanduser("~/pd_analysis")
stats_dir = os.path.join(base_dir, "stats")
os.makedirs(stats_dir, exist_ok=True)

# Read subject data
subjects = pd.read_csv(os.path.join(base_dir, "all_subjects.csv"))

# Function to perform non-parametric statistical test (equivalent to wilcox_test in R)
def run_wilcoxon_test(data, value_col, group_col, group1, group2):
    results = []
    
    for name, group_data in data.groupby("structure"):
        group1_data = group_data[group_data[group_col] == group1][value_col]
        group2_data = group_data[group_data[group_col] == group2][value_col]
        
        # Skip if not enough data
        if len(group1_data) < 2 or len(group2_data) < 2:
            continue
            
        # Run Wilcoxon rank-sum test (Mann-Whitney U test)
        stat, pvalue = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        results.append({
            "structure": name,
            "n1": len(group1_data),
            "n2": len(group2_data),
            "statistic": stat,
            "p": pvalue
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Adjust p-values for multiple comparisons
    if len(results_df) > 0:
        results_df['p.adj'] = multipletests(results_df['p'], method='bonferroni')[1]
        
        # Add significance symbols
        results_df['significance'] = 'ns'
        results_df.loc[results_df['p.adj'] < 0.05, 'significance'] = '*'
        results_df.loc[results_df['p.adj'] < 0.01, 'significance'] = '**'
        results_df.loc[results_df['p.adj'] < 0.001, 'significance'] = '***'
    
    return results_df

# Function to analyze volumes for specific ROIs by group
def analyze_volumes(data, structures, group1, group2, name1, name2):
    # Filter data
    filtered_data = data[(data['structure'].isin(structures)) & 
                         (data['group'].isin([group1, group2]))]
    
    # Run statistical tests
    stats_results = run_wilcoxon_test(filtered_data, 'volume_mm3', 'group', group1, group2)
    
    # Create and save plots
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='structure', y='volume_mm3', hue='group', data=filtered_data)
    
    # Add significance markers
    for i, struct in enumerate(structures):
        struct_stats = stats_results[stats_results['structure'] == struct]
        if not struct_stats.empty and struct_stats['p.adj'].values[0] < 0.05:
            y_max = filtered_data[filtered_data['structure'] == struct]['volume_mm3'].max()
            plt.text(i, y_max * 1.05, struct_stats['significance'].values[0], 
                    ha='center', va='bottom', color='black', fontsize=14)
    
    plt.title(f"Subcortical Volume Comparison: {name1} vs {name2}")
    plt.xlabel("Brain Structure")
    plt.ylabel("Volume (mm³)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(stats_dir, f"{name1.lower()}_vs_{name2.lower()}_volumes.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save stats
    stats_path = os.path.join(stats_dir, f"{name1.lower()}_vs_{name2.lower()}_volume_stats.csv")
    if len(stats_results) > 0:
        stats_results.to_csv(stats_path, index=False)
    
    return stats_results

# Function to analyze cortical thickness
def analyze_thickness(roi_prefix, group1, group2, name1, name2):
    # List all thickness files for this ROI
    thickness_files = glob.glob(os.path.join(stats_dir, "cortical_thickness", f"*{roi_prefix}*.csv"))
    
    if not thickness_files:
        print(f"No files found matching pattern {roi_prefix}")
        return None
    
    # Combine all data
    thickness_data = None
    
    for file in thickness_files:
        data = pd.read_csv(file)
        roi_name = os.path.basename(file).split('_')[-1].replace('.csv', '')
        
        if thickness_data is None:
            thickness_data = data
            # Rename the 3rd column to ROI name
            thickness_data.columns = list(thickness_data.columns[:2]) + [roi_name] + list(thickness_data.columns[3:])
        else:
            # Merge on subject_id
            roi_data = data[['subject_id', data.columns[2]]]
            roi_data.columns = ['subject_id', roi_name]
            thickness_data = pd.merge(thickness_data, roi_data, on='subject_id')
    
    if thickness_data is None:
        return None
    
    # Filter to only include specified groups
    thickness_data = thickness_data[thickness_data['group'].isin([group1, group2])]
    
    # Reshape to long format for analysis
    id_vars = ['subject_id', 'group']
    value_vars = [col for col in thickness_data.columns if col not in id_vars]
    
    thickness_long = pd.melt(
        thickness_data, 
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='region',
        value_name='thickness'
    )
    
    # Run statistical tests
    stats_results = run_wilcoxon_test(thickness_long, 'thickness', 'group', group1, group2)
    
    # Create plots
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='region', y='thickness', hue='group', data=thickness_long)
    plt.title(f"Cortical Thickness Comparison: {name1} vs {name2}")
    plt.xlabel("Brain Region")
    plt.ylabel("Thickness (mm)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(stats_dir, f"{name1.lower()}_vs_{name2.lower()}_{roi_prefix}_thickness.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save stats
    stats_path = os.path.join(stats_dir, f"{name1.lower()}_vs_{name2.lower()}_{roi_prefix}_thickness_stats.csv")
    if len(stats_results) > 0:
        stats_results.to_csv(stats_path, index=False)
    
    return stats_results

# Main analysis

# 1. Analyze subcortical volumes
print("Analyzing subcortical volumes...")
volumes = pd.read_csv(os.path.join(stats_dir, "all_subcortical_volumes.csv"))

# Analyze PIGD vs HC
pigd_structures = ["L_Pall", "R_Pall", "BrStem"]
pigd_vs_hc = analyze_volumes(volumes, pigd_structures, "PIGD", "HC", "PIGD", "HC")

# Analyze TDPD vs HC
tdpd_structures = ["L_Thal", "R_Thal", "L_Caud", "R_Caud"]
tdpd_vs_hc = analyze_volumes(volumes, tdpd_structures, "TDPD", "HC", "TDPD", "HC")

# Analyze PIGD vs TDPD
pigd_vs_tdpd = analyze_volumes(volumes, 
                               pigd_structures + tdpd_structures, 
                               "PIGD", "TDPD", "PIGD", "TDPD")

# 2. Analyze cortical thickness
print("Analyzing cortical thickness...")

# Make sure directory exists
os.makedirs(os.path.join(stats_dir, "cortical_thickness"), exist_ok=True)

# Analyze PIGD-specific ROIs
pigd_rois = analyze_thickness("supplementarymotor|paracentral|precentral", 
                             "PIGD", "HC", "PIGD", "HC")

# Analyze TDPD-specific ROIs
tdpd_rois = analyze_thickness("precentral|postcentral|superiortemporal", 
                             "TDPD", "HC", "TDPD", "HC")

# Compare PIGD vs TDPD
pigd_vs_tdpd_thickness = analyze_thickness("supplementarymotor|paracentral|precentral|postcentral|superiortemporal", 
                                          "PIGD", "TDPD", "PIGD", "TDPD")

# 3. Generate comprehensive report
print("Generating analysis report...")
report_path = os.path.join(stats_dir, "analysis_summary.txt")

with open(report_path, 'w') as report:
    report.write("=== PD Motor Subtype Neuroimaging Analysis ===\n\n")
    report.write(f"Date of Analysis: {date.today()}\n\n")
    
    report.write("1. SUBCORTICAL VOLUMES\n")
    report.write("1.1 PIGD vs HC:\n")
    if pigd_vs_hc is not None and len(pigd_vs_hc) > 0:
        report.write(pigd_vs_hc.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("1.2 TDPD vs HC:\n")
    if tdpd_vs_hc is not None and len(tdpd_vs_hc) > 0:
        report.write(tdpd_vs_hc.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("1.3 PIGD vs TDPD:\n")
    if pigd_vs_tdpd is not None and len(pigd_vs_tdpd) > 0:
        report.write(pigd_vs_tdpd.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("2. CORTICAL THICKNESS\n")
    report.write("2.1 PIGD-specific regions (PIGD vs HC):\n")
    if pigd_rois is not None and len(pigd_rois) > 0:
        report.write(pigd_rois.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("2.2 TDPD-specific regions (TDPD vs HC):\n")
    if tdpd_rois is not None and len(tdpd_rois) > 0:
        report.write(tdpd_rois.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("2.3 PIGD vs TDPD cortical thickness:\n")
    if pigd_vs_tdpd_thickness is not None and len(pigd_vs_tdpd_thickness) > 0:
        report.write(pigd_vs_tdpd_thickness.to_string())
    else:
        report.write("No significant results found.")
    report.write("\n\n")
    
    report.write("=== End of Analysis ===\n")

print("Statistical analysis completed. Results saved to stats/ directory.")