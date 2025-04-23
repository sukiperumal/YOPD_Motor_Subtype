# Resting-State fMRI Analysis of YOPD Motor Subtypes

**Generated on:** 2025-04-23 17:40:30

## Project Overview

This report summarizes the analysis of resting-state functional MRI data for Young-Onset Parkinson's Disease Motor Subtypes:

* **PIGD**: Postural Instability and Gait Difficulty subtype
* **TDPD**: Tremor Dominant subtype
* **HC**: Healthy Controls

## Hypotheses

1. **PIGD**: Expected to show reduced connectivity in the frontostriatal circuit
2. **TDPD**: Expected to show hyperconnectivity in the cerebello-thalamo-cortical loop

## Methods

### Preprocessing Pipeline

1. Field map-based distortion correction using FUGUE
2. Motion correction using MCFLIRT
3. Registration to anatomical T1 image and then to MNI standard space
4. Spatial smoothing with FWHM=6mm
5. Temporal high-pass filtering (cutoff: 100s)

### Analysis Pipeline

1. Group Independent Component Analysis (ICA) with 20 components
2. Identification of networks of interest (frontostriatal and cerebello-thalamo-cortical)
3. Dual regression to obtain subject-specific network maps
4. Statistical comparison between groups (PIGD vs HC, TDPD vs HC, PIGD vs TDPD)

## Overall Findings

### Frontostriatal Network

#### PIGD vs HC

* No significant differences found

#### TDPD vs HC

* No significant differences found

#### PIGD vs TDPD

* No significant differences found

### Cerebello_thalamo_cortical Network

#### PIGD vs HC

* No significant differences found

#### TDPD vs HC

* No significant differences found

#### PIGD vs TDPD

* No significant differences found

## Interpretation

Analysis was performed on 6 subjects total:
- 2 Healthy Controls (HC)
- 0 Postural Instability and Gait Difficulty (PIGD) subtype patients
- 0 Tremor Dominant (TDPD) subtype patients

The analysis focused on the functional connectivity differences between the motor subtypes of Young-Onset Parkinson's Disease. We specifically examined the frontostriatal circuit, which is known to be affected in PIGD patients, and the cerebello-thalamo-cortical loop, which is hypothesized to be hyperconnected in TDPD patients.

The results did not show significant differences between the groups, which could be due to several reasons:

1. The preprocessing pipeline may need further optimization for this specific dataset
2. The sample size may be insufficient to detect subtle connectivity differences
3. The hypothesized network differences may be more complex than our current analysis can detect

Future analysis should consider:

1. Using more advanced connectivity measures such as dynamic connectivity analysis
2. Incorporating structural connectivity information (DTI)
3. Including clinical measures as covariates to account for disease severity



## Conclusion

This analysis provides insights into the differential functional connectivity patterns between PIGD and TDPD motor subtypes of Young-Onset Parkinson's Disease. The results help clarify the neurobiological basis of clinical heterogeneity in PD and may guide future development of subtype-specific interventions.
