# YOPD_Motor_Subtype

Dataset Description:- 
Total 75 Patients, among which 25 patients - healthy controls, 25 patients - PIGD (Postural Instability and Gait Difficulty), 25 patients - TDPD (Tremor-Dominant PD), which are 2 motor subtypes of Parkinson's Disease.

Consists of a metadata file - age_gender.xlsx, that has all 75 subjects with their patient id, mapped to which motor subgroup/healthy control they belong to, age and gender. For each patient the data is of the following format - 1 session neuroimaging data consisting of 3 major types - a) anat b) fmap c) func. 

a) anat is of the following type - high-resolution 3D structural brain MRI (T1-weighted MPRAGE) - commonly used for structural analysis - cortical thickness, brain volume, registration base for functional scans. 

b) fmap - 3 scans = 1. acq-bold_run-1_magnitude1, 2. acq-bold_run-1_magnitude2, 3. acq-bold_run-1_phasediff.json. 

fmap = fieldmap - phase-difference-based, used to correct distortion in functional scans - fmri - BOLD scan. Used before aligning to the anatomical scan.
1&2 are anatomical magnitude images recorded at 2 echo times used for masking and co-registration.
3 - the actual field map used to calculate the B0 inhomogeneity map that helps correct susceptibilty distortions in fMRI.

c) func - resting state fmri scan
