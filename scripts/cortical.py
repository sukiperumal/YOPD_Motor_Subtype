import os
import nibabel as nib
import pandas as pd
from nighres import brain, cortex

# Paths
preprocessed_dir = "./preprocessed"
output_dir = "./cortical_thickness"
os.makedirs(output_dir, exist_ok=True)

# Subjects list
subjects = [f for f in os.listdir(preprocessed_dir) if f.startswith("sub-")]

# Output CSV
thickness_data = []

for subj in subjects:
    subj_dir = os.path.join(preprocessed_dir, subj)
    t1_img_path = os.path.join(subj_dir, f"{subj}_brain.nii.gz")
    
    print(f"Processing {subj}...")

    # Step 1: Tissue classification (WM, GM, CSF)
    mgdm = brain.extract_brain_region(
        main_image=t1_img_path,
        method='mgdm',
        save_data=True,
        output_dir=output_dir,
        file_name=subj
    )

    # Step 2: CRUISE cortex extraction
    cruise = cortex.cruise_cortex_extraction(
        main_image=t1_img_path,
        gmw_image=mgdm['mgdm_gm'],
        wm_image=mgdm['mgdm_wm'],
        save_data=True,
        output_dir=output_dir,
        file_name=subj
    )

    # Step 3: Thickness calculation
    thickness = cortex.laminar_thickness(
        inner_levelset=cruise['cruise_inner_surface'],
        outer_levelset=cruise['cruise_outer_surface'],
        save_data=True,
        output_dir=output_dir,
        file_name=subj
    )

    # Get mean cortical thickness
    thickness_img = nib.load(thickness['thickness'])
    data = thickness_img.get_fdata()
    mean_thickness = data[data > 0].mean()

    thickness_data.append({
        "subject": subj,
        "mean_cortical_thickness": round(mean_thickness, 4)
    })

# Save to CSV
df = pd.DataFrame(thickness_data)
df.to_csv(os.path.join(output_dir, "mean_cortical_thickness.csv"), index=False)
print("✅ Cortical thickness extraction completed.")
