#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import subprocess
import tempfile
import pandas as pd
import nibabel as nib
import numpy as np
import csv
from scipy.ndimage import label as ndimage_label
from bids import BIDSLayout

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────
REGION_LABELS = {'visual': ['Vis'], 'ventral': ['Vent'], 'dorsal': ['Dors']}
TARGET_SPMT = ('spmT_0001', 'spmT_0008', 'spmT_0009')

def get_label_ids(labelfile_path):
    df = pd.read_csv(labelfile_path, sep=r'\s+', engine='python', header=None)
    col_id, col_name = df.columns[0], df.columns[4]
    return {reg: df[df[col_name].str.contains(pattern, na=False)][col_id].tolist()
            for reg, patterns in REGION_LABELS.items() for pattern in patterns}

def voxel_to_world(vox, affine):
    return tuple((affine @ np.array([vox[0], vox[1], vox[2], 1.0]))[:3])

def ants_points_to_mni(native_xyz_list, inv_warp, affine_mat, tmp_dir):
    in_csv = os.path.join(tmp_dir, 'pts_in.csv')
    out_csv = os.path.join(tmp_dir, 'pts_out.csv')
    with open(in_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z', 't'])
        for (x, y, z) in native_xyz_list:
            writer.writerow([-x, -y, z, 0]) # RAS -> LPS
    
    cmd = ['antsApplyTransformsToPoints', '-d', '3', '-i', in_csv, '-o', out_csv,
           '-t', f'[{affine_mat},1]', '-t', inv_warp]
    subprocess.run(cmd, check=True, capture_output=True)
    
    mni_coords = []
    with open(out_csv, 'r') as f:
        for row in csv.DictReader(f):
            mni_coords.append((-float(row['x']), -float(row['y']), float(row['z'])))
    return mni_coords

def main():
    bids_root = "/mnt/store2/projects/progetto_layer/data"
    roi_deriv = "roi_statistics-masked"
    atlas_path = "/mnt/store2/projects/progetto_layer/parcels/Yan2023_MNI152NLin6Asym/200Parcels_Yeo2011_17Networks_FSLMNI152_1mm.nii.gz"
    labelfile = "/mnt/store2/projects/progetto_layer/parcels/Yan2023_MNI152NLin6Asym/fsleyes_lut/200Parcels_Yeo2011_17Networks_LUT.txt"
    
    layout = BIDSLayout(os.path.join(bids_root, "derivatives", "hires_pipeline"), validate=False)
    input_path = os.path.join(bids_root, "derivatives", roi_deriv)
    roi_labels = get_label_ids(labelfile)
    atlas_data = nib.load(atlas_path).get_fdata()
    mni_ref = nib.load(atlas_path) # Usato come riferimento per MNI affine

    all_files = glob.glob(os.path.join(input_path, "sub-*", "func", "*_desc-spmT_*thresh001_statmap.nii.gz"))
    filtered_list = [f for f in all_files if f.endswith(TARGET_SPMT + ('.nii.gz',)) 
                 and any(pattern in f for pattern in TARGET_SPMT)]
    excel_data = {}

    for f in filtered_list:
        filename = os.path.basename(f)
        sub = filename.split('_')[0].split('-')[1]
        contrast = filename.split('_desc-')[1].split('thresh')[0]
        
        print(f"Processing subject {sub} | contrast {contrast}")

        # Recupero trasformazioni
        inv_warp = layout.get(subject=sub, suffix='SyN1InverseWarp', extension='.nii.gz', return_type='file')[0]
        affine_mat = layout.get(subject=sub, suffix='SyN0GenericAffine', extension='.mat', return_type='file')[0]
        
        native_atlas_path = os.path.join(
            input_path, f"sub-{sub}", "anat",
            f"sub-{sub}_space-T1w_desc-resampled_atlas.nii.gz"
        )

        atlas_data = nib.load(native_atlas_path).get_fdata()

        t_img = nib.load(f)
        t_data = t_img.get_fdata()
        native_affine = t_img.affine
        
        if contrast not in excel_data: excel_data[contrast] = []
        
        for region, labels in roi_labels.items():
            print(f"  Analyzing region: {region}")
            roi_mask = np.isin(atlas_data, labels)
            masked_t = np.where(roi_mask, t_data, 0)
            
            labeled_array, num_features = ndimage_label(masked_t > 0)
            for cid in range(1, num_features + 1):
                mask = (labeled_array == cid)
                max_t = np.max(t_data[mask])
                coords = np.argwhere((labeled_array == cid) & (t_data == max_t))[0]
                
                # MNI Transformation
                world_native = voxel_to_world(coords, native_affine)
                with tempfile.TemporaryDirectory() as tmp:
                    mni_mm = ants_points_to_mni([world_native], inv_warp, affine_mat, tmp)[0]
                
                mni_vox = np.linalg.inv(mni_ref.affine) @ np.array(list(mni_mm) + [1.0])
                
                excel_data[contrast].append({
                    'Subject': sub, 'Region': region, 'Cluster_ID': cid, 'Size': np.sum(mask),
                    'Peak_T': round(max_t, 3), 'Native_Vox': tuple(coords),
                    'MNI_mm': tuple(np.round(mni_mm, 2)), 'MNI_Vox': tuple(np.round(mni_vox[:3]).astype(int))
                })

    output_xlsx = os.path.join(input_path, "all_clusters_report.xlsx")
    with pd.ExcelWriter(output_xlsx) as writer:
        for contrast, data in excel_data.items():
            pd.DataFrame(data).to_excel(writer, sheet_name=contrast, index=False)
            
    print(f"Report MNI salvato in: {output_xlsx}")

if __name__ == "__main__":
    main()