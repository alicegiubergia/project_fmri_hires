#!/usr/bin/env python3
import ants
import os
import json
import argparse
from datetime import datetime
import numpy as np
import antspynet
from preprocessing_scripts.preprocess_t1 import bias_correction_spm
from utils import myutils
import shutil

PROCESSING_LOG = []

def log_step(name, input_file, output_file, params=None):
    """Aggiunge un record di step di preprocessing al log globale."""
    PROCESSING_LOG.append({
        "StepName": name,
        "InputFile": os.path.basename(input_file) if input_file else None,
        "OutputFile": os.path.basename(output_file) if output_file else None,
        "Parameters": params if params else {},
        "Timestamp": datetime.now().isoformat(timespec='seconds')
    })

def preprocess_t2(sub_id, t2_file, t1_file, t1_brainmask, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"sub-{sub_id}", "anat", f"sub-{sub_id}_")

    work_file = f"{prefix}desc-orig_t2w.nii.gz"
    shutil.copyfile(t2_file, work_file)

    # === 1. Load images ===
    print(f"[sub-{sub_id}] Loading T1 and T2 images...")
    t1 = ants.image_read(t1_file)
    
    t2_bc_file = f"{prefix}desc-N4_T2w.nii.gz"

    # === 2. Bias Field Correction ===
    bias_correction_spm(work_file, t2_bc_file, mode='t2')

    t2_bc = ants.image_read(t2_bc_file)

    # === 3. Rigid registration T2 → T1 ===
    print(f"[sub-{sub_id}] Registering T2 to T1 (Rigid)...")
    tx_t2_to_t1 = ants.registration(
        fixed=t1,
        moving=t2_bc,
        prefix = prefix + "from-T2_to-T1_",
        type_of_transform="Rigid"
    )

    t2_in_t1 = tx_t2_to_t1['warpedmovout']
    t2_in_t1_fn = f"{prefix}space-T1_T2w.nii.gz"

    ants.image_write(t2_in_t1, t2_in_t1_fn)
    log_step("Move to T1 space", t2_file, t2_in_t1_fn, {"TransformType": "Rigid"})

    seg = antspynet.brain_extraction(t2_in_t1, modality="t2", verbose=True)
    seg_out = f"{os.path.join(output_dir, f'sub-{sub_id}', 'anat')}/brainExtraction_ants/output_t2_seg.nii.gz"
    ants.image_write(seg, seg_out)

    # === 3. Intensity normalization 0-100 ===
    print(f"[sub-{sub_id}] Normalizing T2 intensity (0-100)...")
    img = ants.image_read(t2_in_t1_fn)
    mask = ants.image_read(t1_brainmask)
    # Applica la maschera
    masked = img * mask

    # Calcola min/max solo nei voxel validi
    vals = masked[mask > 0]
    pmean = vals.mean()
    norm = (img / pmean) * 100
    log_step("Intensity Normalization", t2_in_t1_fn, t2_in_t1_fn, {"Range": "0-100"})

    # === 4. Save JSON log ===
 
    final_report = {
        "PipelineDescription": {
            "Name": "Custom T2 Anatomical Preprocessing",
            "Version": "1.0",
            "CodeURL": "https://github.com/alicegiubergia/project_fmri_hires"
        },
        "SoftwareVersions": {
            "ANTs": ants.__version__
        },
        "SubjectID": sub_id,
        "Input": t2_file,
        "OutputDirectory": output_dir,
        "ProcessingSteps": PROCESSING_LOG
    }
    json_output = f"{prefix}preprocess_T2w.json"
    with open(json_output, "w") as f:
        json.dump(final_report, f, indent=4)

    print(f"[sub-{sub_id}] ✅ T2 preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess T2-weighted image using T1 reference and mask (no MNI).")
    parser.add_argument("--sub", required=True, help="Subject ID (e.g. 01).")
    parser.add_argument("--t2", required=True, help="Path to T2-weighted image.")
    parser.add_argument("--t1", required=True, help="Path to T1-weighted image.")
    parser.add_argument("--t1_mask", required=True, help="Path to T1 brain mask.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    args = parser.parse_args()

    # Log
    log_dir = os.path.join(args.output_dir, f"sub-{args.sub}", "anat")
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILE = os.path.join(log_dir, "preprocess_t2.log")
    myutils.set_log_file(LOG_FILE)
    preprocess_t2(
        sub_id=args.sub,
        t2_file=args.t2,
        t1_file=args.t1,
        t1_brainmask=args.t1_mask,
        output_dir=args.output_dir
    )

