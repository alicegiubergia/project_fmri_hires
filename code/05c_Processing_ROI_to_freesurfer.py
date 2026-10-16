#!/bin/bash
#
# ======================================================================
# Script Name: register_roi_MNI152_to_FS_T1.sh
#
# Purpose:
#   Register a ROI in MNI152 space to the subject's FreeSurfer T1 space.
#
# Steps:
#   1) Convert T1.mgz → NIfTI (needed for FLIRT)
#   2) Compute MNI152 → T1 native registration (FLIRT, 12 DOF)
#   3) Apply transform to the ROI (nearest neighbor)
#   4) Convert resampled ROI → .mgz
#
# Requirements:
#   FSL + FreeSurfer
#
# Output:
#   ROI_in_FS_T1.mgz
# ======================================================================

import os
import sys
import ants
import shutil

# ================================
# Input check
# ================================
if len(sys.argv) < 2:
    print("Usage: python 04d_Processing_ROI_to_freesurfer.py <SUBJECT_ID>")
    sys.exit(1)

SUBJECT = sys.argv[1]

# ================================
# Paths
# ================================
FS_DOCKER_IMAGE="cesne/freesurfer:8.0"
SUBJECTS_DIR="/mnt/store2/projects/progetto_layer/data/derivatives/freesurfer"
MYPIPE_DIR="/mnt/store2/projects/progetto_layer/data/derivatives/mypipeline"
ROI_PATH="/mnt/store2/projects/progetto_ME-fMRI/stats_corrected/group/ROI_flip_spacing_union.nii.gz"
MNI_PATH="/usr/local/imaging_software/Templates/MNI/MNI152_T1_1mm_brain_filled.nii.gz"
T1_MGZ=f"{SUBJECTS_DIR}/{SUBJECT}/mri/brainmask.mgz"
T1_NII=f"{MYPIPE_DIR}/{SUBJECT}/anat/{SUBJECT}_desc-brain_T1w.nii.gz"  # opzionale

OUTPUT_NII = f"{SUBJECT}/anat/{SUBJECT}_ROI_in_FS_T1.nii.gz"
OUTPUT_MGZ = f"{SUBJECT}/anat/{SUBJECT}_ROI_in_FS_T1.mgz"  # opzionale

# ================================
# Sanity checks
# ================================
if not os.path.isfile(ROI_PATH):
    raise FileNotFoundError(f"ROI not found: {ROI_PATH}")

if not os.path.isfile(T1_MGZ):
    raise FileNotFoundError(f"FreeSurfer T1.mgz not found: {T1_MGZ}")

if not os.path.isfile(MNI_PATH):
    raise FileNotFoundError(f"MNI template missing: {MNI_PATH}")

print(f">>> Registering ROI from MNI152 → FS T1 for subject {SUBJECT}")

# ================================
# Load images with ANTs
# ================================
print(">>> Loading images...")
moving_T1 = ants.image_read(T1_NII)             # FreeSurfer T1
fixed_MNI = ants.image_read(MNI_PATH)         # MNI template 2mm
roi_MNI = ants.image_read(ROI_PATH)            # ROI in MNI space

# ================================
# ANTs SyN registration
# ================================
print(">>> Running ANTs SyN (non-linear)...")

reg = ants.registration(
    fixed=fixed_MNI,
    moving=moving_T1,
    type_of_transform="SyN",     # come antsRegistrationSyN.sh
    aff_metric="mattes",
    syn_metric="mattes",
)

print(">>> Registration completed.")

# ================================
# Apply non-linear transform to ROI
# ================================
print(">>> Applying transformation to ROI (nearest neighbour)...")

roi_in_T1 = ants.apply_transforms(
    fixed=moving_T1,
    moving=roi_MNI,
    transformlist=reg["invtransforms"],
    interpolator="nearestNeighbor"
)
affine_mat = reg["fwdtransforms"][1]   # in genere il file .mat
shutil.copyfile(affine_mat, f"{MYPIPE_DIR}/{SUBJECT}/anat/{SUBJECT}_MNI_2fs0GenericAffine.mat")
warp_field = reg["fwdtransforms"][0]   # il warp .nii.gz
shutil.copyfile(warp_field, f"{MYPIPE_DIR}/{SUBJECT}/anat/{SUBJECT}_MNI_2fs1Warp.nii.gz")
inv_warp_field = reg["invtransforms"][1]   # il warp .nii.gz
shutil.copyfile(inv_warp_field, f"{MYPIPE_DIR}/{SUBJECT}/anat/{SUBJECT}_MNI_2fs1InverseWarp.nii.gz")



ants.image_write(roi_in_T1, f"{MYPIPE_DIR}/{OUTPUT_NII}")
print(f">>> ROI resampled → {MYPIPE_DIR}/{OUTPUT_NII}")


# Esegue mri_vol2vol dentro il container FreeSurfer.

cmd = [
    "docker", "run", "--rm",
    "--user", f"{os.getuid()}:{os.getgid()}",
    "-v", f"{SUBJECTS_DIR}:/subjects",
    "-v", f"{MYPIPE_DIR}:/data",
    FS_DOCKER_IMAGE,
    "mri_vol2vol",
    "--mov", f"/data/{OUTPUT_NII}",
    "--targ", f"/subjects/{SUBJECT}/mri/T1.mgz",
    "--o",   f"/data/{OUTPUT_MGZ}",
    "--lta", f"/data/{SUBJECT}/anat/{SUBJECT}_desc-N4_T1w_2fs.lta",
    "--interp", "nearest"
]

print(">>> Done.")
