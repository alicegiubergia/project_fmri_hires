#!/usr/bin/env python3
"""
Convert Freesurfer segmentations to LayNii rim files.

Usage:
    python freesurfer_segmentation_to_rim.py /path/to/aseg_0pt5.nii.gz
"""

import os
import nibabel as nb
import numpy as np
from scipy.ndimage import morphology, generate_binary_structure
import argparse

# =============================================================================
# Argument parser
parser = argparse.ArgumentParser(
    description="Convert Freesurfer aseg segmentation to LayNii RIM format"
)
parser.add_argument(
    "input_nii",
    type=str,
    help="Path to Freesurfer aseg NIfTI file (e.g. aseg_0pt5.nii.gz)"
)

args = parser.parse_args()
INPUT = args.input_nii
WM_LABELS = [2, 41]
GM_LABELS = [3, 42]

# =============================================================================
nii = nb.load(INPUT)
data = np.asarray(nii.dataobj)

# -----------------------------------------------------------------------------
# Fill in white matter but only as a border
rim_wm = np.zeros(data.shape, dtype="int32")
for i in WM_LABELS:
    rim_wm[data == i] = 1
struct = generate_binary_structure(3, 3)
rim_inner = morphology.binary_erosion(rim_wm, structure=struct, iterations=2)
rim_inner = rim_inner - rim_wm

# -----------------------------------------------------------------------------
# Fill in gray matter
rim_gm = np.zeros(data.shape, dtype="int32")
for i in GM_LABELS:
    rim_gm[data == i] = 3

# -----------------------------------------------------------------------------
# Generate an outer gray matter border
rim_out = morphology.binary_dilation(rim_gm, structure=struct, iterations=2)

# -----------------------------------------------------------------------------
# Collate all three labels into one in order
rim = np.zeros(data.shape, dtype="int32")
rim[rim_out != 0] = 1
rim[rim_inner != 0] = 2
rim[rim_gm != 0] = 3

print('Saving...')
basename, ext = nii.get_filename().split(os.extsep, 1)
out = nb.Nifti1Image(rim, header=nii.header, affine=nii.affine)
nb.save(out, "{}_{}.{}".format(basename, "rim", ext))

print("Finished.")