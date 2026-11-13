"""
06 - ROI masking in functional space, for the region-level DCM.

The ROIs (dseg, label 3 = GM) are on the 0.2 upsampled grid, while the DCM time-series are extracted from the first-level images, in the native grid. 
For each native voxel this script computes the FRACTION of its volume
occupied by ROI GM sub-voxels. dcm06_region.m uses this fraction as a
weight: the weighted mean over native voxels equals the mean over GM
sub-voxels (exactly with nearest-neighbour upsampling, approximately with
the linear upsampling used for the PSC maps).
 
One mask per subject x family (ROI of spmT_0008 / spmT_0009) x region:
    <ROOT_DIR>/laminar_psc/dcm_region/masks/<sub>/
        <sub>_roi-<family>_label-<region>_desc-gmfrac_mask.nii
 
Saved as uncompressed .nii for SPM12.
They do not depend on the depth metric (LAYER_METHOD is irrelevant here).
 
Usage:  python 06_dcm_masks.py
"""

import os
import numpy as np
import nibabel as nib

import laminar_psc as lps

OUT_DIR = os.path.join(lps.DCM_DIR, 'masks')
FAMILIES = ['mirroring', 'spacing']

def native_reference(sub):
    """Return the reference image in native space, for a given subject."""
    spm_dir = lps.SPM_DIR_PATTERN.format(sub=sub)
    path = lps._find(spm_dir, 'mask') or lps._find(spm_dir, 'con_0001')
    if path is None:
        raise FileNotFoundError(f"no reference image in {spm_dir}")
    return nib.load(path)

def gm_fraction(dseg_path, ref):
    """Compute the fraction of a voxel's volume occupied by ROI GM sub-voxels."""
    dseg = nib.load(dseg_path)
    gm = np.asarray(dseg.dataobj) == 3
    ijk = np.argwhere(gm)
    world = nib.affines.apply_affine(dseg.affine, ijk)
    nat = np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine), world)).astype(int)
    shape = np.array(ref.shape[:3])
    inside = np.all((nat >= 0) & (nat < shape), axis=1)
    counts = np.zeros(tuple(shape), dtype=np.float32)
    np.add.at(counts, tuple(nat[inside].T), 1)
    sub_per_native = (abs(np.linalg.det(ref.affine[:3, :3]))
                      / abs(np.linalg.det(dseg.affine[:3, :3])))
    return counts / sub_per_native, int(gm.sum()), int((~inside).sum())

def main():
    print("output directory:", OUT_DIR)
    for sub in lps.SUBJECTS:
        try:
            ref = native_reference(sub)
        except FileNotFoundError as e:
            print(f"    [SKIP] {sub}: {e}")
            continue  
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
        for fam in FAMILIES:
            for reg in lps.ROIS:
                _, roi_path = lps.subjects_paths(sub, lps.TASK[fam], reg)
                if not os.path.exists(roi_path):
                    print(f"    [SKIP] {sub} {fam} {reg}: missing ROI")
                    continue
                frac, n_sub, n_out = gm_fraction(roi_path, ref)
                out = os.path.join(OUT_DIR, sub,
                                   f"{sub}_roi-{fam}_label-{reg}_desc-gmfrac_mask.nii")
                nib.save(nib.Nifti1Image(frac, ref.affine), out)
                nz = frac > 0
                print(f"  {sub} {fam:9s} {reg:7s}: {n_sub:7d} GM sub-voxels -> "
                      f"{nz.sum():4d} native voxels (equivalent volume "
                      f"{frac.sum():6.1f} voxels; max fraction {frac.max():.2f})"
                      + (f"  [WARNING] {n_out} sub-voxels outside the grid" if n_out else ""))

if __name__ == '__main__':
    main()


    