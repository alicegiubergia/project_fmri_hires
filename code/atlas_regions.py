#!/usr/bin/env python3
"""
Dall'atlante Yan2023 (200 parcels, Yeo 17 networks, MNI152NLin6Asym 1mm)
crea una maschera per regione (dorsal / ventral / visual), unendo tutte le
parcelle il cui nome nella LUT contiene una delle stringhe di REGION_LABELS.

Output in group_check/:
  atlas-Yan2023_space-MNI_label-<roi>_mask.nii.gz
da visualizzare con mricrogl_rois.py (PREFIX = "atlas-Yan2023").
"""
import os
import sys

import numpy as np
import nibabel as nib

PARC_DIR = "/mnt/store1/projects/progetto_layer/parcels/Yan2023_MNI152NLin6Asym"
ATLAS = os.path.join(PARC_DIR, "200Parcels_Yeo2011_17Networks_FSLMNI152_1mm.nii.gz")
LUT = os.path.join(PARC_DIR, "fsleyes_lut", "200Parcels_Yeo2011_17Networks_LUT.txt")
OUTDIR = "/mnt/store1/projects/progetto_layer/data/derivatives/rois/rois_distribution"
PREFIX = "atlas-Yan2023"

REGION_LABELS = {"visual": ["Vis"], "ventral": ["Vent"], "dorsal": ["Dors"]}


def read_lut(path):
    """LUT fsleyes: 'indice r g b nome' -> {indice: nome}"""
    labels = {}
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 2 or tok[0].startswith("#"):
                continue
            labels[int(tok[0])] = tok[-1]
    return labels


def main():
    labels = read_lut(LUT)
    img = nib.load(ATLAS)
    data = np.asarray(img.dataobj).astype(int)

    # assegnazione parcella -> regione, controllando che non finisca in due regioni
    assigned = {}
    for roi, keys in REGION_LABELS.items():
        for idx, name in labels.items():
            if any(k in name for k in keys):
                if idx in assigned:
                    sys.exit(f"{name} corrisponde sia a {assigned[idx]} che a {roi}")
                assigned[idx] = roi

    os.makedirs(OUTDIR, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    for roi in REGION_LABELS:
        idxs = sorted(i for i, r in assigned.items() if r == roi)
        mask = np.isin(data, idxs).astype(np.uint8)
        print(f"\n{roi}: {len(idxs)} parcelle, {int(mask.sum())} voxel")
        for i in idxs:
            print(f"  {i:4d} {labels[i]}")
        nib.save(nib.Nifti1Image(mask, img.affine, hdr),
                 os.path.join(OUTDIR, f"{PREFIX}_space-MNI_label-{roi}_mask.nii.gz"))

    print(f"\nOutput in {OUTDIR}")


if __name__ == "__main__":
    main()