#!/usr/bin/env python3
"""
Porta le ROI (dorsal / ventral / visual) di ogni soggetto in MNI con
antsApplyTransforms e le combina in un file per condizione (baseline,
mirroring, spacing), per controllarne la distribuzione spaziale.
 
Trasformazioni per soggetto (da hires_pipeline/sub-XXX/anat/):
  sub-XXX_from-T1w_to-MNI_SyN0GenericAffine.mat   (obbligatoria)
  sub-XXX_from-T1w_to-MNI_SyN1Warp.nii.gz         (usata se presente)

Output in <ROOT>/rois_distribution/:
  mni/   ROI dei singoli soggetti in MNI
  con-<cond>_space-MNI_desc-rois_label.nii.gz
        1 = dorsal, 2 = ventral, 3 = visual
        4 = overlap (voxel che cade in >1 ROI, anche di soggetti diversi)
  con-<cond>_space-MNI_desc-rois_count.nii.gz   4D, 3 volumi (dorsal,
        ventral, visual): n. di soggetti con quel voxel nella ROI (0-8)
  rois_fsleyes.lut / rois_freesurfer.txt   colori per i viewer
 
Uso:
  python combine_rois.py            # ref = MNI152NLin6Asym res-01 (TemplateFlow)
  python combine_rois.py --force    # rifà anche i warp già fatti
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
 
import numpy as np
import nibabel as nib
 
DATA = "/mnt/store1/projects/progetto_layer/data/derivatives"
ROOT = os.path.join(DATA, "rois")
XFM_DIR = os.path.join(DATA, "hires_pipeline", "{sub}", "anat")
AFFINE = "{sub}_from-T1w_to-MNI_SyN0GenericAffine.mat"
WARP = "{sub}_from-T1w_to-MNI_SyN1Warp.nii.gz"
 
TF_HOME = os.environ.get(
    "TEMPLATEFLOW_HOME", os.path.expanduser("~/.cache/templateflow"))
REF = os.path.join(TF_HOME, "tpl-MNI152NLin6Asym",
                   "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz")

CONTRASTS = {"baseline": "0001", "mirroring": "0008", "spacing": "0009"}
ROIS = ["dorsal", "ventral", "visual"]
OVERLAP = len(ROIS) + 1  # etichetta per voxel che cade in >1 ROI
COLORS = {
    "dorsal": (255, 0, 0),  # rosso
    "ventral": (0, 255, 0),  # verde
    "visual": (0, 0, 255),  # blu
    "overlap": (255, 255, 0),  # giallo
}

def to_mni(src, sub, ref, outdir, force):
    """Applica warp + affine (T1w -> MNI) a una maschera. Ritorna il path."""
    out = os.path.join(outdir, os.path.basename(src).replace(
        "_con-", "_space-MNI_con-"))
    if os.path.exists(out) and not force:
        return out
    xdir = XFM_DIR.format(sub=sub)
    aff = os.path.join(xdir, AFFINE.format(sub=sub))
    warp = os.path.join(xdir, WARP.format(sub=sub))
    if not os.path.exists(aff):
        sys.exit(f"Affine non trovata: {aff}")
    # antsApplyTransforms applica le -t dall'ultima alla prima:
    # prima l'affine, poi il warp
    xfms = []
    if os.path.exists(warp):
        xfms += ["-t", warp]
    else:
        print(f"  ATTENZIONE: {sub} senza warp SyN, uso solo l'affine")
    xfms += ["-t", aff]
    cmd = ["antsApplyTransforms", "-d", "3", "-i", src, "-r", ref,
           "-o", out, "-n", "GenericLabel"] + xfms
    subprocess.run(cmd, check=True)
    return out


def write_luts(outdir):
    """Scrive i file LUT per fsleyes e freesurfer."""
    names = ROIS + ["overlap"]
    # fsleyes
    with open(os.path.join(outdir, "rois_fsleyes.lut"), "w") as f:
        for i, name in enumerate(names, start=1):
            r, g, b = (c / 255 for c in COLORS[name])
            f.write(f"{i} {r:.3f} {g:.3f} {b:.3f} {name}\n")
    with open(os.path.join(outdir, "rois_freesurfer.txt"), "w") as f:
        f.write("0 Unknown 0 0 0 0\n")
        for i, name in enumerate(names, start=1):
            r, g, b = COLORS[name]
            f.write(f"{i} {name} {r} {g} {b}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF,
                    help="template MNI di riferimento (griglia output); "
                         "default: MNI152NLin6Asym res-01 da TemplateFlow")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"Template non trovato: {args.ref}\n"
                 "Controlla TEMPLATEFLOW_HOME o passa --ref")
    if shutil.which("antsApplyTransforms") is None:
        sys.exit("antsApplyTransforms non trovato nel PATH")

    outdir = args.outdir or os.path.join(args.root, "rois_distribution")
    mnidir = os.path.join(outdir, "mni")
    os.makedirs(mnidir, exist_ok=True)

    ref_img = nib.load(args.ref)
    shape = ref_img.shape[:3]

    for cond, num in CONTRASTS.items():
        print(f"\n=== {cond} ===")
        counts = np.zeros(shape + (len(ROIS),), dtype=np.uint8)
        subs_per_roi = {}

        for k, roi in enumerate(ROIS):
            pattern = os.path.join(args.root, "sub-*", f"sub-*_con-spmT_{num}_label-{roi}_mask.nii.gz")
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"  Nessun file trovato per {roi} ({pattern})")
                continue
            print(f"  {roi}: {len(files)} soggetti")
            subs = [re.search(f"(sub-[^_]+)", os.path.basename(p)).group(1) for p in files]
            subs_per_roi[roi] = subs

            for path, sub in zip(files, subs):
                mni_path = to_mni(path, sub, args.ref, mnidir, args.force)
                mask = np.asarray(nib.load(mni_path).dataobj) > 0
                counts[..., k] += mask.astype(np.uint8)

        allsubs = sorted(set().union(*subs_per_roi.values()))
        for roi, subs in subs_per_roi.items():
            missing = sorted(set(allsubs) - set(subs))
            if missing:
                print(f"  ATTENZIONE: {roi} manca per {len(missing)} soggetti: "
                      f"{', '.join(missing)}")

        present = counts > 0
        n_present = present.sum(axis=-1)
        label = np.zeros(shape, dtype=np.uint8)
        single = n_present == 1
        label[single] = np.argmax(present[single], axis=-1) + 1
        label[n_present > 1] = OVERLAP

        hdr = ref_img.header.copy()
        hdr.set_data_dtype(np.uint16)
        nib.save(nib.Nifti1Image(label, ref_img.affine, hdr),
                 os.path.join(outdir, f"con-{cond}_space-MNI_desc-rois_label.nii.gz"))

    write_luts(outdir)
    print(f"Output in {outdir}")

if __name__ == "__main__":
    main()

           