#!/usr/bin/env python3
"""
n_columns.py - number of LN2_COLUMNS columns for a target column diameter.

Two modes:

  area   For each subject x ROI set x region: area of the mid-GM surface
         inside the ROI, and the number of columns that gives the target
         diameter: N = area / (pi * (D/2)^2).
         The area is measured as the iso-surface metric = 0.5 (marching
         cubes) inside GM, on the same metric file used by LN2_LAYERS.
         A voxel-count estimate from the midGM file is printed alongside,
         as a cross-check.

  check  After LN2_COLUMNS: for every <ROI>_columns<N>.nii in the folder,
         how many of the N_LAYERS layers each column covers with at least one
         GM voxel. N_req = columns requested (from the file name), N_found =
         non-empty columns actually present. The column number is acceptable when
         (nearly) every column covers all layers (criterion of Marsh et al.,
         bioRxiv 2025, doi:10.1101/2025.09.09.674189).

Usage:
  python n_columns.py area  [--diam 2.0]
  python n_columns.py check --folder mirroring
"""
import argparse
import os
import re
import numpy as np
import nibabel as nib
from scipy import ndimage

# --------------------------- CONFIG (same as 13_devein_laynii.sh) ----------
ROOT = '/mnt/store1/projects/progetto_layer/data/derivatives'
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 9)]
CON_OF = {'mirroring': 'spmT_0008', 'spacing': 'spmT_0009', 'baseline': 'spmT_0001'}
ROIS = {'dorsal': 'Dors', 'ventral': 'Ven', 'visual': 'Vis'}
LAYERING = 'equidist'
N_LAYERS = 10
GM, CSF, WM = 3, 1, 2           # rim dseg labels (input of LN2_LAYERS)
# ---------------------------------------------------------------------------


def lay(sub, con, roi, suffix):
    return f'{ROOT}/layer_fmri/{sub}/{sub}_con-{con}_label-{roi}_{suffix}.nii'


def rim_f(sub, con, roi):
    return (f'{ROOT}/rois/{sub}/{sub}_con-{con}_desc-polishCleaned_'
            f'label-{roi}_res-up_dseg.nii.gz')


def bbox(mask, pad=2):
    idx = np.nonzero(mask)
    return tuple(slice(max(i.min() - pad, 0), i.max() + pad + 1) for i in idx)


def mid_area(metric_path, rim_path):
    """Area (mm^2) of the iso-surface metric = 0.5 inside GM."""
    from skimage.measure import marching_cubes, mesh_surface_area

    rim_img = nib.load(rim_path)
    rim = np.rint(np.asanyarray(rim_img.dataobj)).astype(np.int16)
    sl = bbox(rim > 0)
    rim = rim[sl]
    met = np.asarray(nib.load(metric_path).dataobj[sl], dtype=np.float32)
    if met.shape != rim.shape:
        raise ValueError(f'grid mismatch: metric {met.shape} vs rim {rim.shape}')
    spacing = rim_img.header.get_zooms()[:3]

    # which side of GM has metric close to 1? (GM voxels touching CSF)
    gm = rim == GM
    near_csf = gm & ndimage.binary_dilation(rim == CSF)
    csf_high = np.nanmean(met[near_csf]) > 0.5 if near_csf.any() else True
    f = np.where(gm, met, 0.0).astype(np.float32)
    f[rim == CSF] = 1.0 if csf_high else 0.0
    f[rim == WM] = 0.0 if csf_high else 1.0
    f[rim == 0] = 0.5                       # outside the ROI: faces removed below

    verts, faces, _, _ = marching_cubes(f, level=0.5, spacing=spacing)
    # keep faces well inside the labelled region (drops the ROI border)
    inside = ndimage.binary_erosion(rim > 0, structure=np.ones((3, 3, 3)))
    vox = np.rint(verts / np.asarray(spacing)).astype(int)
    vox = np.clip(vox, 0, np.asarray(f.shape) - 1)
    ok_v = inside[vox[:, 0], vox[:, 1], vox[:, 2]]
    keep = ok_v[faces].all(axis=1)
    return mesh_surface_area(verts, faces[keep]), spacing


def midgm_count_area(midgm_path):
    """Cross-check: voxels of the midGM file x face area (rough estimate)."""
    if not os.path.exists(midgm_path):
        return np.nan
    img = nib.load(midgm_path)
    z = img.header.get_zooms()[:3]
    return np.count_nonzero(np.asanyarray(img.dataobj)) * z[0] * z[1]


def do_area(diam):
    a_col = np.pi * (diam / 2) ** 2
    print(f'target diameter {diam} mm -> {a_col:.2f} mm^2 per column\n')
    print(f'{"subject":8s} {"set":9s} {"roi":8s} {"area_mm2":>9s} '
          f'{"N":>6s} {"vox_est":>9s}  d(N=5)  d(N=150)')
    rows = []
    for sub in SUBJECTS:
        for rset, con in CON_OF.items():
            for roi in ROIS:
                mp, rp = lay(sub, con, roi, f'metric_{LAYERING}'), rim_f(sub, con, roi)
                if not (os.path.exists(mp) and os.path.exists(rp)):
                    continue
                try:
                    area, _ = mid_area(mp, rp)
                except Exception as e:
                    print(f'{sub} {rset} {roi}: [ERROR] {e}')
                    continue
                n = int(round(area / a_col))
                est = midgm_count_area(lay(sub, con, roi, f'midGM_{LAYERING}'))
                d5, d150 = (2 * np.sqrt(area / (k * np.pi)) for k in (5, 150))
                print(f'{sub:8s} {rset:9s} {roi:8s} {area:9.1f} {n:6d} '
                      f'{est:9.1f}  {d5:6.1f}  {d150:6.2f}')
                rows.append((sub, rset, roi, area, n, est))
    out = os.path.join(ROOT, 'rois', f'n_columns_d{diam:g}mm_{LAYERING}.tsv')
    with open(out, 'w') as fh:
        fh.write('subject\troi_set\troi\tarea_mm2\tn_columns\tarea_voxcount_mm2\n')
        for r in rows:
            fh.write('{}\t{}\t{}\t{:.1f}\t{}\t{:.1f}\n'.format(*r))
    print(f'\n-> {out}')


def do_check(folder):
    """Check every <P>_columns<N>.nii found in roi-deveined/<sub>/<folder>.

    N is read from the file name (13_devein_laynii.sh writes one file per
    column number, e.g. Dors_columns101.nii), so runs with different N can
    sit in the same folder and are all reported.
    """
    pat = re.compile(r'^(?P<P>[A-Za-z]+)_columns(?P<N>\d+)\.nii(\.gz)?$')
    print(f'{"subject":8s} {"roi":5s} {"N_req":>6s} {"N_found":>7s} '
          f'{"all_layers":>10s} {"median_layers":>13s} {"min_layers":>10s}')
    found_any = False
    for sub in SUBJECTS:
        d = os.path.join(ROOT, 'roi-deveined', sub, folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            mt = pat.match(fn)
            if not mt or mt['P'] not in ROIS.values():
                continue
            P, n_req = mt['P'], int(mt['N'])
            lf = os.path.join(d, f'{P}_layers.nii.gz')
            rf = os.path.join(d, f'{P}_rim.nii.gz')
            if not (os.path.exists(lf) and os.path.exists(rf)):
                print(f'{sub:8s} {P:5s} {n_req:6d}  [layers/rim missing]')
                continue
            C = np.rint(np.asanyarray(nib.load(os.path.join(d, fn)).dataobj)).astype(int)
            L = np.rint(np.asanyarray(nib.load(lf).dataobj)).astype(int)
            R = np.rint(np.asanyarray(nib.load(rf).dataobj)).astype(int)
            if not (C.shape == L.shape == R.shape):
                print(f'{sub:8s} {P:5s} {n_req:6d}  [grid mismatch]')
                continue
            m = (C > 0) & (R == GM) & (L >= 1) & (L <= N_LAYERS)
            if not m.any():
                print(f'{sub:8s} {P:5s} {n_req:6d}  [no GM voxels in columns]')
                continue
            pairs = np.unique(np.stack([C[m], L[m]]), axis=1)
            cols, n_lay = np.unique(pairs[0], return_counts=True)
            full = np.mean(n_lay == N_LAYERS)
            print(f'{sub:8s} {P:5s} {n_req:6d} {cols.size:7d} {100 * full:9.1f}% '
                  f'{np.median(n_lay):13.0f} {n_lay.min():10d}')
            found_any = True
    if not found_any:
        print(f'no <ROI>_columns<N>.nii files found under roi-deveined/<sub>/{folder}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['area', 'check'])
    ap.add_argument('--diam', type=float, default=2.0, help='target column diameter (mm)')
    ap.add_argument('--folder', default='mirroring', help='roi-deveined/<sub>/<folder> (check mode)')
    a = ap.parse_args()
    do_area(a.diam) if a.mode == 'area' else do_check(a.folder)