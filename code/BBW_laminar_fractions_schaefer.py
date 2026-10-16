#!/usr/bin/env python3

"""
Percentage of laminar thickness (layers I-VI) per Schaefer parcel,
aggregated into 3 ROIs (dorsal / ventral / visual).

Laminar data source: the BigBrain 6-layer atlas (Wagstyl et al., 2020, PLoS Biol 18:e3000678), 
resampled to fsaverage by the BigBrainWarp toolbox (Paquola et al., 2021, eLife 10:e70119),
files  spaces/tpl-fsaverage/*desc-layer{1..6}_thickness*.)

Pipeline:
    vertex  f_i = t_i / sum_{j=1..6} t_j    (i = I..VI)
    parcel  P_i = mean of f_i over the parcel's vertices
    ROI     mean and sd of P_i across the parcels of the ROI

Outputs: 
    laminar_fractions_perparcel.csv     one row per Schaefer parcel
    laminar_fractions_roi.csv           one row per ROI x layer

Usage: 
    python laminar_fractions_schaefer.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BBW_FSAVERAGE_DIR = Path('/mnt/store1/projects/progetto_layer/BigBrainWarp/spaces_fsaverage/')

# What the desc-layer* files hold:
#   "thickness"  = per-layer thickness, one value per layer
#   "cumulative" = cumulative depth / boundary position, differenced internally
#   "auto"       = decide from the data (monotonic across layers -> cumulative)
# Run inspect_layers.py once to confirm, then pin this to the explicit value.
LAYER_MODE = "auto"

SCHAEFER_DIR = (Path('/mnt/store1/projects/progetto_layer/parcels/Schaefer_FreeSurfer5.3/'))
N_PARCELS = 200
NETWORKS = "17Networks"

ROIS = {
    "visual":  [r"Vis"],
    "dorsal":  [r"Dors"],
    "ventral": [r"Vent"],
} 

HEMIS = ["L", "R"]
OUT_DIR = Path('/mnt/store1/projects/progetto_layer/BigBrainWarp/myoutputs/')

LAYER_NAMES = ["I", "II", "III", "IV", "V", "VI"]
MIN_VERTICES = 20       # numero minimo di vertici per parcel (per calcolare le statistiche)

def _read_surface_scalar(path):
    """Read a surface scalar file (.curv, .txt, .gii/.shape.gii)."""
    path = Path(path)
    suffixes = ''.join(path.suffixes)
    if suffixes.endswith((".gii", ".shape.gii", ".func.gii")):
        return np.asarray(nib.load(path).darrays[0].data, dtype=float)
    if path.suffix == ".txt":
        return np.loadtxt(path, dtype=float)
    return np.asarray(nib.freesurfer.read_morph_data(path), dtype=float)

def load_layer_thickness(hemi, bbw_dir=BBW_FSAVERAGE_DIR):
    """Return (6, n_vertices) with the thickness of layers in mm.
    
    The desc-layer* files hold per value thickness."""

    bbw_dir = Path(bbw_dir)
    layers = []
    for i in range(1, 7):
        hits = sorted(
            (h for h in bbw_dir.glob(f"*hemi-{hemi}*")
             if re.search(rf"layer{i}(?![0-9])", h.name)),
            key=lambda h: str(h),
        )
        if not hits:
            listing = (sorted(path.name for path in bbw_dir.iterdir())[:10]
                                             if bbw_dir.is_dir() else "DIRECTORY NOT FOUND")
            raise FileNotFoundError(
                f"No file found for layer {i}, hemisphere {hemi}.\n"
                                f"  pattern searched: '*hemi-{hemi}*' containing 'layer{i}'\n"
                f"  directory contents: {listing}")
        if len(hits) > 1:
            print(f"[warning] several files for layer {i} hemi-{hemi}, "
                                    f"using {hits[0].name}", file=sys.stderr)
        layers.append(_read_surface_scalar(hits[0]))

    lens = {len(l) for l in layers}
    if len(lens) != 1:
        raise ValueError(f"The 6 layers have different lengths: {lens}")
    A = np.vstack(layers)

    # sanity check: the 6 thicknesses should sum to a cortical thickness in mm
    ok = np.all(np.isfinite(A), axis=0) & (A.sum(axis=0) > 0)
    s = A[:, ok].sum(axis=0).mean()
    if not 1.2 < s < 5.0:
        print(f"[warning] hemi-{hemi}: the 6 layers sum to {s:.3f} per vertex, "
              f"outside the range expected for cortical thickness in mm. "
              f"Check what these files actually hold.", file=sys.stderr)
    return A

def load_schaefer(hemi, schaefer_dir=SCHAEFER_DIR, n_parcels=N_PARCELS, networks=NETWORKS):
    """Return (per-vertex labels, parcel_names) for one hemisphere."""
    fs_hemi = {"L": "lh", "R": "rh"}[hemi]
    fname = Path(schaefer_dir) / (
        f"{fs_hemi}.Schaefer2018_{n_parcels}Parcels_{networks}_order.annot"
    )
    if not fname.is_file():
        raise FileNotFoundError(f"Annot file not found: {fname}")
    labels, _, names = nib.freesurfer.read_annot(fname)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(labels), names

def vertex_fractions(thick):
    """(6, V) thickness -> (6, V) relative fractions; NaN where undefined."""
    thick = np.asarray(thick, dtype=float)
    tot = thick.sum(axis=0)
    ok = np.isfinite(tot) & (tot > 0) & np.all(np.isfinite(thick), axis=0)
    frac = np.full_like(thick, np.nan)
    frac[:, ok] = thick[:, ok] / tot[ok]
    return frac, tot, ok

def assign_roi(name, rois=ROIS):
    """Name of the ROI this parcel belongs to, or None."""
    hit = [roi for roi, pats in rois.items() if any(re.search(p, name) for p in pats)]
    if not hit: return None
    if len(hit) > 1:
        print(f"[warning] '{name}' matches several ROIs {hit}, assigned t {hit[0]}", file=sys.stderr)
    return hit[0]

def per_parcel_table(bbw_dir=BBW_FSAVERAGE_DIR, schaefer_dir=SCHAEFER_DIR,
                     rois=ROIS, hemis=HEMIS,
                     n_parcels=N_PARCELS, networks=NETWORKS,
                     all_parcels=False):
    """One row per Schaefer parcel. 
    all_parcels=True also returns parcels outside the three ROIs."""

    rows = []
    for hemi in hemis:
        thick = load_layer_thickness(hemi, bbw_dir)
        labels, names = load_schaefer(hemi, schaefer_dir, n_parcels, networks)

        if thick.shape[1] != labels.shape[0]:
            raise ValueError(
                f"Vertex count mismatch hor hemi-{hemi}: thickness "
                f"{thick.shape[1]}, annot {labels.shape[0]}. Both files must "
                f"be at the same fsaverage density (164k = 163842 vertices.)"
            )
        frac, tot, ok = vertex_fractions(thick)

        for pi, nm in enumerate(names):
            if "Medial_Wall" in nm or nm.startswith("Background"):
                continue
            roi = assign_roi(nm, rois)
            if roi is None and not all_parcels:
                continue
            m = (labels == pi) & ok
            if m.sum() < MIN_VERTICES:
                continue
            f = 100.0 * frac[:, m]
            row = {"roi": roi, "hemi": hemi, "parcel": nm,
                   "n_vertices": int(m.sum()),
                   "thickness_mm": float(tot[m].mean())}
            for li, lname in enumerate(LAYER_NAMES):
                row[f"L{lname}_pct"] = float(f[li].mean())
                row["supra_I_III_pct"] = float(f[:3].sum(axis=0).mean())
                row["gran_IV_pct"] = float(f[3].mean())
                row["infra_V_VI_pct"] = float(f[4:].sum(axis=0).mean())
                rows.append(row)
    return pd.DataFrame(rows)

def roi_table(per_parcel):
    """Aggregate parcels into ROIs: mean and sd computed across parcels,
    plus a mean weighted by parcel vertex count."""
    rows = []
    sub_all = per_parcel[per_parcel["roi"].notna()]
    for roi, sub in sub_all.groupby("roi"):
        w = sub["n_vertices"].to_numpy(dtype=float)
        for lname in LAYER_NAMES + ["supra_I_III", "gran_IV", "infra_V_VI"]:
            col = f"L{lname}_pct" if lname in LAYER_NAMES else f"{lname}_pct"
            x = sub[col].to_numpy(dtype=float)
            rows.append({
                "roi": roi,
                "layer": lname,
                "mean_pct": x.mean(),
                "sd_pct": x.std(ddof=1) if x.size > 1 else np.nan,
                "sem_pct": (x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan,
                "min_pct": x.min(),
                "max_pct": x.max(),
                "mean_pct_wtd": float(np.average(x, weights=w)),
                "n_parcels": int(x.size),
                "n_vertices": int(w.sum()),
            })
    return pd.DataFrame(rows)

def run(bbw_dir=BBW_FSAVERAGE_DIR, schaefer_dir=SCHAEFER_DIR,
        rois=ROIS, hemis=HEMIS, out_dir=OUT_DIR,
        n_parcels=N_PARCELS, networks=NETWORKS,
        all_parcels=False, write=True):
    pp = per_parcel_table(bbw_dir, schaefer_dir, rois, hemis,
                          n_parcels, networks, all_parcels)
    if pp.empty:
        raise RuntimeError("No parcel selected.")
    rt = roi_table(pp)
    if write:
        os.makedirs(out_dir, exist_ok=True)
        pp.to_csv(Path(out_dir) / ("laminar_fractions_perparcel.csv"), index=False)
        rt.to_csv(Path(out_dir) / ("laminar_fractions_roi.csv"), index=False)
    return pp, rt

def print_roi_table(rt, n_bins=10, deep_is_bin1=True):
    """Readtable table + cut-points on the laminar bins."""
    if rt.empty:
        print("No results.")
        return
    for roi, sub in rt.groupby("roi"):
        s = sub.set_index("layer")
        npar = int(s["n_parcels"].iloc[0])
        print(f"\n{roi} ({npar} parcels)")
        print(f"    {'layer':<12}{'mean %':>10}{'sd %':>8}{'wtd %':>10}"
              f"{'range %':>16}")
        for lname in LAYER_NAMES:
            r = s.loc[lname]
            print(f"    {lname:<12}{r['mean_pct']:>10.2f}{r['sd_pct']:>8.2f}"
                f"{r['mean_pct_wtd']:>10.2f}"
                f"{r['min_pct']:>8.1f}-{r['max_pct']:<7.1f}")
        print(f"  {'sum':<12}{s.loc[LAYER_NAMES, 'mean_pct'].sum():>10.2f}")
        for lname in ["supra_I_III", "gran_IV", "infra_V_VI"]:
            r = s.loc[lname]
            print(f"  {lname:<12}{r['mean_pct']:>10.2f}{r['sd_pct']:>8.2f}"
                  f"{r['mean_pct_wtd']:>10.2f}")

    where = "1 = deep (WM)" if deep_is_bin1 else "1 = superficial (pial)"
    print(f"\nCut-points over {n_bins} bins, {where}:")
    for roi, sub in rt.groupby("roi"):
        s = sub.set_index("layer")["mean_pct"]
        infra, gran = s["infra_V_VI"], s["gran_IV"]
        if deep_is_bin1:
            c1, c2 = n_bins * infra / 100.0, n_bins * (infra + gran) / 100.0
            lo, mid, hi = "infra", "granular", "supra"
        else:
            supra = s["supra_I_III"]
            c1, c2 = n_bins * supra / 100.0, n_bins * (supra + gran) / 100.0
            lo, mid, hi = "supra", "granular", "infra"
        print(f"  {roi:<10} {lo} -> bin {c1:.2f}, {mid} -> bin {c2:.2f}, "
              f"{hi} beyond")

if __name__ == "__main__":
    pp, rt = run()
    print_roi_table(rt)
    print(f"\n{len(pp)} parcels -> laminar_fractions_perparcel.csv")
    print(f"{len(rt)} rows     -> laminar_fractions_roi.csv")
    print(f"in {os.path.abspath(OUT_DIR)}")