#!/usr/bin/env python3
"""
Quick check on what the BigBrainWarp desc-layer* files actually contain:
per-layer thickness (mm), or cumulative depth / boundary position.

Usage:
    python3 inspect_layers.py /path/to/BigBrainWarp/spaces/tpl-fsaverage
"""

import glob
import os
import re
import sys

import numpy as np
import nibabel as nib

def read_scalar(path):
    if path.endswith((".gii", ".shape.gii", ".func.gii")):
        return np.asarray(nib.load(path).darrays[0].data, dtype=float)
    if path.endswith(".txt"):
        return np.loadtxt(path, dtype=float)
    return np.asarray(nib.freesurfer.read_morph_data(path), dtype=float)


def main(d, hemi="L"):
    print(f"directory: {d}\nhemisphere: {hemi}\n")

    found = {}
    for i in range(1, 7):
        hits = sorted(h for h in glob.glob(os.path.join(d, f"*hemi-{hemi}*"))
                      if re.search(rf"layer{i}(?![0-9])", os.path.basename(h)))
        if not hits:
            print(f"!! no file matched for layer {i}")
            return
        if len(hits) > 1:
            print(f"   layer {i}: {len(hits)} matches, using {os.path.basename(hits[0])}")
        found[i] = hits[0]

    print("files:")
    for i, p in found.items():
        print(f"  layer{i}  {os.path.basename(p)}")
    print()

    arrs = [read_scalar(found[i]) for i in range(1, 7)]
    lens = {a.size for a in arrs}
    if len(lens) != 1:
        print(f"!! different lengths across layers: {lens}")
        return
    n = arrs[0].size
    A = np.vstack(arrs)                      # (6, n)
    print(f"vertices: {n}   (fsaverage 164k = 163842)")
    print(f"dtype/range overall: min {np.nanmin(A):.4f}  max {np.nanmax(A):.4f}")
    print(f"non-finite values: {int((~np.isfinite(A)).sum())}")
    print(f"exact zeros: {int((A == 0).sum())}  "
          f"({100 * (A == 0).mean():.1f}% of all entries)\n")

    print(f"  {'layer':<8}{'mean':>10}{'median':>10}{'p5':>10}{'p95':>10}{'max':>10}")
    for i in range(6):
        a = A[i][np.isfinite(A[i])]
        nz = a[a != 0]
        src = nz if nz.size else a
        print(f"  {i+1:<8}{src.mean():>10.4f}{np.median(src):>10.4f}"
              f"{np.percentile(src, 5):>10.4f}{np.percentile(src, 95):>10.4f}"
              f"{src.max():>10.4f}")

    ok = np.all(np.isfinite(A), axis=0) & (A.sum(axis=0) > 0)
    B = A[:, ok]
    print(f"\nusable vertices: {ok.sum()} / {n}")
    print(f"sum over the 6 layers: mean {B.sum(axis=0).mean():.4f}  "
          f"p5 {np.percentile(B.sum(axis=0), 5):.4f}  "
          f"p95 {np.percentile(B.sum(axis=0), 95):.4f}")

    # monotonic layer1 <= layer2 <= ... would indicate cumulative depth
    mono = np.mean(np.all(np.diff(B, axis=0) >= 0, axis=0))
    print(f"vertices with layer1<=layer2<=...<=layer6: {100 * mono:.1f}%")
    print(f"max value per vertex: mean {B.max(axis=0).mean():.4f}")

    print("\n--- interpretation ---")
    s = B.sum(axis=0).mean()
    if mono > 0.9:
        print("Values are monotonically increasing across layers at nearly every")
        print("vertex -> these are CUMULATIVE depths / boundary positions, not")
        print("per-layer thicknesses. Convert with np.diff before computing")
        print("fractions (layer_1 thickness = layer1, layer_i = layer_i - layer_{i-1},")
        print("or the reverse ordering - check which end is the pial surface).")
    elif 1.2 < s < 5.0:
        print(f"The 6 values sum to {s:.2f} per vertex on average, in the range of")
        print("human cortical thickness in mm -> these are PER-LAYER THICKNESSES.")
        print("Use them directly.")
    elif 0.9 < s < 1.1:
        print(f"The 6 values sum to {s:.3f} -> these are already RELATIVE")
        print("FRACTIONS of cortical thickness. Skip the normalisation step.")
    else:
        print(f"The 6 values sum to {s:.4f} per vertex, which matches neither mm")
        print("thickness nor fractions. Check the units before going further.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "L")