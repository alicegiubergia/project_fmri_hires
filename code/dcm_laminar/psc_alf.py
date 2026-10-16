#!/usr/bin/env python3
"""
psc_alf.py — preparazione dei 4D per LN2_DEVEIN (LayNii), a basso uso di memoria

Due modalità:
  psc  : converte un run 4D in percent signal change voxel-wise
         (100 * (x - media) / media), così che il deveining agisca sulle
         fluttuazioni e non sull'intensità di base, che varia con la profondità.
  alf  : stima l'ampiezza delle basse frequenze (ALF) voxel-wise,
         usata da LN2_DEVEIN come proxy del volume venoso (CBV) per strato.

Il 4D non viene mai caricato intero: si lavora una fetta assiale (z) alla volta.
Se l'input è .nii.gz viene prima decompresso in un .nii temporaneo accanto
all'output (leggere fette da un .gz costringerebbe a decomprimere tutto il file
a ogni fetta); il temporaneo viene cancellato alla fine.

Uso:
  python psc_alf.py psc  in_4d.nii[.gz]  out_psc.nii[.gz]  [--mask mask.nii.gz]
  python psc_alf.py alf  in_4d.nii[.gz]  out_alf.nii[.gz]  --tr 2.0 [--band 0.01 0.08] [--mask mask.nii.gz]

Nota sull'ALF: va calcolata su dati senza risposta al task (residui del GLM o un
run di riposo), perché la potenza indotta dal task nelle basse frequenze
gonfierebbe l'ALF proprio nei voxel attivati.
"""
import argparse
import gzip
import os
import shutil
import tempfile
import numpy as np
import nibabel as nib


def open_4d(path, tmpdir):
    """Apre il 4D in modo che le fette si leggano da disco (memmap)."""
    if path.endswith(".gz"):
        tmp = os.path.join(tmpdir, "tmp_4d.nii")
        print(f"  decompressione {os.path.basename(path)} -> file temporaneo ...", flush=True)
        with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
        path = tmp
    img = nib.load(path, mmap=True)
    if img.ndim != 4:
        raise SystemExit(f"{path}: atteso un 4D, trovato {img.shape}")
    return img


def load_mask(mask_path, shape3):
    if not mask_path:
        return None
    m = np.asarray(nib.load(mask_path).dataobj) > 0
    if m.shape != shape3:
        raise SystemExit(f"maschera {m.shape} diversa dal 4D {shape3}")
    return m


def save(arr, ref_img, out, is4d):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    if not is4d:
        hdr.set_data_shape(arr.shape)
    nib.save(nib.Nifti1Image(arr, ref_img.affine, hdr), out)


def do_psc(img, out, mask):
    nx, ny, nz, nt = img.shape
    res = np.zeros((nx, ny, nz, nt), dtype=np.float32)   # l'output deve comunque stare in RAM una volta
    for z in range(nz):
        d = np.asarray(img.dataobj[:, :, z, :], dtype=np.float32)    # nx x ny x nt
        m = d.std(axis=-1) > 0
        if mask is not None:
            m &= mask[:, :, z]
        if not m.any():
            continue
        x = d[m]
        mu = x.mean(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 100.0 * (x - mu) / mu
        r[~np.isfinite(r)] = 0
        res[:, :, z, :][m] = r
    save(res, img, out, True)


def do_alf(img, out, tr, band, mask):
    nx, ny, nz, nt = img.shape
    t = np.arange(nt, dtype=np.float64)
    X = np.vstack([np.ones_like(t), t]).T
    P = X @ np.linalg.pinv(X)                       # proiezione sul trend lineare
    freqs = np.fft.rfftfreq(nt, d=tr)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any():
        raise SystemExit(f"Nessuna frequenza nella banda {band} con TR={tr} e {nt} volumi")
    alf = np.zeros((nx, ny, nz), dtype=np.float32)
    for z in range(nz):
        d = np.asarray(img.dataobj[:, :, z, :], dtype=np.float32)
        m = d.std(axis=-1) > 0
        if mask is not None:
            m &= mask[:, :, z]
        if not m.any():
            continue
        ts = d[m].astype(np.float64)
        ts = ts - ts @ P.T                          # detrend lineare
        amp = np.abs(np.fft.rfft(ts, axis=1)) / nt
        alf[:, :, z][m] = amp[:, sel].mean(axis=1)
        if z % 10 == 0:
            print(f"  fetta {z+1}/{nz}", flush=True)
    save(alf, img, out, False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["psc", "alf"])
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--mask", default=None)
    p.add_argument("--tr", type=float, default=None)
    p.add_argument("--band", type=float, nargs=2, default=[0.01, 0.08])
    a = p.parse_args()
    if a.mode == "alf" and a.tr is None:
        raise SystemExit("--tr è obbligatorio per alf")

    outdir = os.path.dirname(os.path.abspath(a.out))
    with tempfile.TemporaryDirectory(dir=outdir) as tmpdir:
        img = open_4d(a.inp, tmpdir)
        mask = load_mask(a.mask, img.shape[:3])
        if a.mode == "psc":
            do_psc(img, a.out, mask)
        else:
            do_alf(img, a.out, a.tr, a.band, mask)
        del img
