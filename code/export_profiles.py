"""
Esporta i profili PSC (tre livelli + con_0001) in formato long per modelli_laminari_ar1.R.

con_0001 viene campionato con metrica e ROI della famiglia (mirroring -> spmT_0008,
spacing -> spmT_0009). In glp la mappa PSC dipende solo dal contrasto, mentre metrica
e ROI arrivano come argomenti: basta passare i path della famiglia con contrasto spmT_0001.
"""
import os
import numpy as np
import pandas as pd
import group_layer_plotting_psc as glp

OUT_DIR = os.path.join(glp.ROOT_DIR, 'group_layer_modelli')

LEVELS = {
    'mirroring': ['spmT_0002', 'spmT_0003', 'spmT_0004'],
    'spacing':   ['spmT_0005', 'spmT_0006', 'spmT_0007'],
}
X_LEVELS = {'mirroring': [1.0, 2.0, 3.0], 'spacing': [2.0, 4.0, 6.0]}
BASELINE = 'spmT_0001'


def _campiona(sub, contrasto, metric_path, roi_path, design):
    try:
        return glp.sample_psc_profile(sub, contrasto, metric_path, roi_path, design)
    except Exception as e:
        print(f"    [SKIP] {sub} {contrasto}: {e}")
        return None, None


def carica_profili(design):
    righe = []
    for famiglia, contrasti in LEVELS.items():
        x_centrati = np.asarray(X_LEVELS[famiglia]) - np.mean(X_LEVELS[famiglia])
        roi_key = glp.TASK[famiglia]

        for regione in glp.ROIS:
            for sub in glp.SUBJECTS:
                paths = {glp.subject_paths(sub, c, regione) for c in contrasti}
                assert len(paths) == 1, f"{sub} {famiglia} {regione}: path diversi tra livelli"
                metric_path, roi_path = paths.pop()
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    print(f"    [SKIP] {sub} {famiglia} {regione}: metrica o ROI mancante")
                    continue

                da_campionare = [(c, 'livello', i, x_centrati[i - 1])
                                 for i, c in enumerate(contrasti, start=1)]
                da_campionare.append((BASELINE, 'baseline', 0, np.nan))

                for contrasto, condizione, livello, x in da_campionare:
                    prof, counts = _campiona(sub, contrasto, metric_path, roi_path, design)
                    if prof is None:
                        continue
                    righe += [{'subject': sub, 'region': regione, 'family': famiglia,
                               'roi_key': roi_key, 'condition': condizione,
                               'contrast': contrasto, 'level': livello, 'bin': b + 1,
                               'psc': prof[b], 'n_voxel': int(counts[b]), 'x': x}
                              for b in range(glp.NR_LAYERS)]

    return pd.DataFrame(righe).dropna(subset=['psc'])


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    df = carica_profili(glp.load_design())
    if df.empty:
        raise SystemExit("Nessun profilo estratto")

    # la modalita' del denominatore entra nel nome: voxelwise e roi_scalar restano separati
    out = os.path.join(OUT_DIR, f'profili_{glp.DENOMINATOR_MODE}.csv')
    df.to_csv(out, index=False)
    print(df.groupby(['family', 'condition', 'region'])['subject'].nunique().unstack())
    print(f"\nSalvato: {out}\nOra: Rscript modelli_laminari_ar1.R {out}")