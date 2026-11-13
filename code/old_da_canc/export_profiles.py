"""
UNICA sorgente dei profili laminari PSC (tre livelli + con_0001), in formato long.

Scrive un solo file per configurazione:

    <ROOT_DIR>/group_layer_modelli/<METHOD>/profili_<METHOD>.csv

e lo leggono tutti gli script di analisi:
  - modelli_laminari.r          (Rscript modelli_laminari.r <csv>)
  - layer_ff_fb_analysis.py     (via carica_o_estrai)
  - group_layer_fitting_psc.py  (via carica_o_estrai)

La metrica di profondita' (equidist/equivol) si sceglie in
group_layer_plotting_psc.py, oppure da riga di comando:
    LAYER_METHOD=equivol python export_profiles.py

con_0001 viene campionato con metrica e ROI della famiglia (mirroring -> spmT_0008,
spacing -> spmT_0009). In glp la mappa PSC dipende solo dal contrasto, mentre metrica
e ROI arrivano come argomenti: basta passare i path della famiglia con contrasto spmT_0001.

Uso:
    python export_profiles.py            # estrae e (ri)scrive il CSV
    python export_profiles.py --if-missing # estrae solo se il CSV non esiste
"""
import os
import sys
import numpy as np
import pandas as pd
import group_layer_plotting_psc as glp

OUT_DIR = os.path.join(glp.ROOT_DIR, 'group_layer_modelli', glp.OUT_TAG)
CSV_PATH = os.path.join(OUT_DIR, f'profili_{glp.OUT_TAG}.csv')

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
                    print(f"    [SKIP] {sub} {famiglia} {regione}: metrica o ROI mancante "
                          f"({os.path.basename(metric_path)})")
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
                               'psc': prof[b], 'n_voxel': int(counts[b]), 'x': x,
                               'method': glp.METHOD}
                              for b in range(glp.NR_LAYERS)]

    return pd.DataFrame(righe).dropna(subset=['psc'])


def esporta(design=None):
    """Estrae i profili e scrive il CSV unico. Restituisce il DataFrame."""
    if design is None:
        design = glp.load_design()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = carica_profili(design)
    if df.empty:
        raise SystemExit("Nessun profilo estratto")
    df.to_csv(CSV_PATH, index=False)
    print(df.groupby(['family', 'condition', 'region'])['subject'].nunique().unstack())
    print(f"\nSalvato: {CSV_PATH}")
    return df


def carica_o_estrai(design=None):
    """Punto di accesso per gli altri script.

    Legge il CSV della configurazione corrente se esiste, altrimenti lo crea.
    Controlla che il file corrisponda davvero al METHOD corrente, cosi' non si
    mescolano equidist ed equivol per errore.
    """
    if not os.path.exists(CSV_PATH):
        print(f"[export_profiles] {CSV_PATH} non esiste: lo creo.")
        return esporta(design)

    df = pd.read_csv(CSV_PATH, float_precision='round_trip')
    for col, atteso in (('method', glp.METHOD),):
        if col not in df.columns:
            raise ValueError(f"{CSV_PATH}: manca la colonna '{col}' (file della "
                             "versione precedente?). Cancellalo e rigenera.")
        trovati = set(df[col].unique())
        if trovati != {atteso}:
            raise ValueError(f"{CSV_PATH}: {col}={trovati}, atteso {atteso}.")
    print(f"[export_profiles] profili letti da {CSV_PATH} "
          f"({df['subject'].nunique()} soggetti, {len(df)} righe)")
    return df


def profilo(g, nr_layers=glp.NR_LAYERS):
    """Righe long di UN profilo -> (psc[nr_layers], n_voxel[nr_layers]).

    I bin assenti (eliminati da dropna in esportazione) tornano NaN / 0.
    """
    prof = np.full(nr_layers, np.nan)
    counts = np.zeros(nr_layers, dtype=int)
    b = g['bin'].to_numpy(dtype=int) - 1
    prof[b] = g['psc'].to_numpy(dtype=float)
    counts[b] = g['n_voxel'].to_numpy(dtype=int)
    return prof, counts


if __name__ == '__main__':
    if '--if-missing' in sys.argv[1:] and os.path.exists(CSV_PATH):
        print(f"Esiste gia': {CSV_PATH} (niente da fare)")
    else:
        esporta()
    print(f"Ora: Rscript modelli_laminari.r {CSV_PATH}")