"""
Profili laminari di gruppo su percent signal change (PSC).

Modifiche rispetto alla versione precedente:
  1. Le mappe campionate sono PSC calcolato dai con_00XX, non spmT.
  2. Nessuna soglia statistica: si campionano tutti i voxel di GM dentro la ROI.
  3. SF (scaling factor) e indici dei beta costanti letti da SPM.mat, per
     soggetto: il numero di regressori di nuisance (outlier ART) varia, quindi
     le colonne costanti non stanno agli stessi indici per tutti.
  4. Controllo automatico dei pesi di contrasto (vedi check_contrast_weights).
  5. Fix del binning: i voxel con metrica == 1.0 non vengono piu' scartati.
  6. Diagnostica: conteggio voxel per bin, e profilo opzionale della ResMS.

I punti marcati [PLACEHOLDER] vanno verificati o compilati: sono valori che
non ho ricavato dai tuoi dati.
"""

import os
import re
import glob
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ants
from scipy.io import loadmat

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

ROOT_DIR = '/mnt/store1/projects/progetto_layer/data/derivatives'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'group_layer_plotting_psc')
PSC_DIR = os.path.join(ROOT_DIR, 'psc_maps')

# Cartella first-level per soggetto: contiene i con_00XX e i beta_00XX.
SPM_DIR_PATTERN = os.path.join(ROOT_DIR, 'hires_pipeline', '{sub}', 'stats-masked')

# Soggetto di riferimento: da qui si stampa il riepilogo del disegno (SF per
# condizione e per contrasto, pesi dei contrasti) e si ricava il termine di
# paragone per i controlli. NON e' il disegno usato per il calcolo: SPM.mat
# viene comunque letto per ogni soggetto, perche' il numero di regressori di
# nuisance (outlier ART) varia e le colonne costanti cadono a indici diversi.
REFERENCE_SUBJECT = 'sub-001'

SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004",
            "sub-005", "sub-006", "sub-007", "sub-008"]

ROIS = ["dorsal", "ventral", "visual"]

# Chiavi invariate: le stringhe spmT_XXXX restano perche' compaiono nei path
# di ROI e metrica. Il file con_ corrispondente viene derivato per sostituzione.
TASK = {'baseline': 'spmT_0001', 'mirroring': 'spmT_0008', 'spacing': 'spmT_0009'}
MAPPING = {
    'spmT_0001': 'baseline',
    'spmT_0002': 'mirroring',
    'spmT_0003': 'mirroring',
    'spmT_0004': 'mirroring',
    'spmT_0005': 'spacing',
    'spmT_0006': 'spacing',
    'spmT_0007': 'spacing',
    'spmT_0008': 'mirroring',
    'spmT_0009': 'spacing',
}

# Il contrasto 0001 e' baseline > 0 (PSC assoluto). Tutti gli altri sono
# differenziali (lettura manipolata > lettura normale): sono Delta PSC.
ABSOLUTE_CONTRASTS = {'spmT_0001'}

UPSAMP_SPACING = (0.2, 0.2, 0.2)
NR_LAYERS = 10

# Soglia minima sul beta costante per considerare un voxel valido.
# Con grand mean scaling a 100 il fondo e' vicino a zero; 1.0 e' una scelta mia,
# non un valore derivato dai tuoi dati. Controlla l'istogramma di beta_costante.
MIN_SESSION_MEAN = 1.0

# Come normalizzare il PSC.
#   'voxelwise'  -> divide per il beta costante di OGNI voxel. Rimuove le
#                   disomogeneita' locali di ricezione, ma inietta nei profili
#                   il profilo di profondita' del segnale di base (vedi
#                   diagnostic_psc_denominator.png).
#   'roi_scalar' -> divide per la media del beta costante su tutta la GM della
#                   ROI: un solo numero. Rimuove le differenze di guadagno tra
#                   soggetti e tra ROI senza toccare la FORMA del profilo.
# Per un'analisi laminare, dove la forma e' la variabile dipendente,
# 'roi_scalar' e' piu' conservativo. Le cache sono separate, quindi si possono
# girare entrambe le modalita' senza ricalcolare tutto.
DENOMINATOR_MODE = 'voxelwise'

PLOT_VOXEL_COUNTS = True     # pannello diagnostico con i voxel per bin
PLOT_BASELINE = True         # profilo laminare del denominatore del PSC

COLOR_ROI = {
    'visual': '#0f9d58',
    'ventral': '#ffab40',
    'dorsal': '#4285f4',
}


# ----------------------------------------------------------------------------
# LETTURA SPM.mat
# ----------------------------------------------------------------------------

def read_spm_info(spm_dir):
    """Estrae da SPM.mat: dt, HRF, indici dei beta costanti, durate condizioni."""
    mat_path = os.path.join(spm_dir, 'SPM.mat')
    m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    SPM = m['SPM']

    dt = float(SPM.xBF.dt)
    bf = np.asarray(SPM.xBF.bf, dtype=float)
    if bf.ndim > 1:
        bf = bf[:, 0]            # colonna 1 = HRF canonica

    # iB = indici (1-based) delle colonne costanti di sessione
    iB = np.atleast_1d(np.asarray(SPM.xX.iB, dtype=int)).ravel()

    # Durate delle condizioni, prima sessione
    sess = np.atleast_1d(SPM.Sess)[0]
    U = np.atleast_1d(sess.U)
    conditions = {}
    for u in U:
        name = np.atleast_1d(u.name)[0]
        dur = np.atleast_1d(np.asarray(u.dur, dtype=float))
        conditions[str(name)] = dur

    # Contrasti
    xCon = np.atleast_1d(SPM.xCon)
    contrasts = {}
    for i, c in enumerate(xCon, start=1):
        contrasts[i] = {
            'name': str(np.atleast_1d(c.name)[0]),
            'c': np.asarray(c.c, dtype=float).ravel(),
        }

    # Nomi delle colonne, es. 'Sn(1) baseline*bf(1)': servono per sapere
    # quale condizione sta in quale colonna del contrasto.
    col_names = [str(x) for x in np.atleast_1d(SPM.xX.name)]

    return {'dt': dt, 'bf': bf, 'iB': iB, 'col_names': col_names,
            'conditions': conditions, 'contrasts': contrasts}


def compute_sf(dt, bf, duration):
    """Altezza di picco del regressore convoluto per un blocco isolato."""
    n = max(1, int(round(duration / dt)))
    x = np.convolve(np.ones(n), bf)
    return float(x.max())


def condition_sfs(info, verbose=True):
    """SF di ciascuna condizione del disegno (diagnostica, nessun controllo)."""
    dt, bf = info['dt'], info['bf']
    per_cond = {}
    for name, durs in info['conditions'].items():
        d = np.asarray(durs, dtype=float)
        sf = compute_sf(dt, bf, float(d.mean()))
        per_cond[name] = sf
        if verbose:
            n_bins = max(1, int(round(float(d.mean()) / dt)))
            print(f"    '{name}': durata {d.mean():.3f} s "
                  f"(range {d.min():.3f}-{d.max():.3f}), "
                  f"{n_bins} bin microtime, SF = {sf:.4f}")
    return per_cond


def parse_column_conditions(info):
    """Mappa indice di colonna (1-based) -> (nome condizione, indice basis fn).

    Le colonne che non sono regressori di condizione (movimento, costanti)
    non compaiono nella mappa.
    """
    out = {}
    for j, nm in enumerate(info['col_names'], start=1):
        m = re.match(r'^Sn\(\d+\)\s+(.*?)\*bf\((\d+)\)$', nm.strip())
        if m:
            out[j] = (m.group(1), int(m.group(2)))
    return out


def contrast_sf(info, con_number, cond_sf, rel_tol=0.02):
    """SF valido per un singolo contrasto.

    Guarda solo le condizioni con peso diverso da zero: i regressori di non
    interesse (qui END e fixPic) hanno SF molto diverso ma non entrano nel
    contrasto, quindi non devono influenzare il calcolo.

    Si ferma se le condizioni *effettivamente pesate* hanno SF diversi: in quel
    caso un unico scalare applicato al con_ non e' valido e il PSC va calcolato
    dai singoli beta.
    """
    c = info['contrasts'][con_number]['c']
    colmap = parse_column_conditions(info)
    used_cols = np.nonzero(np.abs(c) > 1e-12)[0] + 1   # 1-based

    sfs, used_names, unknown = {}, [], []
    for j in used_cols:
        j = int(j)
        if j not in colmap:
            unknown.append(j)
            continue
        name, bf_idx = colmap[j]
        if bf_idx != 1:
            raise ValueError(
                f"con_{con_number:04d} pesa la basis function {bf_idx} "
                f"(colonna {j}, '{name}'). Il PSC come lo stiamo calcolando "
                "vale solo per la HRF canonica.")
        # Il suffisso dei modulatori parametrici va tolto SOLO se il nome non
        # e' gia' una condizione nota: 'fixPic'.split('x') darebbe 'fi'.
        if name in cond_sf:
            base = name
        else:
            m_pm = re.match(r'^(.*?)x.+\^\d+$', name)
            base = m_pm.group(1) if m_pm else name
        if base not in cond_sf:
            unknown.append(j)
            continue
        sfs[base] = cond_sf[base]
        used_names.append(base)

    if unknown:
        print(f"    [ATTENZIONE] con_{con_number:04d}: colonne pesate non "
              f"riconducibili a una condizione: {unknown}. Verifica che non "
              "siano regressori di nuisance pesati per errore.")

    if not sfs:
        raise ValueError(f"con_{con_number:04d}: nessuna condizione riconosciuta.")

    vals = np.array(list(sfs.values()))
    spread = (vals.max() - vals.min()) / vals.mean()
    if spread > rel_tol:
        raise ValueError(
            f"con_{con_number:04d} pesa condizioni con SF diversi "
            f"({100 * spread:.1f}%): {sfs}. Calcola il PSC dai singoli beta.")

    return float(vals.mean()), sorted(set(used_names))


def check_contrast_weights(info, con_number):
    """Verifica che i pesi positivi del contrasto sommino a 1.

    Perche' conta: se spmT_0008 e' la media dei tre livelli, il vettore di
    contrasto deve essere [1/3 1/3 1/3] e non [1 1 1]. Con [1 1 1] il con_
    e' la somma, e il PSC risulta 3 volte troppo grande. Lo stesso vale per
    i contrasti differenziali: pesi positivi a somma 1, negativi a somma -1.

    Restituisce il fattore per cui dividere per normalizzare (1.0 se ok).
    """
    c = info['contrasts'][con_number]['c']
    pos = c[c > 0].sum()
    neg = c[c < 0].sum()

    if pos == 0:
        raise ValueError(f"Contrasto {con_number}: nessun peso positivo.")

    ok_pos = abs(pos - 1.0) < 1e-6
    ok_neg = (neg == 0) or (abs(neg + 1.0) < 1e-6)

    if not (ok_pos and ok_neg):
        print(f"    [ATTENZIONE] con_{con_number:04d} "
              f"('{info['contrasts'][con_number]['name']}'): "
              f"somma pesi positivi = {pos:.4f}, negativi = {neg:.4f}. "
              f"Normalizzo dividendo per {pos:.4f}.")
    return float(pos)


# ----------------------------------------------------------------------------
# MAPPE PSC
# ----------------------------------------------------------------------------

def _find(spm_dir, stem):
    hits = (glob.glob(os.path.join(spm_dir, stem + '.nii'))
            + glob.glob(os.path.join(spm_dir, stem + '.nii.gz')))
    return hits[0] if hits else None


def load_session_mean(spm_dir, iB):
    """Media dei beta costanti (uno per sessione): e' il denominatore del PSC.

    Restituisce (array, immagine di riferimento per affine/header).
    """
    means, ref_img = [], None
    for idx in iB:
        b_path = _find(spm_dir, f'beta_{int(idx):04d}')
        if b_path is None:
            raise FileNotFoundError(f"beta_{int(idx):04d} non trovato in {spm_dir}")
        img = nib.load(b_path)
        ref_img = img if ref_img is None else ref_img
        means.append(img.get_fdata())

    session_mean = np.mean(np.stack(means, axis=-1), axis=-1)
    session_mean[session_mean < MIN_SESSION_MEAN] = np.nan
    return session_mean, ref_img


def make_psc_map(spm_dir, con_number, SF, norm_factor, iB, out_path):
    """PSC = 100 * SF * con / mean(beta_costanti), normalizzato per i pesi.

    iB deve essere quello del soggetto: il numero di regressori di nuisance
    varia, quindi le colonne costanti non stanno agli stessi indici per tutti.
    """
    con_path = _find(spm_dir, f'con_{con_number:04d}')
    if con_path is None:
        raise FileNotFoundError(f"con_{con_number:04d} non trovato in {spm_dir}")

    con_img = nib.load(con_path)
    con = con_img.get_fdata()

    if DENOMINATOR_MODE == 'roi_scalar':
        # Solo il numeratore: la divisione avviene al campionamento, con uno
        # scalare calcolato sulla GM della ROI.
        psc = 100.0 * SF * (con / norm_factor)
    else:
        session_mean, _ = load_session_mean(spm_dir, iB)
        psc = 100.0 * SF * (con / norm_factor) / session_mean

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(psc.astype(np.float32),
                             con_img.affine, con_img.header), out_path)
    return out_path


def resample_ants(input_path, output_path, spacing, interp_type=0):
    """Resample via ANTs. interp_type=0 = lineare.

    Ora e' legittimo: la mappa PSC non e' sogliata, quindi l'interpolazione
    non mescola valori validi con zeri strutturali.
    """
    print(f"  [Resampling ANTs] -> {os.path.basename(output_path)}")
    img = ants.image_read(input_path)
    out = ants.resample_image(img, spacing, use_voxels=False,
                              interp_type=interp_type)
    ants.image_write(out, output_path)


# ----------------------------------------------------------------------------
# CAMPIONAMENTO LAMINARE
# ----------------------------------------------------------------------------

def sample_map(map_path, metric_path, roi_path, nr_layers=NR_LAYERS):
    """Profilo laminare di una mappa qualsiasi dentro la GM della ROI.

    Restituisce (profilo, conteggio_voxel_per_bin).
    Nessuna soglia statistica: la selezione spaziale la fa la ROI.
    """
    mask = (nib.load(roi_path).get_fdata() == 3)          # 3 = gray matter
    data_x = nib.load(metric_path).get_fdata()[mask]

    img = nib.load(map_path)
    if len(img.shape) > 3:
        data_y = img.get_fdata()[:, :, :, 0][mask]
    else:
        data_y = img.get_fdata()[mask]

    valid = np.isfinite(data_y) & np.isfinite(data_x)
    data_x, data_y = data_x[valid], data_y[valid]

    if data_y.size == 0:
        return None, None

    # Fix: clip cosi' i voxel con metrica == 1.0 finiscono nell'ultimo bin
    bins = np.clip((data_x * nr_layers).astype(int), 0, nr_layers - 1)

    profile = np.full(nr_layers, np.nan)
    counts = np.zeros(nr_layers, dtype=int)
    for i in range(nr_layers):
        sel = (bins == i)
        counts[i] = sel.sum()
        if counts[i] > 0:
            profile[i] = np.mean(data_y[sel])

    return profile, counts


_SCALAR_CACHE = {}


def roi_scalar_denominator(sub, roi_path, reference):
    """Media del beta costante sulla GM della ROI: un solo numero.

    E' il denominatore usato in modalita' 'roi_scalar'. Riusa la mappa del
    session mean gia' upsamplata per la diagnostica, quindi non ricalcola nulla.
    """
    key = (sub, roi_path)
    if key in _SCALAR_CACHE:
        return _SCALAR_CACHE[key]

    b_path = get_subject_baseline_upsampled(sub, reference)
    mask = (nib.load(roi_path).get_fdata() == 3)
    vals = nib.load(b_path).get_fdata()[mask]
    vals = vals[np.isfinite(vals)]
    s = float(np.mean(vals)) if vals.size else np.nan

    _SCALAR_CACHE[key] = s
    return s


def sample_psc_profile(sub, contrast, metric_path, roi_path, reference):
    """Profilo laminare in PSC, secondo DENOMINATOR_MODE.

    Restituisce (profilo, conteggi). In 'voxelwise' la mappa e' gia' PSC; in
    'roi_scalar' la mappa e' il solo numeratore e la divisione avviene qui.
    """
    map_path = get_subject_psc_upsampled(sub, contrast, reference)
    prof, counts = sample_map(map_path, metric_path, roi_path)
    if prof is None:
        return None, None

    if DENOMINATOR_MODE == 'roi_scalar':
        s = roi_scalar_denominator(sub, roi_path, reference)
        if not np.isfinite(s) or s <= 0:
            print(f"    [SKIP] {sub}: denominatore di ROI non valido ({s})")
            return None, None
        prof = prof / s

    return prof, counts


def subject_paths(sub, contrast, region):
    """Path di metrica e ROI (invariati: usano la chiave spmT)."""
    task_key = TASK[MAPPING[contrast]]
    metric = (f'{ROOT_DIR}/layer_fmri/{sub}/'
              f'{sub}_con-{task_key}_label-{region}_metric_equidist.nii')
    roi = (f'{ROOT_DIR}/rois/{sub}/'
           f'{sub}_con-{task_key}_desc-polishCleaned_label-{region}'
           f'_res-up_dseg.nii.gz')
    return metric, roi


_DESIGN_CACHE = {}


def get_subject_design(sub, reference, verbose=True):
    """Disegno del singolo soggetto, con confronto rispetto al riferimento.

    Serve perche' il numero di regressori di nuisance (es. gli outlier di ART)
    varia tra soggetti: le condizioni sono le stesse, ma le colonne costanti
    cadono a indici diversi. I con_ sono gia' corretti per costruzione; quello
    che va preso per soggetto e' iB.
    """
    if sub in _DESIGN_CACHE:
        return _DESIGN_CACHE[sub]

    info = read_spm_info(SPM_DIR_PATTERN.format(sub=sub))
    cond_sf = condition_sfs(info, verbose=False)

    sf, norm = {}, {}
    for contrast in MAPPING.keys():
        n = int(contrast.split('_')[1])
        norm[n] = check_contrast_weights(info, n)
        sf[n], _ = contrast_sf(info, n, cond_sf)

    d = {'sf': sf, 'norm': norm, 'iB': info['iB'],
         'n_cols': len(info['col_names']), 'cond_sf': cond_sf}

    if verbose:
        print(f"  {sub}: {d['n_cols']} colonne, "
              f"costanti {info['iB'].tolist()}")

    # Le condizioni devono essere le stesse: se SF o pesi cambiano, il
    # confronto tra soggetti non e' piu' su quantita' omogenee.
    for n in sf:
        if abs(sf[n] - reference['sf'][n]) > 1e-6:
            print(f"    [ATTENZIONE] {sub} con_{n:04d}: SF {sf[n]:.4f} "
                  f"diverso dal riferimento {reference['sf'][n]:.4f}")
        if abs(norm[n] - reference['norm'][n]) > 1e-6:
            print(f"    [ATTENZIONE] {sub} con_{n:04d}: pesi di contrasto "
                  f"diversi dal riferimento ({norm[n]:.4f} vs "
                  f"{reference['norm'][n]:.4f})")

    _DESIGN_CACHE[sub] = d
    return d


def get_subject_baseline_upsampled(sub, reference):
    """Mappa upsamplata del denominatore del PSC (media dei beta costanti).

    Serve per verificare se il denominatore ha un andamento sistematico lungo
    la profondita' corticale: se ce l'ha, la normalizzazione PSC non e' un
    semplice riscalamento ma modifica la forma del profilo laminare.
    """
    up_path = os.path.join(PSC_DIR, sub, f'{sub}_desc-sessionMean_res-up.nii.gz')
    if os.path.exists(up_path):
        return up_path

    native_path = os.path.join(PSC_DIR, sub, f'{sub}_desc-sessionMean.nii.gz')
    if not os.path.exists(native_path):
        d = get_subject_design(sub, reference, verbose=False)
        spm_dir = SPM_DIR_PATTERN.format(sub=sub)
        session_mean, ref_img = load_session_mean(spm_dir, d['iB'])
        os.makedirs(os.path.dirname(native_path), exist_ok=True)
        nib.save(nib.Nifti1Image(session_mean.astype(np.float32),
                                 ref_img.affine, ref_img.header), native_path)

    resample_ants(native_path, up_path, UPSAMP_SPACING, interp_type=0)
    return up_path


def plot_baseline_profiles(design):
    """Profilo laminare del denominatore del PSC, per set di ROI e per ROI.

    Un pannello per ciascun set di ROI (baseline / mirroring / spacing): le
    maschere sono diverse, quindi selezionano voxel diversi e il profilo del
    segnale di base non e' necessariamente lo stesso.

    Riga 1: forma normalizzata (profilo diviso per la sua media sui bin),
    perche' l'intensita' assoluta varia tra soggetti per guadagno di ricezione.
    Se la curva e' piatta, dividere per il denominatore e' un riscalamento e
    non tocca la forma dei profili PSC. Se non lo e', la tocca, e la modalita'
    'roi_scalar' evita il problema.
    Riga 2: valori grezzi.
    """
    print(f"\n{'=' * 50}\n Diagnostica: denominatore del PSC\n{'=' * 50}")

    families = list(TASK.keys())          # baseline, mirroring, spacing
    fig, axes = plt.subplots(2, len(families),
                             figsize=(5.5 * len(families), 10),
                             squeeze=False, sharex=True)
    x_bins = np.arange(1, NR_LAYERS + 1)
    has_data = False
    summary = {}

    for col, family in enumerate(families):
        ax_shape, ax_raw = axes[0, col], axes[1, col]
        roi_key = TASK[family]
        # un contrasto qualsiasi di questa famiglia: serve solo per i path
        contrast = next(c for c, f in MAPPING.items() if f == family)

        for region in ROIS:
            shapes, raws = [], []
            for sub in SUBJECTS:
                metric_path, roi_path = subject_paths(sub, contrast, region)
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    continue
                try:
                    b_path = get_subject_baseline_upsampled(sub, design)
                except Exception as e:
                    print(f"    [SKIP] {sub}: {e}")
                    continue

                prof, _ = sample_map(b_path, metric_path, roi_path)
                if prof is None:
                    continue
                raws.append(prof)
                m = np.nanmean(prof)
                shapes.append(prof / m if m and np.isfinite(m) else prof * np.nan)

            if not shapes:
                print(f"  [WARNING] nessun dato per {family} / {region}")
                continue

            color = COLOR_ROI.get(region, 'black')
            for arr, ax in ((np.array(shapes), ax_shape),
                            (np.array(raws), ax_raw)):
                n_valid = np.sum(~np.isnan(arr), axis=0)
                mu = np.nanmean(arr, axis=0)
                se = (np.nanstd(arr, axis=0, ddof=1)
                      / np.sqrt(np.maximum(n_valid, 1)))
                ax.plot(x_bins, mu, marker='o', linewidth=2.5, color=color,
                        label=f"{region} (N={arr.shape[0]})")
                ax.fill_between(x_bins, mu - se, mu + se, color=color,
                                alpha=0.20, edgecolor='none')

            summary[f'{family}/{region}'] = np.nanmean(np.array(shapes),
                                                       axis=0).tolist()
            has_data = True

        ax_shape.axhline(1.0, color='0.5', linewidth=1)
        ax_shape.set_title(f"{family}  [{roi_key}]", fontsize=12,
                           fontweight='bold')
        ax_raw.set_xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
        for ax in (ax_shape, ax_raw):
            ax.set_xlim((1, NR_LAYERS))
            ax.set_xticks(x_bins)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc="best", frameon=True, fontsize=9)

    if not has_data:
        plt.close(fig)
        return

    axes[0, 0].set_ylabel("Denominatore normalizzato (media bin = 1)",
                          fontsize=11)
    axes[1, 0].set_ylabel("Beta costante (unita' arbitrarie)", fontsize=11)
    fig.suptitle("Forma del denominatore del PSC lungo la profondita', "
                 "per set di ROI\npiatto = la normalizzazione non altera la "
                 "forma dei profili", fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "diagnostic_psc_denominator.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f" -> salvato: {save_path}")

    with open(os.path.join(OUTPUT_DIR,
                           'psc_denominator_profiles.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def get_subject_psc_upsampled(sub, contrast, reference):
    """Genera (se serve) la mappa PSC upsamplata per soggetto e contrasto.

    reference e' il disegno di riferimento prodotto da load_design(); i
    parametri effettivamente usati sono pero' quelli del soggetto.
    """
    con_number = int(contrast.split('_')[1])
    spm_dir = SPM_DIR_PATTERN.format(sub=sub)

    tag = 'psc' if DENOMINATOR_MODE == 'voxelwise' else 'pscnum'
    up_path = os.path.join(PSC_DIR, sub,
                           f'{sub}_desc-{contrast}_{tag}_res-up.nii.gz')
    if os.path.exists(up_path):
        return up_path

    native_path = os.path.join(PSC_DIR, sub,
                               f'{sub}_desc-{contrast}_{tag}.nii.gz')
    if not os.path.exists(native_path):
        d = get_subject_design(sub, reference, verbose=False)
        make_psc_map(spm_dir, con_number,
                     d['sf'][con_number], d['norm'][con_number],
                     d['iB'], native_path)

    resample_ants(native_path, up_path, UPSAMP_SPACING, interp_type=0)
    return up_path


def load_design():
    """Riepilogo del disegno dal soggetto di riferimento: SF, iB, normalizzazioni.

    Il risultato serve come termine di paragone per get_subject_design(), non
    come sorgente dei parametri di calcolo.
    """
    ref_dir = SPM_DIR_PATTERN.format(sub=REFERENCE_SUBJECT)
    info = read_spm_info(ref_dir)

    print(f"Disegno letto da {REFERENCE_SUBJECT} ({ref_dir})")
    print(f"  dt = {info['dt']:.4f} s")
    print("  SF per condizione:")
    cond_sf = condition_sfs(info)
    print(f"  beta costanti (1-based): {info['iB'].tolist()}")

    print("  contrasti:")
    sf, norm = {}, {}
    for contrast in MAPPING.keys():
        n = int(contrast.split('_')[1])
        c = info['contrasts'][n]
        norm[n] = check_contrast_weights(info, n)
        sf[n], used = contrast_sf(info, n, cond_sf)
        print(f"    con_{n:04d} '{c['name']}': condizioni {used}, "
              f"SF = {sf[n]:.4f}, /{norm[n]:.4f}")

    return {'sf': sf, 'iB': info['iB'], 'norm': norm,
            'cond_sf': cond_sf, 'info': info}


# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    design = load_design()

    # Ogni soggetto ha un numero diverso di regressori di nuisance, quindi le
    # colonne costanti cadono a indici diversi: iB va letto per soggetto.
    print("\nDisegni per soggetto:")
    for sub in SUBJECTS:
        try:
            get_subject_design(sub, design)
        except Exception as e:
            print(f"  {sub}: [ERRORE lettura SPM.mat] {e}")

    with open(os.path.join(OUTPUT_DIR, 'design_summary.json'), 'w') as f:
        json.dump({'reference_subject': REFERENCE_SUBJECT,
                   'denominator_mode': DENOMINATOR_MODE,
                   'condition_sf': design['cond_sf'],
                   'contrast_sf': {str(k): v for k, v in design['sf'].items()},
                   'norm_factors': {str(k): v
                                    for k, v in design['norm'].items()},
                   'per_subject': {s: {'n_cols': d['n_cols'],
                                       'iB': d['iB'].tolist()}
                                   for s, d in _DESIGN_CACHE.items()}},
                  f, indent=2)

    if PLOT_BASELINE:
        plot_baseline_profiles(design)

    for contrast in MAPPING.keys():
        print(f"\n{'=' * 50}\n Contrasto: {contrast} ({MAPPING[contrast]})\n{'=' * 50}")

        n_panels = 2 if PLOT_VOXEL_COUNTS else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(9, 6 * n_panels),
                                 squeeze=False)
        ax = axes[0, 0]
        ax_n = axes[1, 0] if PLOT_VOXEL_COUNTS else None

        x_bins = np.arange(1, NR_LAYERS + 1)
        has_data = False

        for region in ROIS:
            print(f" -> ROI: {region}")
            profiles, counts_all = [], []

            for sub in SUBJECTS:
                metric_path, roi_path = subject_paths(sub, contrast, region)
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    print(f"    [SKIP] {sub}: metrica o ROI mancante")
                    continue
                try:
                    prof, counts = sample_psc_profile(
                        sub, contrast, metric_path, roi_path, design)
                except Exception as e:
                    print(f"    [SKIP] {sub}: {e}")
                    continue
                if prof is None:
                    print(f"    [SKIP] {sub}: nessun voxel valido")
                    continue
                profiles.append(prof)
                counts_all.append(counts)

            if not profiles:
                print(f"  [WARNING] nessun soggetto valido per {region}")
                continue

            M = np.array(profiles)
            n_valid = np.sum(~np.isnan(M), axis=0)
            mean_profile = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1)
            se = sd / np.sqrt(np.maximum(n_valid, 1))

            color = COLOR_ROI.get(region, 'black')
            ax.plot(x_bins, mean_profile, marker='o', linewidth=2.5,
                    color=color, label=f"{region} (N={M.shape[0]})")
            ax.fill_between(x_bins, mean_profile - se, mean_profile + se,
                            color=color, alpha=0.20, edgecolor='none')

            if ax_n is not None:
                ax_n.plot(x_bins, np.mean(np.array(counts_all), axis=0),
                          marker='s', linewidth=1.8, color=color, label=region)

            has_data = True

        if not has_data:
            plt.close(fig)
            print(f" [WARNING] nessun dato per {contrast}")
            continue

        is_abs = contrast in ABSOLUTE_CONTRASTS
        ylab = ("Percent signal change (%)" if is_abs
                else "$\\Delta$ percent signal change (%)")
        subtitle = ("lettura parole normali > riposo implicito" if is_abs
                    else "lettura manipolata > lettura parole normali")

        ax.axhline(0, color='0.5', linewidth=1, linestyle='-')
        ax.set_xlim((1, NR_LAYERS))
        ax.set_xticks(x_bins)
        ax.set_xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"Group mean layer profile ($\\pm$ SE)\n"
                     f"{contrast} ({MAPPING[contrast]}) - {subtitle}\n"
                     f"denominatore: {DENOMINATOR_MODE}",
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc="best", frameon=True)

        if ax_n is not None:
            ax_n.set_xlim((1, NR_LAYERS))
            ax_n.set_xticks(x_bins)
            ax_n.set_xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
            ax_n.set_ylabel("Voxel per bin (media soggetti)", fontsize=11)
            ax_n.set_title("Diagnostica: numero di voxel campionati per bin",
                           fontsize=11)
            ax_n.grid(True, linestyle='--', alpha=0.4)
            ax_n.legend(loc="best", frameon=True)

        fig.tight_layout()
        save_path = os.path.join(
            OUTPUT_DIR,
            f"group_psc_{contrast}_{MAPPING[contrast]}_{DENOMINATOR_MODE}.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f" -> salvato: {save_path}")


if __name__ == '__main__':
    main()