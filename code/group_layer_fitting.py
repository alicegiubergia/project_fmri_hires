"""
Profili laminari per contrasto + modulazione parametrica + distanza
inter-regionale, in PERCENT SIGNAL CHANGE.

Differenza rispetto alla versione precedente: i profili non sono piu' estratti
dai con_ grezzi (unita' arbitrarie) ma dalle mappe PSC

    PSC = 100 * SF * (con / pesi) / mean(beta costanti)

generate e messe in cache da group_layer_plotting_psc.py, che va tenuto nella
stessa cartella. SF, indici dei beta costanti e normalizzazione dei pesi sono
letti da SPM.mat per ogni soggetto.

Struttura dei contrasti:
  con_0001   -> lettura di parole NORMALI > riposo implicito
  con_0002..0004 -> parole con mirroring > parole normali (3 livelli)
  con_0005..0007 -> parole con spacing   > parole normali (3 livelli)

Per ogni soggetto x ROI x condizione si ottengono:
  'read_normal' -> profilo di con_0001 (lettura normale), in PSC
  'level1..3'   -> profilo di ciascuno dei tre contrasti di livello, in Delta PSC
  'slope'       -> profilo della pendenza parametrica sui tre livelli
                   (non e' un contrasto SPM, e' derivato) -- calcolato SOLO
                   per le famiglie ordinali, vedi PARAMETRIC_FAMILIES
"""

import os
import itertools
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import group_layer_plotting_psc as glp

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = '/mnt/store1/projects/progetto_layer/data/derivatives'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'group_layer_parametric_psc')

SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006",
            "sub-007", "sub-008"]
ROIS = ["dorsal", "ventral", "visual"]

BASELINE = 'con_0001'                  # lettura parole normali > riposo
CONDITIONS = {
    'mirroring': ['con_0002', 'con_0003', 'con_0004'],
    'spacing':   ['con_0005', 'con_0006', 'con_0007'],
}

ROI_KEY_FOR = {
    'mirroring': 'spmT_0008',
    'spacing':   'spmT_0009',
}

LEVEL_VALUES = [1, 2, 3]

# Il profilo 'slope' assume che i tre livelli siano una progressione ORDINATA
# della stessa manipolazione. Per lo spacing e' vero (2 < 4 < 6). Per il
# mirroring flip_LR e flip_UD sono qualitativamente diversi, quindi la
# "pendenza" va letta per quello che e' (vedi parametric_slopes): con tre
# livelli e x centrata, coincide con (livello3 - livello1)/2, cioe' un
# contrasto fra il primo e il terzo dei con_ elencati in CONDITIONS, che
# ignora il livello centrale. Qui la teniamo per entrambe le famiglie.
PARAMETRIC_FAMILIES = ['mirroring', 'spacing']

UPSAMP_SPACING = (0.2, 0.2, 0.2)       # deve coincidere con glp.UPSAMP_SPACING
NR_LAYERS = 10                          # per i grafici
COARSE_EDGES = [0, 3, 7, 10]            # deep / middle / superficial
COARSE_NAMES = ['deep', 'middle', 'superficial']

PROFILE_TYPES = ['read_normal', 'level1', 'level2', 'level3', 'slope']

LEVEL_ORDER = ['level1', 'level2', 'level3']

SLOPE_KEY = 'slope'

COLOR_ROI = {'visual': '#0f9d58', 'ventral': '#ffab40', 'dorsal': '#4285f4'}
COLOR_PAIR = {('dorsal', 'ventral'): '#d1495b',
              ('dorsal', 'visual'): '#30638e',
              ('ventral', 'visual'): '#edae49'}


# ============================================================
# UTILITY
# ============================================================

def mean_se(arr, axis=0):
    """Media e SE ignorando i NaN, con n = numero di valori effettivamente
    finiti lungo l'asse (non il numero nominale di soggetti)."""
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis).astype(float)
    sd = np.nanstd(arr, axis=axis, ddof=1)
    se = np.divide(sd, np.sqrt(n), out=np.full_like(sd, np.nan, dtype=float),
                   where=n > 1)
    return m, se


# ============================================================
# ESTRAZIONE
# ============================================================

def load_roi(sub, region, roi_key):
    """Maschera GM e metrica di profondita', definite una volta sola e
    identiche per tutti i contrasti."""
    metric_path = (f'{ROOT_DIR}/layer_fmri/{sub}/'
                   f'{sub}_con-{roi_key}_label-{region}_metric_equivol.nii')
    dseg_path = (f'{ROOT_DIR}/rois/{sub}/'
                 f'{sub}_con-{roi_key}_desc-polishCleaned_'
                 f'label-{region}_res-up_dseg.nii.gz')
    if not (os.path.exists(metric_path) and os.path.exists(dseg_path)):
        print(f"    [WARNING] ROI o metrica mancante: {sub} | {region}")
        return None

    mask = nib.load(dseg_path).get_fdata() == 3
    depth = nib.load(metric_path).get_fdata()[mask]
    # clip: metric == 1.0 finirebbe in un bin fuori range
    layer_id = np.clip((depth * NR_LAYERS).astype(int), 0, NR_LAYERS - 1)
    return mask, layer_id


def load_psc_upsampled(sub, con, design):
    """Mappa PSC upsamplata per un contrasto.

    Delega a group_layer_plotting_psc: la mappa viene calcolata una volta e
    riusata dalla cache in psc_maps/, quindi i due script condividono
    esattamente gli stessi valori.
    """
    contrast = con.replace('con_', 'spmT_')       # chiave usata da glp.MAPPING
    try:
        path = glp.get_subject_psc_upsampled(sub, contrast, design)
    except Exception as e:
        print(f"    [WARNING] PSC non disponibile: {sub} | {con} | {e}")
        return None
    return nib.load(path).get_fdata()


def profile_from_map(data, mask, layer_id):
    """Media per bin di layer. Nessuna soglia, nessun filtro sugli zeri."""
    # Resamplare a uno spacing non garantisce la stessa griglia della dseg
    # se il FOV di partenza differisce: senza questo controllo il masking
    # sarebbe silenziosamente sbagliato.
    if data.shape != mask.shape:
        raise ValueError(f"griglie non allineate: PSC {data.shape} vs "
                         f"maschera {mask.shape}")
    vals = data[mask]
    ok = np.isfinite(vals)
    prof = np.full(NR_LAYERS, np.nan)
    for i in range(NR_LAYERS):
        sel = (layer_id == i) & ok
        if sel.any():
            prof[i] = np.mean(vals[sel])
    return prof


def condition_profiles(sub, region, cond, roi_key, design):
    """Lettura normale e i tre livelli, estratti DENTRO la stessa ROI.

    In modalita' 'roi_scalar' le mappe contengono il solo numeratore del PSC:
    la divisione per il segnale di base della ROI avviene qui, dopo aver
    costruito i profili, con un unico scalare per (soggetto, ROI).
    """
    roi = load_roi(sub, region, roi_key)
    if roi is None:
        return None
    mask, layer_id = roi

    scale = 1.0
    if glp.DENOMINATOR_MODE == 'roi_scalar':
        dseg_path = (f'{ROOT_DIR}/rois/{sub}/'
                     f'{sub}_con-{roi_key}_desc-polishCleaned_'
                     f'label-{region}_res-up_dseg.nii.gz')
        scale = glp.roi_scalar_denominator(sub, dseg_path, design)
        if not np.isfinite(scale) or scale <= 0:
            print(f"    [WARNING] denominatore di ROI non valido: "
                  f"{sub} | {region} ({scale})")
            return None

    base_map = load_psc_upsampled(sub, BASELINE, design)
    if base_map is None:
        return None
    base_prof = profile_from_map(base_map, mask, layer_id) / scale

    rows = []
    for con in CONDITIONS[cond]:
        m = load_psc_upsampled(sub, con, design)
        if m is None:
            return None
        rows.append(profile_from_map(m, mask, layer_id) / scale)
    return base_prof, np.array(rows)


# ============================================================
# STATISTICHE
# ============================================================

def coarsen(prof10):
    """Da 10 bin a 3 (deep / middle / superficial)."""
    return np.array([np.nanmean(prof10[..., a:b], axis=-1)
                     for a, b in zip(COARSE_EDGES[:-1], COARSE_EDGES[1:])]).T


def parametric_slopes(levels_by_layer, level_values):
    """
    Pendenza lineare dell'attivazione sui livelli, per ogni layer.
    Input (n_levels, n_layers) -> output (n_layers,), in % per livello.
    """
    x = np.asarray(level_values, dtype=float)
    x = x - x.mean()
    return (x @ levels_by_layer) / np.sum(x ** 2)


def profile_set(base_prof, mat, cond):
    """
    Un vettore di profilo per ogni contrasto -- niente medie fra livelli --
    piu' il profilo derivato della pendenza parametrica, solo dove la
    progressione dei livelli e' ordinata.
    """
    d = {'read_normal': np.asarray(base_prof, dtype=float)}
    for i in range(mat.shape[0]):
        d[f'level{i + 1}'] = np.asarray(mat[i], dtype=float)

    if cond in PARAMETRIC_FAMILIES:
        d['slope'] = parametric_slopes(mat, LEVEL_VALUES)
    else:
        d['slope'] = np.full(mat.shape[1], np.nan)
    return d


def laminar_shape(prof):
    """
    Come un profilo cambia lungo la profondita': fit b0 + b1*d + b2*d^2.

    Su un profilo di livello (che e' una DIFFERENZA rispetto alla lettura
    normale) questi termini descrivono la forma del reclutamento AGGIUNTIVO,
    non dell'attivazione totale.

    b1 positivo = piu' segnale in superficie. E' il termine piu' esposto al
    drenaggio venoso residuo -- pero' su un contrasto differenziale il
    guadagno vascolare comune si cancella al primo ordine, quindi qui e'
    meno pericoloso che su read_normal.
    b2 positivo = profilo a U, layer medi risparmiati rispetto a
    superficiali e profondi: firma attesa del feedback. Negativo =
    concentrazione nei layer medi, firma feedforward. Il leakage riempie
    l'avvallamento centrale, quindi lavora CONTRO un b2 positivo: trovarlo
    e' conservativo.
    """
    d = np.arange(len(prof), dtype=float)
    d = (d - d.mean()) / d.std()
    ok = np.isfinite(prof)
    if ok.sum() < 4:
        return np.nan, np.nan
    X = np.column_stack([np.ones(ok.sum()), d[ok], d[ok] ** 2])
    beta, *_ = np.linalg.lstsq(X, np.asarray(prof)[ok], rcond=None)
    return float(beta[1]), float(beta[2])


def ff_fb_scores(prof10):
    """
    Punteggi dello schema originale, sul profilo coarse.
      FF = middle - (superficial + deep)/2   -> picco nei layer medi
      FB = deep - superficial                -> sbilanciamento verso il basso
    In PSC questi punteggi sono in punti percentuali e diventano
    confrontabili tra soggetti, cosa che in unita' arbitrarie non erano.
    """
    c = coarsen(np.asarray(prof10, dtype=float))
    deep, mid, sup = c[0], c[1], c[2]
    return deep, mid, sup, mid - (sup + deep) / 2.0, deep - sup


def cosine_distance(u, v, demean):
    """
    1 - cos(u, v) sui bin finiti in ENTRAMBI i profili.

    demean=False -> vettori grezzi. Include la componente costante: se
        entrambe le ROI hanno segnale mediamente positivo la distanza resta
        piccola qualunque sia la forma.
    demean=True  -> vettori centrati; equivale a 1 - r di Pearson, range
        [0, 2]. E' la misura di FORMA.

    Nota sul passaggio al PSC: la versione demeaned e' invariante a un
    fattore di scala costante per ROI, quindi cambia SOLO nella misura in cui
    il denominatore del PSC ha un andamento lungo la profondita' corticale.
    Se la diagnostica di glp (diagnostic_psc_denominator.png) mostra un
    denominatore piatto, queste figure saranno quasi identiche a prima.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 3:
        return np.nan
    u, v = u[ok], v[ok]
    if demean:
        u = u - u.mean()
        v = v - v.mean()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(1.0 - np.dot(u, v) / (nu * nv))


# ============================================================
# PIPELINE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
bins = np.arange(1, NR_LAYERS + 1)
PAIRS = list(itertools.combinations(ROIS, 2))

if tuple(UPSAMP_SPACING) != tuple(glp.UPSAMP_SPACING):
    raise ValueError(f"UPSAMP_SPACING {UPSAMP_SPACING} diverso da quello di "
                     f"glp {glp.UPSAMP_SPACING}: le mappe PSC in cache non "
                     "sarebbero sulla griglia attesa.")

# SF, beta costanti e pesi dei contrasti (letti da SPM.mat, per soggetto)
design = glp.load_design()

# Quale condizione sperimentale sta dietro a ciascun 'livello': le etichette
# level1/2/3 nelle figure seguono l'ordine di CONDITIONS, non un ordine
# sperimentale, quindi conviene averlo scritto.
_colmap = glp.parse_column_conditions(design['info'])
print("\nCorrispondenza livelli -> condizioni:")
for cond, cons in CONDITIONS.items():
    labels = []
    for k, con in enumerate(cons, start=1):
        n = int(con.split('_')[1])
        c = design['info']['contrasts'][n]['c']
        pos = [_colmap[int(j)][0] for j in np.nonzero(c > 1e-12)[0] + 1
               if int(j) in _colmap]
        labels.append(f"level{k}={con}={'+'.join(pos) if pos else '?'}")
    print(f"  {cond:10s}: " + "  ".join(labels))

# aggregato per (condizione, ROI), per le figure di gruppo
results = {(c, r): {'levels': [], 'slopes': [], 'read_normal': [], 'subs': []}
           for c in CONDITIONS for r in ROIS}
# by_sub[sub][cond][region] = {'read_normal':(10,), 'level1':(10,), ..., 'slope':(10,)}
by_sub = {}

for sub in SUBJECTS:
    print(f"\n=== {sub} ===")
    by_sub[sub] = {c: {} for c in CONDITIONS}
    for region in ROIS:
        for cond in CONDITIONS:
            roi_key = ROI_KEY_FOR[cond]
            got = condition_profiles(sub, region, cond, roi_key, design)
            if got is None:
                print(f"  [SKIP] {sub} | {region} | {cond}")
                continue
            base_prof, mat = got

            pset = profile_set(base_prof, mat, cond)
            by_sub[sub][cond][region] = pset

            r = results[(cond, region)]
            r['levels'].append(mat)
            r['slopes'].append(pset['slope'])
            r['read_normal'].append(base_prof)
            r['subs'].append(sub)

            lin, quad = laminar_shape(pset['slope'])
            if np.isfinite(lin):
                print(f"  {region:8s} {cond:10s} [{roi_key}] "
                      f"slope: lin={lin:+.4f} quad={quad:+.4f}")
            else:
                lin1, quad1 = laminar_shape(pset['level1'])
                print(f"  {region:8s} {cond:10s} [{roi_key}] "
                      f"(non parametrico) level1: lin={lin1:+.4f} "
                      f"quad={quad1:+.4f}")

# ------------------------------------------------------------------
# tabella per-contrasto: profilo coarse, FF/FB, forma laminare
# ------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, f'profile_scores_psc_{glp.DENOMINATOR_MODE}.csv'), 'w') as f:
    f.write("subject,condition,roi,roi_key,profile_type,"
            "deep,middle,superficial,FF,FB,linear,quadratic\n")
    for sub in SUBJECTS:
        for cond in CONDITIONS:
            for region in ROIS:
                pset = by_sub.get(sub, {}).get(cond, {}).get(region)
                if pset is None:
                    continue
                for ptype in PROFILE_TYPES:
                    p = pset[ptype]
                    deep, mid, sup, ff, fb = ff_fb_scores(p)
                    lin, quad = laminar_shape(p)
                    f.write(f"{sub},{cond},{region},{ROI_KEY_FOR[cond]},{ptype},"
                            f"{deep:.6f},{mid:.6f},{sup:.6f},"
                            f"{ff:.6f},{fb:.6f},{lin:.6f},{quad:.6f}\n")

# ------------------------------------------------------------------
# distanze inter-regionali, per contrasto
# ------------------------------------------------------------------
# agg[(cond, ptype, res, pair)] = lista di distanze demeaned (una per sub)
agg = {}
dist_rows = []

for sub in SUBJECTS:
    for cond in CONDITIONS:
        for a, b in PAIRS:
            pa = by_sub.get(sub, {}).get(cond, {}).get(a)
            pb = by_sub.get(sub, {}).get(cond, {}).get(b)
            if pa is None or pb is None:
                continue
            for ptype in PROFILE_TYPES:
                ua, ub = pa[ptype], pb[ptype]
                for res, va, vb in (('10bin', ua, ub),
                                    ('3bin', coarsen(ua), coarsen(ub))):
                    d_raw = cosine_distance(va, vb, demean=False)
                    d_dem = cosine_distance(va, vb, demean=True)
                    dist_rows.append((sub, cond, ptype, res, a, b, d_raw, d_dem))
                    agg.setdefault((cond, ptype, res, (a, b)), []).append(d_dem)

with open(os.path.join(OUTPUT_DIR, f'interregional_distance_psc_{glp.DENOMINATOR_MODE}.csv'), 'w') as f:
    f.write("subject,condition,profile_type,resolution,roi_a,roi_b,"
            "cos_dist_raw,cos_dist_demeaned\n")
    for row in dist_rows:
        f.write("{},{},{},{},{},{},{:.6f},{:.6f}\n".format(*row))
    # Distanze sui profili MEDI di gruppo: corrispondono alle figure ma
    # non sono osservazioni indipendenti e non hanno SE. Solo descrittive.
    for cond in CONDITIONS:
        for ptype in PROFILE_TYPES:
            grp = {}
            for region in ROIS:
                arr = [by_sub[s][cond][region][ptype] for s in SUBJECTS
                       if region in by_sub.get(s, {}).get(cond, {})]
                if arr:
                    grp[region] = np.nanmean(np.array(arr), axis=0)
            for a, b in PAIRS:
                if a not in grp or b not in grp:
                    continue
                for res, va, vb in (('10bin', grp[a], grp[b]),
                                    ('3bin', coarsen(grp[a]), coarsen(grp[b]))):
                    f.write(f"GROUP-MEAN,{cond},{ptype},{res},{a},{b},"
                            f"{cosine_distance(va, vb, False):.6f},"
                            f"{cosine_distance(va, vb, True):.6f}\n")

print("\n--- distanza coseno inter-regionale, 10 bin, demeaned ---")
for cond in CONDITIONS:
    for a, b in PAIRS:
        line = f"  {cond:10s} {a:8s} vs {b:8s} |"
        for ptype in PROFILE_TYPES:
            vals = np.array(agg.get((cond, ptype, '10bin', (a, b)), []),
                            dtype=float)
            if vals.size == 0 or not np.any(np.isfinite(vals)):
                line += f" {ptype}=n/a"
                continue
            m, se = mean_se(vals)
            line += f" {ptype}={m:.3f}+/-{se:.3f}"
        print(line)

# --- figura 1: distanza sui tre LIVELLI, con la pendenza staccata ---
# La linea continua congiunge solo level1-2-3. Per lo spacing questa e' una
# progressione ordinata di degradazione: se la distanza cresce lungo l'asse,
# la divergenza fra ROI e' guidata dalla manipolazione. Per il mirroring i
# tre punti sono condizioni distinte e l'asse NON e' ordinato: la linea li
# congiunge solo per leggibilita', la sua pendenza non significa nulla.
# La pendenza parametrica compare solo per le famiglie ordinali.

for res in ('10bin', '3bin'):
    for cond in CONDITIONS:
        show_slope = cond in PARAMETRIC_FAMILIES
        fig, ax = plt.subplots(figsize=(8.5, 5))
        xs = np.arange(len(LEVEL_ORDER))
        x_slope = len(LEVEL_ORDER) + 0.6          # dopo il separatore

        PAIR_DODGE = 0.34
        JITTER = 0.045
        for pi, pair in enumerate(PAIRS):
            c = COLOR_PAIR.get(pair, 'black')

            # tre livelli: linea continua
            mat = np.full((len(LEVEL_ORDER), len(SUBJECTS)), np.nan)
            for i, pt in enumerate(LEVEL_ORDER):
                v = agg.get((cond, pt, res, pair), [])
                mat[i, :len(v)] = v
            m, se = mean_se(mat, axis=1)
            ax.plot(xs, m, marker='o', lw=2.5, color=c,
                    label=f"{pair[0]} vs {pair[1]}")
            ax.fill_between(xs, m - se, m + se, color=c, alpha=0.18,
                            edgecolor='none')
            for j in range(mat.shape[1]):
                ax.plot(xs, mat[:, j], color=c, alpha=0.15, lw=0.8, zorder=0)

            # pendenza: punto isolato, stessa scala ma nessuna linea
            if show_slope:
                v = np.array(agg.get((cond, SLOPE_KEY, res, pair), []),
                             dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    ms, ses = mean_se(v)
                    xp = x_slope + (pi - (len(PAIRS) - 1) / 2) * PAIR_DODGE
                    jit = (np.arange(v.size) - (v.size - 1) / 2) * JITTER
                    ax.scatter(xp + jit, v, s=26, color=c, alpha=0.30, zorder=1)
                    ax.errorbar(xp, ms, yerr=ses, fmt='o', ms=8, lw=2.5,
                                capsize=5, color=c, zorder=2)

        if show_slope:
            # separatore: marca il cambio di tipo di quantita'
            ax.axvline(len(LEVEL_ORDER) - 0.4, color='0.6', lw=1.2, ls='-')
            ax.set_xticks(list(xs) + [x_slope])
            ax.set_xticklabels(LEVEL_ORDER + ['pendenza\n(derivata)'])
            ax.set_xlim(-0.4, x_slope + 0.5)
            ax.set_xlabel("Livello di degradazione")
        else:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(LEVEL_ORDER)
            ax.set_xlim(-0.4, len(LEVEL_ORDER) - 0.6)
            ax.set_xlabel("Condizione (asse non ordinato)")

        ax.axhline(1.0, color='k', lw=0.8, ls=':')   # profili scorrelati
        ax.set_ylabel("1 - r  (distanza coseno, profili centrati)")
        ax.set_title(f"Dissimilarita' laminare inter-regionale: {cond} [{res}]",
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.4)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,
                    f'interregional_distance_{cond}_{res}.png'), dpi=300)
        plt.close(fig)

# --- figura 2: distanza sulla LETTURA NORMALE, a parte ---
# Quanto differiscono i profili laminari delle ROI durante la lettura di
# parole normali. E' una domanda diversa da quella della figura 1 -- non
# riguarda l'effetto della degradazione ma l'organizzazione laminare di
# base della rete della lettura -- e su una quantita' diversa: qui i
# profili sono attivazioni (PSC), non differenze.
for res in ('10bin', '3bin'):
    fig, axes = plt.subplots(1, len(CONDITIONS),
                             figsize=(4.5 * len(CONDITIONS), 4.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, cond in zip(axes, CONDITIONS):
        xs = np.arange(len(PAIRS))
        for i, pair in enumerate(PAIRS):
            vals = np.array(agg.get((cond, 'read_normal', res, pair), []),
                            dtype=float)
            if vals.size == 0:
                continue
            m, se = mean_se(vals)
            c = COLOR_PAIR.get(pair, 'black')
            jit = (rng.random(vals.size) - 0.5) * 0.18
            ax.scatter(i + jit, vals, s=28, color=c, alpha=0.35, zorder=1)
            ax.errorbar(i, m, yerr=se, fmt='o', ms=9, lw=2, capsize=5,
                        color=c, zorder=2)
        ax.axhline(1.0, color='k', lw=0.8, ls=':')   # profili scorrelati
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a}\nvs {b}" for a, b in PAIRS], fontsize=9)
        ax.set_xlim(-0.5, len(PAIRS) - 0.5)
        # La ROI dipende dalla condizione anche per read_normal: stesso
        # contrasto, ma estratto in maschere diverse.
        ax.set_title(f"ROI di {cond} [{ROI_KEY_FOR[cond]}]", fontsize=10)
        ax.grid(True, axis='y', ls='--', alpha=0.4)
    axes[0].set_ylabel("1 - r  (distanza coseno, profili centrati)")
    fig.suptitle(f"Dissimilarita' laminare inter-regionale durante la "
                 f"lettura normale [{res}]", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'interregional_readnormal_{res}.png'), dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------
# figure sui profili
# ------------------------------------------------------------------

# --- profilo della lettura normale ---
for cond in CONDITIONS:
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in ROIS:
        arr = results[(cond, region)]['read_normal']
        if not arr:
            continue
        arr = np.array(arr)
        n = arr.shape[0]
        m, se = mean_se(arr)
        c = COLOR_ROI.get(region, 'black')
        ax.plot(bins, m, marker='o', lw=2.5, color=c, label=f"{region} (N={n})")
        ax.fill_between(bins, m - se, m + se, color=c, alpha=0.20,
                        edgecolor='none')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xticks(bins)
    ax.set_xlabel("Bin di layer (1=WM, 10=CSF)")
    ax.set_ylabel("Percent signal change (%)\nlettura parole normali > riposo")
    ax.set_title(f"Profilo della lettura normale nella ROI di {cond} "
                 f"[{ROI_KEY_FOR[cond]}]", fontsize=12, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'readnormal_profile_{cond}.png'), dpi=300)
    plt.close(fig)

# --- profili per livello (gia' condizione > lettura normale) ---
for cond in CONDITIONS:
    fig, axes = plt.subplots(1, len(ROIS), figsize=(5 * len(ROIS), 4.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, ROIS):
        r = results[(cond, region)]
        if not r['levels']:
            continue
        arr = np.array(r['levels'])                   # (N, 3, 10)
        n = arr.shape[0]
        c = COLOR_ROI.get(region, 'black')
        for li in range(arr.shape[1]):
            m, se = mean_se(arr[:, li])
            ax.plot(bins, m, marker='o', lw=2, color=c,
                    alpha=0.40 + 0.60 * li / 2,
                    label=f"livello {LEVEL_VALUES[li]}")
            ax.fill_between(bins, m - se, m + se, color=c, alpha=0.12,
                            edgecolor='none')
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_title(f"{region} (N={n})")
        ax.set_xlabel("Bin di layer (1=WM, 10=CSF)")
        ax.set_xticks(bins)
        ax.grid(True, ls='--', alpha=0.4)
    axes[0].set_ylabel(f"$\\Delta$ percent signal change (%)\n"
                       f"{cond} > lettura normale")
    axes[-1].legend(loc='best', frameon=True)
    fig.suptitle(f"Profilo laminare per livello di {cond}",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'profiles_{cond}.png'), dpi=300)
    plt.close(fig)

# --- profilo della modulazione parametrica (solo famiglie ordinali) ---
for cond in CONDITIONS:
    if cond not in PARAMETRIC_FAMILIES:
        print(f"[skip] figura parametrica per {cond}: livelli non ordinati")
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in ROIS:
        r = results[(cond, region)]
        if not r['slopes']:
            continue
        arr = np.array(r['slopes'])
        n = arr.shape[0]
        m, se = mean_se(arr)
        c = COLOR_ROI.get(region, 'black')
        ax.plot(bins, m, marker='o', lw=2.5, color=c, label=f"{region} (N={n})")
        ax.fill_between(bins, m - se, m + se, color=c, alpha=0.20,
                        edgecolor='none')
        for s in arr:                   # con N=8 i singoli contano
            ax.plot(bins, s, color=c, alpha=0.20, lw=0.8, zorder=0)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xticks(bins)
    ax.set_xlabel("Bin di layer (1=WM, 10=CSF)")
    ax.set_ylabel("Pendenza parametrica ($\\Delta$ PSC % per livello)")
    ax.set_title(f"Modulazione parametrica: {cond}",
                 fontsize=12, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'parametric_slope_{cond}.png'), dpi=300)
    plt.close(fig)

print(f"\nModalita' denominatore: {glp.DENOMINATOR_MODE}")
print(f"Output in: {OUTPUT_DIR}")