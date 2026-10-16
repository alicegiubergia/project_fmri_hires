"""
Strada A: inferenza feedforward/feedback dai profili laminari in PSC.

Logica dell'analisi
-------------------
La regola anatomica e' che il feedforward termina nei layer intermedi, mentre
il feedback bersaglia i layer superficiali e profondi. La forma assoluta di un
profilo non e' pero' interpretabile direttamente, perche' il GE-BOLD ha un bias
superficiale dovuto al drenaggio venoso.

Per questo il test NON e' sulla forma di una condizione singola, ma sul
**trend tra livelli dentro lo stesso compartimento**: confrontando livelli
diversi alla stessa profondita', il guadagno vascolare depth-dependent si
cancella perche' e' identico nelle due condizioni. L'effetto di interesse e'
quindi l'interazione livello x compartimento.

Riusa le funzioni di group_layer_plotting_psc.py (stessa cartella).

"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import group_layer_plotting_psc as glp


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

OUT_DIR = os.path.join(glp.ROOT_DIR, 'group_layer_ffb')

# [PLACEHOLDER] Definizione dei compartimenti sui 10 bin (1 = WM, 10 = CSF).
# Il bin 1 e il bin 10 sono i piu' esposti al volume parziale con WM e CSF:
# di default li escludo. Verifica che la direzione della metrica LayNii sia
# davvero 1=WM -> 10=CSF nel tuo output.
COMPARTMENTS = {
    'deep':        [2, 3],
    'middle':      [4, 5, 6, 7],
    'superficial': [8, 9],
}

# Modalita' di analisi per famiglia.
#   'ordinal'     -> i livelli sono gradazioni della stessa manipolazione:
#                    si stima una pendenza per soggetto e compartimento.
#   'categorical' -> i livelli sono condizioni qualitativamente diverse:
#                    ANOVA a misure ripetute condizione x compartimento, piu'
#                    i contrasti a coppie.
# Lo spacing e' ordinale (2 < 4 < 6). Il mirroring e' categoriale: flip_LR e
# flip_UD sono manipolazioni qualitativamente diverse, non due gradi della
# stessa cosa.
FAMILY_MODE = {'mirroring': 'categorical', 'spacing': 'ordinal'}

# [PLACEHOLDER] Coppie di condizioni da confrontare nel mirroring. Ogni coppia
# produce un contrasto per soggetto e compartimento; l'effetto depth-specific
# e' la differenza del contrasto tra compartimenti. Cambia questa lista se ti
# interessano confronti diversi.
MIRRORING_PAIRS = [('flip_LR', 'flip_UD'),
                   ('flip_LRUD', 'flip_LR'),
                   ('flip_LRUD', 'flip_UD')]

N_PERM = 10000
RNG_SEED = 0

COLOR_COMP = {'deep': '#4285f4', 'middle': '#0f9d58', 'superficial': '#db4437'}


# ----------------------------------------------------------------------------
# ESTRAZIONE
# ----------------------------------------------------------------------------

def positive_conditions(info, con_number):
    """Condizioni con peso POSITIVO nel contrasto: il livello sperimentale.

    Nei contrasti differenziali 'livello > baseline' la condizione positiva e'
    il livello, quella negativa e' la lettura normale.
    """
    c = info['contrasts'][con_number]['c']
    colmap = glp.parse_column_conditions(info)

    # Una stessa condizione compare una volta per sessione: Sn(1), Sn(2),
    # Sn(3) hanno ciascuna la propria colonna. Vanno contate una volta sola,
    # altrimenti un contrasto su una singola condizione sembra pesarne tre.
    names = []
    for j in np.nonzero(c > 1e-12)[0] + 1:
        if int(j) in colmap:
            nome = colmap[int(j)][0]
            if nome not in names:
                names.append(nome)
    return names


def level_value(cond_name):
    """Valore ordinale del livello (solo per le famiglie 'ordinal')."""
    m = re.match(r'^spacing_(\d+)$', cond_name)
    return float(m.group(1)) if m else np.nan


def build_profile_table(design):
    """Tabella dei profili: (sub, roi, contrast) -> array di NR_LAYERS valori PSC."""
    table = {}
    for contrast in glp.MAPPING.keys():
        for region in glp.ROIS:
            for sub in glp.SUBJECTS:
                metric_path, roi_path = glp.subject_paths(sub, contrast, region)
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    continue
                try:
                    prof, counts = glp.sample_psc_profile(
                        sub, contrast, metric_path, roi_path, design)
                except Exception as e:
                    print(f"  [SKIP] {sub} {contrast} {region}: {e}")
                    continue
                if prof is None:
                    continue
                table[(sub, region, contrast)] = {'profile': prof,
                                                  'counts': counts}
    return table


def compartment_values(profile):
    """Media del profilo dentro ciascun compartimento (bin 1-based)."""
    out = {}
    for name, bins in COMPARTMENTS.items():
        idx = [b - 1 for b in bins]
        vals = profile[idx]
        out[name] = np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan
    return out


# ----------------------------------------------------------------------------
# STATISTICA
# ----------------------------------------------------------------------------

def subject_slopes(table, design, region, family):
    """Pendenza del PSC attraverso i livelli, per soggetto e compartimento.

    family: 'mirroring' o 'spacing'. Usa solo i contrasti dei singoli livelli
    (non le medie 0008/0009).
    """
    info = design['info']

    # Contrasti di livello di questa famiglia, con il loro valore ordinale
    levels = []
    for contrast, fam in glp.MAPPING.items():
        if fam != family or contrast in ('spmT_0001', 'spmT_0008', 'spmT_0009'):
            continue
        n = int(contrast.split('_')[1])
        conds = positive_conditions(info, n)
        if len(conds) != 1:
            print(f"  [ATTENZIONE] {contrast} pesa positivamente piu' di una "
                  f"condizione ({conds}): non e' un contrasto di livello "
                  f"singolo, lo salto nel trend.")
            continue
        levels.append((contrast, conds[0], level_value(conds[0])))

    levels = [l for l in levels if np.isfinite(l[2])]
    if len(levels) < 2:
        return None, levels

    x = np.array([l[2] for l in levels], dtype=float)
    x = x - x.mean()                      # centrata: la pendenza e' interpretabile

    comps = list(COMPARTMENTS.keys())
    slopes = {c: [] for c in comps}
    subs_used = []

    for sub in glp.SUBJECTS:
        y_by_comp = {c: [] for c in comps}
        ok = True
        for contrast, _, _ in levels:
            entry = table.get((sub, region, contrast))
            if entry is None:
                ok = False
                break
            cv = compartment_values(entry['profile'])
            for c in comps:
                y_by_comp[c].append(cv[c])
        if not ok:
            continue
        if not all(np.all(np.isfinite(y_by_comp[c])) for c in comps):
            continue

        for c in comps:
            y = np.array(y_by_comp[c], dtype=float)
            slopes[c].append(float(np.polyfit(x, y, 1)[0]))
        subs_used.append(sub)

    if not subs_used:
        return None, levels

    return ({'slopes': {c: np.array(slopes[c]) for c in comps},
             'subjects': subs_used, 'x': x}, levels)


def sign_flip_test(v, n_perm=N_PERM, seed=RNG_SEED):
    """Test a due code su media != 0 con sign-flipping (N piccolo)."""
    v = np.asarray(v, dtype=float)
    obs = v.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, v.size))
    null = (signs * v).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def family_levels(design, family):
    """Contrasti di livello di una famiglia -> [(contrast, condizione, valore)]."""
    info = design['info']
    levels = []
    for contrast, fam in glp.MAPPING.items():
        if fam != family or contrast in ('spmT_0001', 'spmT_0008', 'spmT_0009'):
            continue
        n = int(contrast.split('_')[1])
        conds = positive_conditions(info, n)
        if len(conds) != 1:
            print(f"  [ATTENZIONE] {contrast} pesa positivamente piu' di una "
                  f"condizione ({conds}): non e' un contrasto di livello "
                  f"singolo, lo salto.")
            continue
        levels.append((contrast, conds[0], level_value(conds[0])))
    return levels


def build_cube(table, design, region, family):
    """Matrice Y[soggetto, condizione, compartimento] + etichette.

    Tiene solo i soggetti con tutte le celle presenti: l'analisi a misure
    ripetute richiede il disegno bilanciato.
    """
    levels = family_levels(design, family)
    comps = list(COMPARTMENTS.keys())
    if len(levels) < 2:
        return None, levels, comps

    rows, subs = [], []
    for sub in glp.SUBJECTS:
        cell = np.full((len(levels), len(comps)), np.nan)
        for i, (contrast, _, _) in enumerate(levels):
            entry = table.get((sub, region, contrast))
            if entry is None:
                break
            cv = compartment_values(entry['profile'])
            for j, c in enumerate(comps):
                cell[i, j] = cv[c]
        if np.all(np.isfinite(cell)):
            rows.append(cell)
            subs.append(sub)

    if not rows:
        return None, levels, comps
    return (np.stack(rows), subs), levels, comps


def rm_anova_interaction(Y, n_perm=N_PERM, seed=RNG_SEED):
    """Interazione condizione x compartimento, misure ripetute su entrambi.

    Y ha forma [soggetti, condizioni, compartimenti].

    Il p-value per permutazione si ottiene rimuovendo prima gli effetti
    principali (doppio centraggio entro soggetto) e poi permutando le etichette
    di condizione dentro ogni soggetto: cosi' la distribuzione nulla riguarda
    l'interazione e non i main effect.
    """
    n, C, K = Y.shape

    def f_interaction(A):
        gm = A.mean()
        m_c = A.mean(axis=(0, 2))                     # medie per condizione
        m_k = A.mean(axis=(0, 1))                     # medie per compartimento
        m_ck = A.mean(axis=0)
        m_s = A.mean(axis=(1, 2))
        m_sc = A.mean(axis=2)
        m_sk = A.mean(axis=1)

        ss_int = n * np.sum((m_ck - m_c[:, None] - m_k[None, :] + gm) ** 2)
        resid = (A
                 - m_sc[:, :, None] - m_sk[:, None, :] - m_ck[None, :, :]
                 + m_s[:, None, None] + m_c[None, :, None] + m_k[None, None, :]
                 - gm)
        ss_err = np.sum(resid ** 2)
        df_int = (C - 1) * (K - 1)
        df_err = (n - 1) * df_int
        if ss_err <= 0 or df_err <= 0:
            return np.nan
        return (ss_int / df_int) / (ss_err / df_err)

    F_obs = f_interaction(Y)
    df_int = (C - 1) * (K - 1)
    df_err = (n - 1) * df_int
    p_param = float(1 - stats.f.cdf(F_obs, df_int, df_err)) if np.isfinite(F_obs) else np.nan

    # Doppio centraggio entro soggetto: restano solo interazione + residuo
    D = (Y - Y.mean(axis=2, keepdims=True) - Y.mean(axis=1, keepdims=True)
         + Y.mean(axis=(1, 2), keepdims=True))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        P = np.stack([D[s][rng.permutation(C)] for s in range(n)])
        F_null = f_interaction(P)
        if np.isfinite(F_null) and F_null >= F_obs:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)

    return {'F': float(F_obs), 'df': [df_int, df_err],
            'p_param': p_param, 'p_perm': float(p_perm)}


def analyse_categorical(table, design, region, family):
    """ANOVA condizione x compartimento + contrasti a coppie depth-specific."""
    cube, levels, comps = build_cube(table, design, region, family)
    if cube is None:
        print(f"  [WARNING] {region} / {family}: dati insufficienti")
        return None
    Y, subs = cube
    names = [l[1] for l in levels]

    print(f"\n  {region} / {family}  (N={len(subs)}, condizioni: {names})")

    out = {'n': len(subs), 'conditions': names, 'compartments': comps,
           'means': {}, 'anova': {}, 'pairwise': {}}

    for j, c in enumerate(comps):
        m = Y[:, :, j].mean(axis=0)
        se = Y[:, :, j].std(axis=0, ddof=1) / np.sqrt(Y.shape[0])
        out['means'][c] = {n_: [float(m[i]), float(se[i])]
                           for i, n_ in enumerate(names)}
        print("    " + f"{c:>12}: " +
              "  ".join(f"{n_}={m[i]:+.3f}%" for i, n_ in enumerate(names)))

    anova = rm_anova_interaction(Y)
    out['anova'] = anova
    print(f"    [ANOVA] condizione x compartimento: F({anova['df'][0]},"
          f"{anova['df'][1]})={anova['F']:.2f}  p={anova['p_param']:.4f}  "
          f"p_perm={anova['p_perm']:.4f}")

    # Contrasti a coppie: differenza tra condizioni, poi differenza tra
    # compartimenti (e' quest'ultima l'effetto depth-specific).
    pairs = MIRRORING_PAIRS if family == 'mirroring' else [
        (names[i], names[j]) for i in range(len(names))
        for j in range(i + 1, len(names))]

    for a, b in pairs:
        if a not in names or b not in names:
            print(f"    [SKIP] coppia ({a}, {b}): condizione assente")
            continue
        ia, ib = names.index(a), names.index(b)
        diff = Y[:, ia, :] - Y[:, ib, :]              # [soggetti, compartimenti]f

        entry = {'per_compartment': {}, 'interactions': {}}
        for j, c in enumerate(comps):
            mean, p_perm = sign_flip_test(diff[:, j])
            t, p_t = stats.ttest_1samp(diff[:, j], 0.0)
            entry['per_compartment'][c] = {'mean': mean, 't': float(t),
                                           'p_t': float(p_t), 'p_perm': p_perm}
        for c1, c2 in [('superficial', 'middle'), ('deep', 'middle'),
                       ('superficial', 'deep')]:
            if c1 not in comps or c2 not in comps:
                continue
            d = diff[:, comps.index(c1)] - diff[:, comps.index(c2)]
            mean, p_perm = sign_flip_test(d)
            t, p_t = stats.ttest_1samp(d, 0.0)
            entry['interactions'][f'{c1}-{c2}'] = {'mean': mean, 't': float(t),
                                                   'p_t': float(p_t),
                                                   'p_perm': p_perm}
            print(f"    [{a} vs {b}] {c1} - {c2}: {mean:+.4f}  "
                  f"t={t:+.2f}  p_t={p_t:.4f}  p_perm={p_perm:.4f}")

        out['pairwise'][f'{a}_vs_{b}'] = entry

    out['denominator_mode'] = glp.DENOMINATOR_MODE
    out['_cube'] = Y.tolist()
    return out


def analyse_ordinal(table, design, region, family):
    """Trend per compartimento + interazione (differenze tra compartimenti)."""
    res, levels = subject_slopes(table, design, region, family)
    if res is None:
        print(f"  [WARNING] {region} / {family}: dati insufficienti")
        return None

    print(f"\n  {region} / {family}  (N={len(res['subjects'])}, "
          f"livelli: {[(l[1], l[2]) for l in levels]})")

    out = {'n': len(res['subjects']), 'levels': [(l[1], l[2]) for l in levels],
           'per_compartment': {}, 'interactions': {}}

    for c, v in res['slopes'].items():
        t, p_t = stats.ttest_1samp(v, 0.0)
        mean, p_perm = sign_flip_test(v)
        d = mean / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan
        out['per_compartment'][c] = {'mean_slope': mean, 't': float(t),
                                     'p_t': float(p_t), 'p_perm': p_perm,
                                     'cohen_d': float(d)}
        print(f"    {c:>12}: pendenza {mean:+.4f} %/livello  "
              f"t={t:+.2f}  p_t={p_t:.4f}  p_perm={p_perm:.4f}  d={d:+.2f}")

    # L'interazione: la pendenza differisce tra compartimenti?
    # E' questo il test depth-specific, immune al guadagno vascolare costante.
    for a, b in [('superficial', 'middle'), ('deep', 'middle'),
                 ('superficial', 'deep')]:
        if a not in res['slopes'] or b not in res['slopes']:
            continue
        diff = res['slopes'][a] - res['slopes'][b]
        t, p_t = stats.ttest_1samp(diff, 0.0)
        mean, p_perm = sign_flip_test(diff)
        out['interactions'][f'{a}-{b}'] = {'mean_diff': mean, 't': float(t),
                                           'p_t': float(p_t), 'p_perm': p_perm}
        print(f"    [interazione] {a} - {b}: {mean:+.4f}  "
              f"t={t:+.2f}  p_t={p_t:.4f}  p_perm={p_perm:.4f}")

    out['_slopes'] = {c: v.tolist() for c, v in res['slopes'].items()}
    return out


def analyse_region(table, design, region, family):
    """Sceglie l'analisi in base alla modalita' della famiglia."""
    mode = FAMILY_MODE.get(family, 'ordinal')
    if mode == 'categorical':
        return analyse_categorical(table, design, region, family)
    return analyse_ordinal(table, design, region, family)


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def plot_slopes(results, family):
    """Pendenze per compartimento (famiglie ordinali), punti = soggetti."""
    regions = [r for r in glp.ROIS if results.get(r) is not None]
    if not regions:
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5),
                             squeeze=False, sharey=True)
    comps = list(COMPARTMENTS.keys())
    rng = np.random.default_rng(RNG_SEED)

    for ax, region in zip(axes[0], regions):
        r = results[region]
        for i, c in enumerate(comps):
            v = np.array(r['_slopes'][c])
            ax.bar(i, v.mean(), color=COLOR_COMP.get(c, 'gray'), alpha=0.30,
                   width=0.6, zorder=1)
            ax.scatter(np.full(v.size, i) + rng.uniform(-.08, .08, v.size), v,
                       color=COLOR_COMP.get(c, 'gray'), alpha=0.6, s=35, zorder=3)
            se = v.std(ddof=1) / np.sqrt(v.size)
            ax.errorbar(i, v.mean(), yerr=se, color='k', capsize=4, zorder=4)

        ax.axhline(0, color='0.4', linewidth=1)
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, fontsize=10)
        ax.set_title(f"{region} (N={r['n']})", fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    axes[0][0].set_ylabel("Pendenza PSC tra livelli (% per livello)", fontsize=11)
    fig.suptitle(f"Effetto del livello di {family} per compartimento\n"
                 f"le differenze tra compartimenti sono l'effetto depth-specific",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'slopes_{family}.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def plot_categorical(results, family):
    """Barre raggruppate: compartimento sull'asse x, una barra per condizione."""
    regions = [r for r in glp.ROIS if results.get(r) is not None]
    if not regions:
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(5.5 * len(regions), 5),
                             squeeze=False, sharey=True)

    for ax, region in zip(axes[0], regions):
        r = results[region]
        comps, conds = r['compartments'], r['conditions']
        width = 0.8 / len(conds)
        cmap = plt.get_cmap('tab10')

        for k, cond in enumerate(conds):
            m = [r['means'][c][cond][0] for c in comps]
            se = [r['means'][c][cond][1] for c in comps]
            x = np.arange(len(comps)) + (k - (len(conds) - 1) / 2) * width
            ax.bar(x, m, width=width, yerr=se, capsize=3, label=cond,
                   color=cmap(k), alpha=0.85)

        ax.axhline(0, color='0.4', linewidth=1)
        ax.set_xticks(np.arange(len(comps)))
        ax.set_xticklabels(comps, fontsize=10)
        a = r['anova']
        ax.set_title(f"{region} (N={r['n']})\n"
                     f"interazione F={a['F']:.2f}, p_perm={a['p_perm']:.3f}",
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=9, frameon=True)

    axes[0][0].set_ylabel("$\\Delta$ percent signal change (%)", fontsize=11)
    fig.suptitle(f"{family}: condizioni per compartimento corticale",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'conditions_{family}.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(RNG_SEED)

    design = glp.load_design()
    print("\nEstrazione profili...")
    table = build_profile_table(design)
    print(f"  {len(table)} profili estratti")

    # Tabella lunga, per rifare le statistiche altrove (R, JASP...)
    csv_path = os.path.join(OUT_DIR, 'laminar_profiles_psc.csv')
    with open(csv_path, 'w') as f:
        f.write(f"# denominator_mode={glp.DENOMINATOR_MODE}\n")
        f.write("subject,roi,contrast,family,bin,psc,n_voxels\n")
        for (sub, region, contrast), e in sorted(table.items()):
            fam = glp.MAPPING[contrast]
            for i in range(glp.NR_LAYERS):
                f.write(f"{sub},{region},{contrast},{fam},{i + 1},"
                        f"{e['profile'][i]:.6f},{e['counts'][i]}\n")
    print(f"  -> {csv_path}")

    all_results = {}
    for family in ['mirroring', 'spacing']:
        mode = FAMILY_MODE.get(family, 'ordinal')
        print(f"\n{'=' * 60}\n Famiglia: {family}  (modalita': {mode})\n{'=' * 60}")
        results = {r: analyse_region(table, design, r, family) for r in glp.ROIS}
        all_results[family] = results
        if mode == 'categorical':
            plot_categorical(results, family)
        else:
            plot_slopes(results, family)

    with open(os.path.join(OUT_DIR, 'ffb_stats.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n-> {os.path.join(OUT_DIR, 'ffb_stats.json')}")


if __name__ == '__main__':
    main()