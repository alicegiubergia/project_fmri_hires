"""
05 - Feedforward/feedback inference from the laminar PSC profiles (route A).

Logic of the analysis
-------------------
The anatomical rule is that feedforward terminates in the middle layers, while
feedback targets the superficial and deep layers. The absolute shape of a
profile, however, cannot be interpreted directly, because GE-BOLD has a
superficial bias caused by venous draining.

For this reason the test is NOT on the shape of a single condition, but on the
**trend across levels within the same compartment**: comparing different
levels at the same depth, the depth-dependent vascular gain cancels out
because it is identical in the two conditions. The effect of interest is
therefore the level x compartment interaction.

Profiles are neither extracted here nor saved to a CSV of its own: they come
from the single CSV of 01_export_profiles.py (created on the fly if missing).

Output: <ROOT_DIR>/laminar_psc/<METHOD>/05_ff_fb_compartments/
Usage:  [LAYER_METHOD=equivol] python 05_ff_fb_compartments.py

"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import laminar_psc as lps


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

OUT_DIR = lps.step_dir('05_ff_fb_compartments')

# Compartments: defined once in laminar_psc.py (BigBrain option
# deep 1-4, middle 5-6, superficial 7-10).
COMPARTMENTS = lps.COMPARTMENTS

# Analysis mode per family.
#   'ordinal'     -> the levels are grades of the same manipulation:
#                    one slope is estimated per subject and compartment.
#   'categorical' -> the levels are qualitatively different conditions:
#                    repeated-measures ANOVA condition x compartment, plus
#                    pairwise contrasts.
# Spacing is ordinal (2 < 4 < 6). Mirroring is categorical: flip_LR and
# flip_UD are qualitatively different manipulations, not two degrees of the
# same thing.
FAMILY_MODE = {'mirroring': 'categorical', 'spacing': 'ordinal'}

# [PLACEHOLDER] Pairs of conditions to compare in mirroring. Each pair
# yields one contrast per subject and compartment; the depth-specific effect
# is the difference of that contrast between compartments. Change this list
# if you are interested in other comparisons.
MIRRORING_PAIRS = [('flip_LR', 'flip_UD'),
                   ('flip_LRUD', 'flip_LR'),
                   ('flip_LRUD', 'flip_UD')]

N_PERM = 10000
RNG_SEED = 0

COLOR_COMP = {'deep': '#4285f4', 'middle': '#0f9d58', 'superficial': '#db4437'}


# ----------------------------------------------------------------------------
# EXTRACTION
# ----------------------------------------------------------------------------

def positive_conditions(info, con_number):
    """Conditions with a POSITIVE weight in the contrast: the experimental level.

    In the differential contrasts 'level > baseline' the positive condition
    is the level, the negative one is normal reading.
    """
    c = info['contrasts'][con_number]['c']
    colmap = lps.parse_column_conditions(info)

    # The same condition appears once per session: Sn(1), Sn(2), Sn(3) each
    # have their own column. They must be counted once, otherwise a contrast
    # on a single condition would seem to weight three.
    names = []
    for j in np.nonzero(c > 1e-12)[0] + 1:
        if int(j) in colmap:
            name = colmap[int(j)][0]
            if name not in names:
                names.append(name)
    return names


def level_value(cond_name):
    """Ordinal value of the level (only for 'ordinal' families)."""
    m = re.match(r'^spacing_(\d+)$', cond_name)
    return float(m.group(1)) if m else np.nan


def build_profile_table(df):
    """Profile table: (sub, roi, contrast) -> {'profile', 'counts'}.

    Built from the single CSV of 01_export_profiles.py. Only the level
    contrasts are needed (the analyses here do not use con_0001, 0008, 0009).
    """
    table = {}
    lv = df[df['condition'] == 'level']
    for (sub, region, contrast), g in lv.groupby(['subject', 'region', 'contrast']):
        prof, counts = lps.profile_from_rows(g)
        table[(sub, region, contrast)] = {'profile': prof, 'counts': counts}
    return table


def compartment_values(profile):
    """Mean of the profile within each compartment (1-based bins)."""
    out = {}
    for name, bins in COMPARTMENTS.items():
        idx = [b - 1 for b in bins]
        vals = profile[idx]
        out[name] = np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan
    return out


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------

def subject_slopes(table, design, region, family):
    """PSC slope across levels, per subject and compartment.

    family: 'mirroring' or 'spacing'. Uses only the single-level contrasts
    (not the 0008/0009 means).
    """
    info = design['info']

    # Level contrasts of this family, with their ordinal value
    levels = []
    for contrast, fam in lps.MAPPING.items():
        if fam != family or contrast in ('spmT_0001', 'spmT_0008', 'spmT_0009'):
            continue
        n = int(contrast.split('_')[1])
        conds = positive_conditions(info, n)
        if len(conds) != 1:
            print(f"  [WARNING] {contrast} positively weights more than one "
                  f"condition ({conds}): it is not a single-level "
                  f"contrast, skipping it in the trend.")
            continue
        levels.append((contrast, conds[0], level_value(conds[0])))

    levels = [l for l in levels if np.isfinite(l[2])]
    if len(levels) < 2:
        return None, levels

    x = np.array([l[2] for l in levels], dtype=float)
    x = x - x.mean()                      # centred: the slope is interpretable

    comps = list(COMPARTMENTS.keys())
    slopes = {c: [] for c in comps}
    subs_used = []

    for sub in lps.SUBJECTS:
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
    """Two-sided sign-flipping test of mean != 0 (small N)."""
    v = np.asarray(v, dtype=float)
    obs = v.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, v.size))
    null = (signs * v).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def family_levels(design, family):
    """Level contrasts of a family -> [(contrast, condition, value)]."""
    info = design['info']
    levels = []
    for contrast, fam in lps.MAPPING.items():
        if fam != family or contrast in ('spmT_0001', 'spmT_0008', 'spmT_0009'):
            continue
        n = int(contrast.split('_')[1])
        conds = positive_conditions(info, n)
        if len(conds) != 1:
            print(f"  [WARNING] {contrast} positively weights more than one "
                  f"condition ({conds}): it is not a single-level "
                  f"contrast, skipping it.")
            continue
        levels.append((contrast, conds[0], level_value(conds[0])))
    return levels


def build_cube(table, design, region, family):
    """Matrix Y[subject, condition, compartment] + labels.

    Keeps only subjects with all cells present: the repeated-measures
    analysis requires a balanced design.
    """
    levels = family_levels(design, family)
    comps = list(COMPARTMENTS.keys())
    if len(levels) < 2:
        return None, levels, comps

    rows, subs = [], []
    for sub in lps.SUBJECTS:
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
    """Condition x compartment interaction, repeated measures on both.

    Y has shape [subjects, conditions, compartments].

    The permutation p-value is obtained by first removing the main effects
    (double centring within subject) and then permuting the condition labels
    within each subject: in this way the null distribution concerns the
    interaction and not the main effects.
    """
    n, C, K = Y.shape

    def f_interaction(A):
        gm = A.mean()
        m_c = A.mean(axis=(0, 2))                     # means per condition
        m_k = A.mean(axis=(0, 1))                     # means per compartment
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

    # Double centring within subject: only interaction + residual remain
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
    """Condition x compartment ANOVA + depth-specific pairwise contrasts."""
    cube, levels, comps = build_cube(table, design, region, family)
    if cube is None:
        print(f"  [WARNING] {region} / {family}: insufficient data")
        return None
    Y, subs = cube
    names = [l[1] for l in levels]

    print(f"\n  {region} / {family}  (N={len(subs)}, conditions: {names})")

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
    print(f"    [ANOVA] condition x compartment: F({anova['df'][0]},"
          f"{anova['df'][1]})={anova['F']:.2f}  p={anova['p_param']:.4f}  "
          f"p_perm={anova['p_perm']:.4f}")

    # Pairwise contrasts: difference between conditions, then difference
    # between compartments (the latter is the depth-specific effect).
    pairs = MIRRORING_PAIRS if family == 'mirroring' else [
        (names[i], names[j]) for i in range(len(names))
        for j in range(i + 1, len(names))]

    for a, b in pairs:
        if a not in names or b not in names:
            print(f"    [SKIP] pair ({a}, {b}): condition missing")
            continue
        ia, ib = names.index(a), names.index(b)
        diff = Y[:, ia, :] - Y[:, ib, :]              # [subjects, compartments]

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

    out['method'] = lps.METHOD
    out['_cube'] = Y.tolist()
    return out


def analyse_ordinal(table, design, region, family):
    """Trend per compartment + interaction (differences between compartments)."""
    res, levels = subject_slopes(table, design, region, family)
    if res is None:
        print(f"  [WARNING] {region} / {family}: insufficient data")
        return None

    print(f"\n  {region} / {family}  (N={len(res['subjects'])}, "
          f"levels: {[(l[1], l[2]) for l in levels]})")

    out = {'n': len(res['subjects']), 'levels': [(l[1], l[2]) for l in levels],
           'per_compartment': {}, 'interactions': {}}

    for c, v in res['slopes'].items():
        t, p_t = stats.ttest_1samp(v, 0.0)
        mean, p_perm = sign_flip_test(v)
        d = mean / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan
        out['per_compartment'][c] = {'mean_slope': mean, 't': float(t),
                                     'p_t': float(p_t), 'p_perm': p_perm,
                                     'cohen_d': float(d)}
        print(f"    {c:>12}: slope {mean:+.4f} %/level  "
              f"t={t:+.2f}  p_t={p_t:.4f}  p_perm={p_perm:.4f}  d={d:+.2f}")

    # The interaction: does the slope differ between compartments?
    # This is the depth-specific test, immune to a constant vascular gain.
    for a, b in [('superficial', 'middle'), ('deep', 'middle'),
                 ('superficial', 'deep')]:
        if a not in res['slopes'] or b not in res['slopes']:
            continue
        diff = res['slopes'][a] - res['slopes'][b]
        t, p_t = stats.ttest_1samp(diff, 0.0)
        mean, p_perm = sign_flip_test(diff)
        out['interactions'][f'{a}-{b}'] = {'mean_diff': mean, 't': float(t),
                                           'p_t': float(p_t), 'p_perm': p_perm}
        print(f"    [interaction] {a} - {b}: {mean:+.4f}  "
              f"t={t:+.2f}  p_t={p_t:.4f}  p_perm={p_perm:.4f}")

    out['_slopes'] = {c: v.tolist() for c, v in res['slopes'].items()}
    return out


def analyse_region(table, design, region, family):
    """Chooses the analysis according to the family's mode."""
    mode = FAMILY_MODE.get(family, 'ordinal')
    if mode == 'categorical':
        return analyse_categorical(table, design, region, family)
    return analyse_ordinal(table, design, region, family)


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def plot_slopes(results, family):
    """Slopes per compartment (ordinal families), points = subjects."""
    regions = [r for r in lps.ROIS if results.get(r) is not None]
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

    axes[0][0].set_ylabel("PSC slope across levels (% per level)", fontsize=11)
    fig.suptitle(f"Effect of {family} level per compartment "
                 f"[{lps.METHOD}]\n"
                 f"differences between compartments are the depth-specific effect",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'slopes_{family}.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def plot_categorical(results, family):
    """Grouped bars: compartment on the x axis, one bar per condition."""
    regions = [r for r in lps.ROIS if results.get(r) is not None]
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
                     f"interaction F={a['F']:.2f}, p_perm={a['p_perm']:.3f}",
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=9, frameon=True)

    axes[0][0].set_ylabel("$\\Delta$ percent signal change (%)", fontsize=11)
    fig.suptitle(f"{family}: conditions per cortical compartment "
                 f"[{lps.METHOD}]",
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

    design = lps.load_design()
    print(f"\nProfiles from the single CSV ({lps.METHOD})...")
    table = build_profile_table(lps.load_or_extract(design))
    print(f"  {len(table)} level profiles")

    all_results = {}
    for family in ['mirroring', 'spacing']:
        mode = FAMILY_MODE.get(family, 'ordinal')
        print(f"\n{'=' * 60}\n Family: {family}  (mode: {mode})\n{'=' * 60}")
        results = {r: analyse_region(table, design, r, family) for r in lps.ROIS}
        all_results[family] = results
        if mode == 'categorical':
            plot_categorical(results, family)
        else:
            plot_slopes(results, family)

    all_results['_config'] = {'method': lps.METHOD,
                              'profiles_csv': lps.PROFILES_CSV,
                              'compartments': COMPARTMENTS}
    with open(os.path.join(OUT_DIR, 'ffb_stats.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n-> {os.path.join(OUT_DIR, 'ffb_stats.json')}")


if __name__ == '__main__':
    main()