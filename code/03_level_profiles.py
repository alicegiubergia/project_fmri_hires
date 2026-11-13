"""
03 - Profiles per level + parametric modulation + inter-regional distance,
in PERCENT SIGNAL CHANGE.

Profiles are in PSC

    PSC = 100 * SF * (con / weights) / mean(constant betas)

and are NOT extracted here: they come from the single CSV of 01_export_profiles.py
(created on the fly if missing), the same one used by 04_laminar_models.R and
05_ff_fb_compartments.py.

Output: <ROOT_DIR>/laminar_psc/<METHOD>/03_level_profiles/
Usage:  [LAYER_METHOD=equivol] python 03_level_profiles.py

Contrast structure:
  con_0001   -> reading of NORMAL words > implicit rest
  con_0002..0004 -> mirrored words > normal words (3 levels)
  con_0005..0007 -> spaced words   > normal words (3 levels)

For each subject x ROI x condition we get:
  'read_normal' -> profile of con_0001 (normal reading), in PSC
  'level1..3'   -> profile of each of the three level contrasts, in Delta PSC
  'slope'       -> profile of the parametric slope over the three levels
                   (not an SPM contrast, it is derived) -- computed ONLY
                   for the families listed in PARAMETRIC_FAMILIES
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

import laminar_psc as lps

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = lps.step_dir('03_level_profiles')

SUBJECTS = lps.SUBJECTS
ROIS = lps.ROIS

BASELINE = 'con_0001'                  # normal word reading > rest
CONDITIONS = {
    'mirroring': ['con_0002', 'con_0003', 'con_0004'],
    'spacing':   ['con_0005', 'con_0006', 'con_0007'],
}

ROI_KEY_FOR = {fam: lps.TASK[fam] for fam in CONDITIONS}   # spmT_0008 / spmT_0009

LEVEL_VALUES = [1, 2, 3]

# The 'slope' profile assumes that the three levels are an ORDERED progression
# of the same manipulation. For spacing this holds (2 < 4 < 6). For
# mirroring, flip_LR and flip_UD are qualitatively different, so the
# "slope" must be read for what it is (see parametric_slopes): with three
# levels and centred x, it equals (level3 - level1)/2, i.e. a contrast
# between the first and the third con_ listed in CONDITIONS, which ignores
# the middle level. Here we keep it for both families.
PARAMETRIC_FAMILIES = ['mirroring', 'spacing']

# In the per-level profile figures, overlay in black the profile of normal
# word reading (con_0001), extracted in the ROI of the family.
SHOW_BASELINE = True
# The levels are Delta PSC relative to normal reading, the baseline is PSC
# relative to rest: the two quantities have very different scales. With True
# the baseline goes on a secondary axis on the right, with False on the same axis.
BASELINE_SECONDARY_AXIS = False

NR_LAYERS = lps.NR_LAYERS
# Compartments from laminar_psc.py (deep 1-4, middle 5-6, superficial 7-10),
# converted to slice edges on 0-based bins: [0, 4, 6, 10].
COARSE_NAMES = ['deep', 'middle', 'superficial']
COARSE_EDGES = [0] + [max(lps.COMPARTMENTS[c]) for c in COARSE_NAMES]
assert all(lps.COMPARTMENTS[c] == list(range(a + 1, b + 1))
           for c, a, b in zip(COARSE_NAMES, COARSE_EDGES[:-1], COARSE_EDGES[1:])), \
    "compartments must be contiguous bins in the order deep, middle, superficial"

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
    """Mean and SE ignoring NaNs, with n = number of values that are actually
    finite along the axis (not the nominal number of subjects)."""
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis).astype(float)
    sd = np.nanstd(arr, axis=axis, ddof=1)
    se = np.divide(sd, np.sqrt(n), out=np.full_like(sd, np.nan, dtype=float),
                   where=n > 1)
    return m, se


# ============================================================
# EXTRACTION
# ============================================================

def profiles_table(df):
    """Single long CSV -> {(sub, family, region): (base_prof, mat)}.

    base_prof: (NR_LAYERS,) profile of con_0001 in the ROI of the family.
    mat:       (3, NR_LAYERS) levels 1..3, in the order of CONDITIONS
               (laminar_psc.LEVELS uses the same order).
    As in the version that did its own extraction: if the baseline or one of
    the three levels is missing, the subject/family/ROI combination is skipped.
    """
    for fam, cons in CONDITIONS.items():
        expected = [c.replace('con_', 'spmT_') for c in cons]
        if expected != lps.LEVELS[fam]:
            raise ValueError(f"order of the {fam} levels differs from "
                             f"laminar_psc.LEVELS: {expected} vs {lps.LEVELS[fam]}")

    out = {}
    for (sub, fam, region), g in df.groupby(['subject', 'family', 'region']):
        base = g[g['condition'] == 'baseline']
        lv = g[g['condition'] == 'level']
        levels_list = [lv[lv['level'] == k] for k in range(1, len(CONDITIONS[fam]) + 1)]
        if base.empty or any(x.empty for x in levels_list):
            continue
        base_prof, _ = lps.profile_from_rows(base)
        mat = np.array([lps.profile_from_rows(x)[0] for x in levels_list])
        out[(sub, fam, region)] = (base_prof, mat)
    return out


# ============================================================
# STATISTICS
# ============================================================

def coarsen(prof10):
    """From 10 bins to 3 (deep / middle / superficial)."""
    return np.array([np.nanmean(prof10[..., a:b], axis=-1)
                     for a, b in zip(COARSE_EDGES[:-1], COARSE_EDGES[1:])]).T


def parametric_slopes(levels_by_layer, level_values):
    """
    Linear slope of activation over levels, for each layer.
    Input (n_levels, n_layers) -> output (n_layers,), in % per level.
    """
    x = np.asarray(level_values, dtype=float)
    x = x - x.mean()
    return (x @ levels_by_layer) / np.sum(x ** 2)


def profile_set(base_prof, mat, cond):
    """
    One profile vector per contrast -- no averaging across levels --
    plus the derived parametric-slope profile, only where the progression
    of levels is ordered.
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
    How a profile changes across depth: fit b0 + b1*d + b2*d^2.

    On a level profile (which is a DIFFERENCE relative to normal reading)
    these terms describe the shape of the ADDITIONAL recruitment, not of
    the total activation.

    positive b1 = more signal at the surface. It is the term most exposed to
    residual venous draining -- but on a differential contrast the common
    vascular gain cancels at first order, so here it is less dangerous than
    on read_normal.
    positive b2 = U-shaped profile, middle layers spared relative to
    superficial and deep: the expected feedback signature. Negative =
    concentration in the middle layers, feedforward signature. Leakage fills
    the central dip, so it works AGAINST a positive b2: finding one is
    conservative.
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
    Scores of the original scheme, on the coarse profile.
      FF = middle - (superficial + deep)/2   -> peak in the middle layers
      FB = deep - superficial                -> imbalance towards depth
    In PSC these scores are in percentage points and become comparable
    across subjects, which they were not in arbitrary units.
    """
    c = coarsen(np.asarray(prof10, dtype=float))
    deep, mid, sup = c[0], c[1], c[2]
    return deep, mid, sup, mid - (sup + deep) / 2.0, deep - sup


def cosine_distance(u, v, demean):
    """
    1 - cos(u, v) over the bins that are finite in BOTH profiles.

    demean=False -> raw vectors. Includes the constant component: if both
        ROIs have an on-average positive signal the distance stays small
        whatever the shape.
    demean=True  -> centred vectors; equivalent to 1 - Pearson r, range
        [0, 2]. It is the SHAPE measure.

    Note on the switch to PSC: the demeaned version is invariant to a
    constant scale factor per ROI, so it changes ONLY to the extent that
    the PSC denominator has a trend across cortical depth.
    If the diagnostics of 02_contrast_profiles.py (diagnostic_psc_denominator.png) show a
    flat denominator, these figures will be nearly identical to the previous ones.
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

# SF, constant betas and contrast weights (read from SPM.mat, per subject)
design = lps.load_design()

# Profiles from the single CSV (created if missing)
print(f"\nMetric: {lps.METHOD}")
PROFILES = profiles_table(lps.load_or_extract(design))

# Which experimental condition is behind each 'level': the level1/2/3
# labels in the figures follow the order of CONDITIONS, not an experimental
# order, so it is worth printing it.
_colmap = lps.parse_column_conditions(design['info'])
print("\nLevels -> conditions:")
for cond, cons in CONDITIONS.items():
    labels = []
    for k, con in enumerate(cons, start=1):
        n = int(con.split('_')[1])
        c = design['info']['contrasts'][n]['c']
        pos = [_colmap[int(j)][0] for j in np.nonzero(c > 1e-12)[0] + 1
               if int(j) in _colmap]
        labels.append(f"level{k}={con}={'+'.join(pos) if pos else '?'}")
    print(f"  {cond:10s}: " + "  ".join(labels))

# aggregated per (condition, ROI), for the group figures
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
            got = PROFILES.get((sub, cond, region))
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
                      f"(non-parametric) level1: lin={lin1:+.4f} "
                      f"quad={quad1:+.4f}")

# ------------------------------------------------------------------
# per-contrast table: coarse profile, FF/FB, laminar shape
# ------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, f'profile_scores_{lps.METHOD}.csv'), 'w') as f:
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
# inter-regional distances, per contrast
# ------------------------------------------------------------------
# agg[(cond, ptype, res, pair)] = list of demeaned distances (one per sub)
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

with open(os.path.join(OUTPUT_DIR, f'interregional_distance_{lps.METHOD}.csv'), 'w') as f:
    f.write("subject,condition,profile_type,resolution,roi_a,roi_b,"
            "cos_dist_raw,cos_dist_demeaned\n")
    for row in dist_rows:
        f.write("{},{},{},{},{},{},{:.6f},{:.6f}\n".format(*row))
    # Distances on the group MEAN profiles: they match the figures but are
    # not independent observations and have no SE. Descriptive only.
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

print("\n--- inter-regional cosine distance, 10 bins, demeaned ---")
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

# --- figure 1: distance over the three LEVELS, with the slope set apart ---
# The solid line only joins level1-2-3. For spacing this is an ordered
# progression of degradation: if the distance grows along the axis, the
# divergence between ROIs is driven by the manipulation. For mirroring the
# three points are distinct conditions and the axis is NOT ordered: the line
# joins them only for readability, its slope means nothing.
# The parametric slope is shown only for the families in PARAMETRIC_FAMILIES.

for res in ('10bin', '3bin'):
    for cond in CONDITIONS:
        show_slope = cond in PARAMETRIC_FAMILIES
        fig, ax = plt.subplots(figsize=(8.5, 5))
        xs = np.arange(len(LEVEL_ORDER))
        x_slope = len(LEVEL_ORDER) + 0.6          # after the separator

        PAIR_DODGE = 0.34
        JITTER = 0.045
        for pi, pair in enumerate(PAIRS):
            c = COLOR_PAIR.get(pair, 'black')

            # three levels: solid line
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

            # slope: isolated point, same scale but no line
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
            # separator: marks the change of quantity
            ax.axvline(len(LEVEL_ORDER) - 0.4, color='0.6', lw=1.2, ls='-')
            ax.set_xticks(list(xs) + [x_slope])
            ax.set_xticklabels(LEVEL_ORDER + ['slope\n(derived)'])
            ax.set_xlim(-0.4, x_slope + 0.5)
            ax.set_xlabel("Degradation level")
        else:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(LEVEL_ORDER)
            ax.set_xlim(-0.4, len(LEVEL_ORDER) - 0.6)
            ax.set_xlabel("Condition (unordered axis)")

        ax.axhline(1.0, color='k', lw=0.8, ls=':')   # uncorrelated profiles
        ax.set_ylabel("1 - r  (cosine distance, centred profiles)")
        ax.set_title(f"Inter-regional laminar dissimilarity: {cond} [{res}, {lps.METHOD}]",
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.4)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,
                    f'interregional_distance_{cond}_{res}.png'), dpi=300)
        plt.close(fig)

# --- figure 2: distance during NORMAL READING, separately ---
# How much the laminar profiles of the ROIs differ during normal word
# reading. It is a different question from figure 1 -- it is not about the
# effect of degradation but about the baseline laminar organisation of the
# reading network -- and on a different quantity: here the profiles are
# activations (PSC), not differences.
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
        ax.axhline(1.0, color='k', lw=0.8, ls=':')   # uncorrelated profiles
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a}\nvs {b}" for a, b in PAIRS], fontsize=9)
        ax.set_xlim(-0.5, len(PAIRS) - 0.5)
        # The ROI depends on the condition also for read_normal: same
        # contrast, but extracted in different masks.
        ax.set_title(f"{cond} ROI [{ROI_KEY_FOR[cond]}]", fontsize=10)
        ax.grid(True, axis='y', ls='--', alpha=0.4)
    axes[0].set_ylabel("1 - r  (cosine distance, centred profiles)")
    fig.suptitle(f"Inter-regional laminar dissimilarity during "
                 f"normal reading [{res}, {lps.METHOD}]", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'interregional_readnormal_{res}.png'), dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------
# profile figures
# ------------------------------------------------------------------

# --- normal reading profile ---
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
    ax.set_xlabel("Layer bin (1=WM, 10=CSF)")
    ax.set_ylabel("Percent signal change (%)\nnormal word reading > rest")
    ax.set_title(f"Normal reading profile in the {cond} ROI "
                 f"[{ROI_KEY_FOR[cond]}, {lps.METHOD}]", fontsize=12, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'readnormal_profile_{cond}.png'), dpi=300)
    plt.close(fig)

# --- per-level profiles (already condition > normal reading) ---
for cond in CONDITIONS:
    fig, axes = plt.subplots(1, len(ROIS), figsize=(5.4 * len(ROIS), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
 
    # shared limits for the secondary axis, so the panels stay comparable
    # on the baseline too
    base_lim = None
    if SHOW_BASELINE and BASELINE_SECONDARY_AXIS:
        all_vals = []
        for region in ROIS:
            b = results[(cond, region)]['read_normal']
            if b:
                mb, seb = mean_se(np.array(b))
                all_vals.append(mb - seb)
                all_vals.append(mb + seb)
        if all_vals:
            all_vals = np.concatenate(all_vals)
            margin = 0.05 * (np.nanmax(all_vals) - np.nanmin(all_vals) or 1.0)
            base_lim = (np.nanmin(all_vals) - margin, np.nanmax(all_vals) + margin)
 
    base_handle = None
    for k, (ax, region) in enumerate(zip(axes, ROIS)):
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
                    label=f"level {LEVEL_VALUES[li]}")
            ax.fill_between(bins, m - se, m + se, color=c, alpha=0.12,
                            edgecolor='none')
 
        if SHOW_BASELINE and r['read_normal']:
            base = np.array(r['read_normal'])
            mb, seb = mean_se(base)
            ax_b = ax.twinx() if BASELINE_SECONDARY_AXIS else ax
            line_obj, = ax_b.plot(bins, mb, marker='s', ms=4, lw=2, ls='--',
                               color='k', zorder=5,
                               label='normal reading > rest')
            ax_b.fill_between(bins, mb - seb, mb + seb, color='k',
                              alpha=0.10, edgecolor='none', zorder=4)
            base_handle = line_obj
            if BASELINE_SECONDARY_AXIS:
                if base_lim is not None:
                    ax_b.set_ylim(base_lim)
                # right-axis label only on the last panel
                if k == len(ROIS) - 1:
                    ax_b.set_ylabel("PSC (%) - normal reading > rest",
                                    fontsize=10)
                else:
                    ax_b.set_yticklabels([])
 
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_title(f"{region} (N={n})")
        ax.set_xlabel("Layer bin (1=WM, 10=CSF)")
        ax.set_xticks(bins)
        ax.grid(True, ls='--', alpha=0.4)
 
    axes[0].set_ylabel(f"$\\Delta$ percent signal change (%)\n"
                       f"{cond} > normal reading")
 
    h, l = axes[-1].get_legend_handles_labels()
    if base_handle is not None:
        h = list(h) + [base_handle]
        l = list(l) + ['normal reading > rest'
                       + (' (right axis)' if BASELINE_SECONDARY_AXIS else '')]
    axes[-1].legend(h, l, loc='best', frameon=True, fontsize=8)
 
    fig.suptitle(f"Laminar profile per level of {cond} [{lps.METHOD}]",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'profiles_{cond}.png'), dpi=300)
    plt.close(fig)

# --- parametric modulation profile (families in PARAMETRIC_FAMILIES only) ---
for cond in CONDITIONS:
    if cond not in PARAMETRIC_FAMILIES:
        print(f"[skip] parametric figure for {cond}: unordered levels")
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
        for s in arr:                   # with N=8 individual subjects matter
            ax.plot(bins, s, color=c, alpha=0.20, lw=0.8, zorder=0)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xticks(bins)
    ax.set_xlabel("Layer bin (1=WM, 10=CSF)")
    ax.set_ylabel("Parametric slope ($\\Delta$ PSC % per level)")
    ax.set_title(f"Parametric modulation: {cond} [{lps.METHOD}]",
                 fontsize=12, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,
                f'parametric_slope_{cond}.png'), dpi=300)
    plt.close(fig)

print(f"\nMetric: {lps.METHOD}")
print(f"Output in: {OUTPUT_DIR}")