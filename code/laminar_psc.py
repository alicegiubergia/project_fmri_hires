"""
laminar_psc.py - shared configuration and functions for the laminar PSC analysis.

Not meant to be run: it is imported by the numbered scripts.

    01_export_profiles.py     -> writes the single profiles CSV
    02_contrast_profiles.py   -> group profiles per contrast + diagnostics
    03_level_profiles.py      -> levels, parametric slope, inter-regional distances
    04_laminar_models.R       -> LMMs with AR(1) residuals
    05_ff_fb_compartments.py  -> feedforward/feedback test per compartment

Contents:
  - CONFIG: subjects, ROIs, contrasts, depth metric (METHOD) and
    output folder structure;
  - reading SPM.mat (SF, constant columns, contrast weights);
  - PSC maps = 100 * SF * (con / weights) / mean(constant betas), voxelwise,
    cached in laminar_psc/psc_maps/;
  - laminar sampling (10 bins, ROI GM only, no threshold);
  - extraction, writing and reading of the single profiles CSV.

Depth metric: LAYER_METHOD=equivol (default) or equidist.
    LAYER_METHOD=equidist python 01_export_profiles.py

Output structure:
    <ROOT_DIR>/laminar_psc/
        psc_maps/                     shared cache (does not depend on the metric)
        <METHOD>/01_profiles/profiles_<METHOD>.csv
        <METHOD>/02_contrast_profiles/
        <METHOD>/03_level_profiles/
        <METHOD>/04_laminar_models/
        <METHOD>/05_ff_fb_compartments/

Items marked [PLACEHOLDER] must be checked or filled in: they are values
that were not derived from your data.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import ants
from scipy.io import loadmat

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

ROOT_DIR = '/mnt/store1/projects/progetto_layer/data/derivatives'
BASE_DIR = os.path.join(ROOT_DIR, 'laminar_psc')
# The cached PSC maps do NOT depend on the depth metric (the metric only
# enters at sampling time), so psc_maps/ is shared between equidist and
# equivol.
PSC_DIR = os.path.join(BASE_DIR, 'psc_maps')

# First-level folder per subject: contains con_00XX and beta_00XX.
SPM_DIR_PATTERN = os.path.join(ROOT_DIR, 'hires_pipeline', '{sub}', 'stats-masked')

# Reference subject: used to print the design summary (SF per condition and
# per contrast, contrast weights) and as the benchmark for the checks. It is
# NOT the design used for the computation: SPM.mat is read for every subject,
# because the number of nuisance regressors (ART outliers) varies and the
# constant columns end up at different indices.
REFERENCE_SUBJECT = 'sub-001'

SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004",
            "sub-005", "sub-006", "sub-007", "sub-008"]

ROIS = ["dorsal", "ventral", "visual"]

# Keys unchanged: the spmT_XXXX strings stay because they appear in the ROI
# and metric paths. The matching con_ file is derived by substitution.
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

# Contrast 0001 is baseline > 0 (absolute PSC). All the others are
# differential (manipulated reading > normal reading): they are Delta PSC.
ABSOLUTE_CONTRASTS = {'spmT_0001'}

UPSAMP_SPACING = (0.2, 0.2, 0.2)
NR_LAYERS = 10

# Cortical compartments on the 10 bins (1 = WM, 10 = CSF), the same for all ROIs.
# Choice: option 2 of the "Laminar fractions" slide (BigBrain laminar fractions,
# BBW_laminar_fractions_schaefer.py). Single definition: used by
# 03_level_profiles.py, 05_ff_fb_compartments.py and the upcoming DCM.
COMPARTMENTS = {
    'deep':        [1, 2, 3, 4],
    'middle':      [5, 6],
    'superficial': [7, 8, 9, 10],
}

# Minimum constant beta for a voxel to be considered valid.
# With grand mean scaling to 100 the background is close to zero; 1.0 is my
# choice, not a value derived from your data. Check the histogram of the constant beta.
MIN_SESSION_MEAN = 1.0

# PSC is normalised voxel by voxel: it is divided by the constant beta of
# EACH voxel. This removes local receive-coil inhomogeneities, but carries any
# depth trend of the baseline signal into the profiles: check it in
# diagnostic_psc_denominator.png.
#
# METHOD is the ONLY place where the depth metric is chosen: every script
# reads it from here (lps.METHOD). It can be changed without touching the
# code, from the command line:
#     LAYER_METHOD=equidist python 01_export_profiles.py
METHOD = os.environ.get('LAYER_METHOD', 'equivol')

_VALID_METHODS = ('equidist', 'equivol')
if METHOD not in _VALID_METHODS:
    raise ValueError(f"LAYER_METHOD={METHOD!r}: allowed values {_VALID_METHODS}")

# All outputs of a metric live in <BASE_DIR>/<METHOD>/, one subfolder per
# script, named after the script.
METHOD_DIR = os.path.join(BASE_DIR, METHOD)


def step_dir(step):
    """Output folder of a script, e.g. step_dir('03_level_profiles')."""
    d = os.path.join(METHOD_DIR, step)
    os.makedirs(d, exist_ok=True)
    return d


PROFILES_CSV = os.path.join(METHOD_DIR, '01_profiles', f'profiles_{METHOD}.csv')

# Region-level DCM: mean signal over the whole ROI GM, so it does NOT
# depend on the depth metric and lives outside the <METHOD>/ folders.
DCM_DIR = os.path.join(BASE_DIR, 'dcm_region')

COLOR_ROI = {
    'visual': '#0f9d58',
    'ventral': '#ffab40',
    'dorsal': '#4285f4',
}


# Level contrasts per family, in the order level 1, 2, 3, and the x value
# of each level (centred at export).
LEVELS = {
    'mirroring': ['spmT_0002', 'spmT_0003', 'spmT_0004'],
    'spacing':   ['spmT_0005', 'spmT_0006', 'spmT_0007'],
}
X_LEVELS = {'mirroring': [1.0, 2.0, 3.0], 'spacing': [2.0, 4.0, 6.0]}
BASELINE = 'spmT_0001'


# ----------------------------------------------------------------------------
# READING SPM.mat
# ----------------------------------------------------------------------------

def read_spm_info(spm_dir):
    """Extracts from SPM.mat: dt, HRF, constant beta indices, condition durations."""
    mat_path = os.path.join(spm_dir, 'SPM.mat')
    m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    SPM = m['SPM']

    dt = float(SPM.xBF.dt)
    bf = np.asarray(SPM.xBF.bf, dtype=float)
    if bf.ndim > 1:
        bf = bf[:, 0]            # column 1 = canonical HRF

    # iB = (1-based) indices of the session constant columns
    iB = np.atleast_1d(np.asarray(SPM.xX.iB, dtype=int)).ravel()

    # Condition durations, first session
    sess = np.atleast_1d(SPM.Sess)[0]
    U = np.atleast_1d(sess.U)
    conditions = {}
    for u in U:
        name = np.atleast_1d(u.name)[0]
        dur = np.atleast_1d(np.asarray(u.dur, dtype=float))
        conditions[str(name)] = dur

    # Contrasts
    xCon = np.atleast_1d(SPM.xCon)
    contrasts = {}
    for i, c in enumerate(xCon, start=1):
        contrasts[i] = {
            'name': str(np.atleast_1d(c.name)[0]),
            'c': np.asarray(c.c, dtype=float).ravel(),
        }

    # Column names, e.g. 'Sn(1) baseline*bf(1)': needed to know which
    # condition sits in which column of the contrast.
    col_names = [str(x) for x in np.atleast_1d(SPM.xX.name)]

    return {'dt': dt, 'bf': bf, 'iB': iB, 'col_names': col_names,
            'conditions': conditions, 'contrasts': contrasts}


def compute_sf(dt, bf, duration):
    """Peak height of the convolved regressor for an isolated block."""
    n = max(1, int(round(duration / dt)))
    x = np.convolve(np.ones(n), bf)
    return float(x.max())


def condition_sfs(info, verbose=True):
    """SF of each condition in the design (diagnostics only, no checks)."""
    dt, bf = info['dt'], info['bf']
    per_cond = {}
    for name, durs in info['conditions'].items():
        d = np.asarray(durs, dtype=float)
        sf = compute_sf(dt, bf, float(d.mean()))
        per_cond[name] = sf
        if verbose:
            n_bins = max(1, int(round(float(d.mean()) / dt)))
            print(f"    '{name}': duration {d.mean():.3f} s "
                  f"(range {d.min():.3f}-{d.max():.3f}), "
                  f"{n_bins} microtime bins, SF = {sf:.4f}")
    return per_cond


def parse_column_conditions(info):
    """Map column index (1-based) -> (condition name, basis function index).

    Columns that are not condition regressors (motion, constants) are not
    in the map.
    """
    out = {}
    for j, nm in enumerate(info['col_names'], start=1):
        m = re.match(r'^Sn\(\d+\)\s+(.*?)\*bf\((\d+)\)$', nm.strip())
        if m:
            out[j] = (m.group(1), int(m.group(2)))
    return out


def contrast_sf(info, con_number, cond_sf, rel_tol=0.02):
    """Valid SF for a single contrast.

    Only looks at conditions with non-zero weight: the regressors of no
    interest (here END and fixPic) have a very different SF but do not enter
    the contrast, so they must not affect the computation.

    Stops if the conditions *actually weighted* have different SFs: in that
    case a single scalar applied to the con_ is not valid and PSC must be
    computed from the individual betas.
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
                f"con_{con_number:04d} weights basis function {bf_idx} "
                f"(column {j}, '{name}'). PSC as computed here "
                "is only valid for the canonical HRF.")
        # The parametric-modulator suffix is stripped ONLY if the name is not
        # already a known condition: 'fixPic'.split('x') would give 'fi'.
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
        print(f"    [WARNING] con_{con_number:04d}: weighted columns that do not "
              f"map to a condition: {unknown}. Check that they are not "
              "nuisance regressors weighted by mistake.")

    if not sfs:
        raise ValueError(f"con_{con_number:04d}: no recognised condition.")

    vals = np.array(list(sfs.values()))
    spread = (vals.max() - vals.min()) / vals.mean()
    if spread > rel_tol:
        raise ValueError(
            f"con_{con_number:04d} weights conditions with different SFs "
            f"({100 * spread:.1f}%): {sfs}. Compute PSC from the individual betas.")

    return float(vals.mean()), sorted(set(used_names))


def check_contrast_weights(info, con_number):
    """Checks that the positive contrast weights sum to 1.

    Why it matters: if spmT_0008 is the mean of the three levels, the contrast
    vector must be [1/3 1/3 1/3] and not [1 1 1]. With [1 1 1] the con_ is
    the sum, and PSC comes out 3 times too large. The same applies to
    differential contrasts: positive weights summing to 1, negative to -1.

    Returns the factor to divide by for normalisation (1.0 if ok).
    """
    c = info['contrasts'][con_number]['c']
    pos = c[c > 0].sum()
    neg = c[c < 0].sum()

    if pos == 0:
        raise ValueError(f"Contrast {con_number}: no positive weight.")

    ok_pos = abs(pos - 1.0) < 1e-6
    ok_neg = (neg == 0) or (abs(neg + 1.0) < 1e-6)

    if not (ok_pos and ok_neg):
        print(f"    [WARNING] con_{con_number:04d} "
              f"('{info['contrasts'][con_number]['name']}'): "
              f"sum of positive weights = {pos:.4f}, negative = {neg:.4f}. "
              f"Normalising by dividing by {pos:.4f}.")
    return float(pos)


# ----------------------------------------------------------------------------
# PSC MAPS
# ----------------------------------------------------------------------------

def _find(spm_dir, stem):
    hits = (glob.glob(os.path.join(spm_dir, stem + '.nii'))
            + glob.glob(os.path.join(spm_dir, stem + '.nii.gz')))
    return hits[0] if hits else None


def load_session_mean(spm_dir, iB):
    """Mean of the constant betas (one per session): the PSC denominator.

    Returns (array, reference image for affine/header).
    """
    means, ref_img = [], None
    for idx in iB:
        b_path = _find(spm_dir, f'beta_{int(idx):04d}')
        if b_path is None:
            raise FileNotFoundError(f"beta_{int(idx):04d} not found in {spm_dir}")
        img = nib.load(b_path)
        ref_img = img if ref_img is None else ref_img
        means.append(img.get_fdata())

    session_mean = np.mean(np.stack(means, axis=-1), axis=-1)
    session_mean[session_mean < MIN_SESSION_MEAN] = np.nan
    return session_mean, ref_img


def make_psc_map(spm_dir, con_number, SF, norm_factor, iB, out_path):
    """PSC = 100 * SF * con / mean(constant betas), normalised by the weights.

    iB must be the subject's own: the number of nuisance regressors varies,
    so the constant columns are not at the same indices for everyone.
    """
    con_path = _find(spm_dir, f'con_{con_number:04d}')
    if con_path is None:
        raise FileNotFoundError(f"con_{con_number:04d} not found in {spm_dir}")

    con_img = nib.load(con_path)
    con = con_img.get_fdata()

    session_mean, _ = load_session_mean(spm_dir, iB)
    psc = 100.0 * SF * (con / norm_factor) / session_mean

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(psc.astype(np.float32),
                             con_img.affine, con_img.header), out_path)
    return out_path


def resample_ants(input_path, output_path, spacing, interp_type=0):
    """Resample with ANTs. interp_type=0 = linear.

    Legitimate here: the PSC map is not thresholded, so interpolation does
    not mix valid values with structural zeros.
    """
    print(f"  [Resampling ANTs] -> {os.path.basename(output_path)}")
    img = ants.image_read(input_path)
    out = ants.resample_image(img, spacing, use_voxels=False,
                              interp_type=interp_type)
    ants.image_write(out, output_path)


# ----------------------------------------------------------------------------
# LAMINAR SAMPLING
# ----------------------------------------------------------------------------

def sample_map(map_path, metric_path, roi_path, nr_layers=NR_LAYERS):
    """Laminar profile of any map within the ROI GM.

    Returns (profile, voxel_count_per_bin).
    No statistical threshold: the ROI does the spatial selection.
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

    # Fix: clip so that voxels with metric == 1.0 end up in the last bin
    bins = np.clip((data_x * nr_layers).astype(int), 0, nr_layers - 1)

    profile = np.full(nr_layers, np.nan)
    counts = np.zeros(nr_layers, dtype=int)
    for i in range(nr_layers):
        sel = (bins == i)
        counts[i] = sel.sum()
        if counts[i] > 0:
            profile[i] = np.mean(data_y[sel])

    return profile, counts


def sample_psc_profile(sub, contrast, metric_path, roi_path, reference):
    """Laminar PSC profile (voxelwise PSC map). Returns (profile, counts)."""
    map_path = get_subject_psc_upsampled(sub, contrast, reference)
    return sample_map(map_path, metric_path, roi_path)


def subject_paths(sub, contrast, region):
    """Metric and ROI paths (unchanged: they use the spmT key)."""
    task_key = TASK[MAPPING[contrast]]
    metric = (f'{ROOT_DIR}/layer_fmri/{sub}/'
              f'{sub}_con-{task_key}_label-{region}_metric_{METHOD}.nii')
    roi = (f'{ROOT_DIR}/rois/{sub}/'
           f'{sub}_con-{task_key}_desc-polishCleaned_label-{region}'
           f'_res-up_dseg.nii.gz')
    return metric, roi


_DESIGN_CACHE = {}


def get_subject_design(sub, reference, verbose=True):
    """Design of a single subject, compared with the reference.

    Needed because the number of nuisance regressors (e.g. ART outliers)
    varies across subjects: the conditions are the same, but the constant
    columns end up at different indices. The con_ files are already correct
    by construction; what must be taken per subject is iB.
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
        print(f"  {sub}: {d['n_cols']} columns, "
              f"constants {info['iB'].tolist()}")

    # The conditions must be the same: if SF or weights change, the
    # comparison across subjects is no longer between homogeneous quantities.
    for n in sf:
        if abs(sf[n] - reference['sf'][n]) > 1e-6:
            print(f"    [WARNING] {sub} con_{n:04d}: SF {sf[n]:.4f} "
                  f"differs from the reference {reference['sf'][n]:.4f}")
        if abs(norm[n] - reference['norm'][n]) > 1e-6:
            print(f"    [WARNING] {sub} con_{n:04d}: contrast weights "
                  f"differ from the reference ({norm[n]:.4f} vs "
                  f"{reference['norm'][n]:.4f})")

    _DESIGN_CACHE[sub] = d
    return d


def get_subject_baseline_upsampled(sub, reference):
    """Upsampled map of the PSC denominator (mean of the constant betas).

    Used to check whether the denominator has a systematic trend across
    cortical depth: if it does, PSC normalisation is not a simple rescaling
    but changes the shape of the laminar profile.
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


def get_subject_psc_upsampled(sub, contrast, reference):
    """Generates (if needed) the upsampled PSC map for a subject and contrast.

    reference is the reference design produced by load_design(); the
    parameters actually used, however, are the subject's own.
    """
    con_number = int(contrast.split('_')[1])
    spm_dir = SPM_DIR_PATTERN.format(sub=sub)

    tag = 'psc'
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
    """Design summary from the reference subject: SF, iB, normalisations.

    The result is the benchmark for get_subject_design(), not the source of
    the parameters used in the computation.
    """
    ref_dir = SPM_DIR_PATTERN.format(sub=REFERENCE_SUBJECT)
    info = read_spm_info(ref_dir)

    print(f"Design read from {REFERENCE_SUBJECT} ({ref_dir})")
    print(f"  dt = {info['dt']:.4f} s")
    print("  SF per condition:")
    cond_sf = condition_sfs(info)
    print(f"  constant betas (1-based): {info['iB'].tolist()}")

    print("  contrasts:")
    sf, norm = {}, {}
    for contrast in MAPPING.keys():
        n = int(contrast.split('_')[1])
        c = info['contrasts'][n]
        norm[n] = check_contrast_weights(info, n)
        sf[n], used = contrast_sf(info, n, cond_sf)
        print(f"    con_{n:04d} '{c['name']}': conditions {used}, "
              f"SF = {sf[n]:.4f}, /{norm[n]:.4f}")

    return {'sf': sf, 'iB': info['iB'], 'norm': norm,
            'cond_sf': cond_sf, 'info': info}


# ----------------------------------------------------------------------------
# SINGLE PROFILES CSV
# ----------------------------------------------------------------------------
# con_0001 is sampled with the metric and ROI of the family (mirroring ->
# spmT_0008, spacing -> spmT_0009): the PSC map only depends on the contrast,
# while metric and ROI are passed as arguments.

def _sample(sub, contrast, metric_path, roi_path, design):
    try:
        return sample_psc_profile(sub, contrast, metric_path, roi_path, design)
    except Exception as e:
        print(f"    [SKIP] {sub} {contrast}: {e}")
        return None, None


def extract_profiles(design):
    """Extracts all profiles (levels 1..3 + con_0001) in long format."""
    rows = []
    for family, contrasts in LEVELS.items():
        x_centered = np.asarray(X_LEVELS[family]) - np.mean(X_LEVELS[family])
        roi_key = TASK[family]

        for region in ROIS:
            print(f"  {family} {region}: extracting profiles")
            for sub in SUBJECTS:
                print(f"    {sub}: ", end='', flush=True)
                paths = {subject_paths(sub, c, region) for c in contrasts}
                assert len(paths) == 1, f"{sub} {family} {region}: different paths across levels"
                metric_path, roi_path = paths.pop()
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    print(f"    [SKIP] {sub} {family} {region}: missing metric or ROI "
                          f"({os.path.basename(metric_path)})")
                    continue

                to_sample = [(c, 'level', i, x_centered[i - 1])
                                 for i, c in enumerate(contrasts, start=1)]
                to_sample.append((BASELINE, 'baseline', 0, np.nan))

                for contrast, condition, level, x in to_sample:
                    print(f"      {contrast} ({condition}): ", end='', flush=True)
                    prof, counts = _sample(sub, contrast, metric_path, roi_path, design)
                    if prof is None:
                        continue
                    rows += [{'subject': sub, 'region': region, 'family': family,
                               'roi_key': roi_key, 'condition': condition,
                               'contrast': contrast, 'level': level, 'bin': b + 1,
                               'psc': prof[b], 'n_voxel': int(counts[b]), 'x': x,
                               'method': METHOD}
                              for b in range(NR_LAYERS)]

    return pd.DataFrame(rows).dropna(subset=['psc'])


def export_profiles(design=None):
    """Extracts the profiles and writes the single CSV. Returns the DataFrame."""
    if design is None:
        design = load_design()
    os.makedirs(os.path.dirname(PROFILES_CSV), exist_ok=True)
    df = extract_profiles(design)
    if df.empty:
        raise SystemExit("No profile extracted")
    df.to_csv(PROFILES_CSV, index=False)
    print(df.groupby(['family', 'condition', 'region'])['subject'].nunique().unstack())
    print(f"\nSaved: {PROFILES_CSV}")
    return df


def load_or_extract(design=None):
    """Entry point for scripts 03 and 05.

    Reads the CSV of the current metric if it exists, otherwise creates it.
    Checks that the file really matches METHOD, so that equidist and equivol
    are never mixed by mistake.
    """
    if not os.path.exists(PROFILES_CSV):
        print(f"[laminar_psc] {PROFILES_CSV} does not exist: creating it.")
        return export_profiles(design)

    df = pd.read_csv(PROFILES_CSV, float_precision='round_trip')
    if 'method' not in df.columns:
        raise ValueError(f"{PROFILES_CSV}: column 'method' is missing. "
                         "Delete it and regenerate it with 01_export_profiles.py.")
    found = set(df['method'].unique())
    if found != {METHOD}:
        raise ValueError(f"{PROFILES_CSV}: method={found}, expected {METHOD}.")
    print(f"[laminar_psc] profiles read from {PROFILES_CSV} "
          f"({df['subject'].nunique()} subjects, {len(df)} rows)")
    return df


def profile_from_rows(g, nr_layers=NR_LAYERS):
    """Long-format rows of ONE profile -> (psc[nr_layers], n_voxel[nr_layers]).

    Missing bins (dropped by dropna at export) come back as NaN / 0.
    """
    prof = np.full(nr_layers, np.nan)
    counts = np.zeros(nr_layers, dtype=int)
    b = g['bin'].to_numpy(dtype=int) - 1
    prof[b] = g['psc'].to_numpy(dtype=float)
    counts[b] = g['n_voxel'].to_numpy(dtype=int)
    return prof, counts