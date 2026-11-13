"""
02 - Group laminar profiles per contrast, in PSC, plus diagnostics.

For each contrast (spmT_0001..0009, each in the ROI of its family):
mean across subjects +/- SE per ROI, plus the number of voxels per bin. Also:
  - diagnostic_psc_denominator.png: depth profile of the PSC denominator
    (constant beta), to check that voxelwise normalisation does not alter
    the shape of the profiles;
  - design_summary.json: SF, contrast weights, constant columns per subject.

Samples the maps directly (it also needs con_0001 in the baseline ROI and the
0008/0009 means, which are not in the single CSV) and saves no profiles.

Output: <ROOT_DIR>/laminar_psc/<METHOD>/02_contrast_profiles/
Usage:  [LAYER_METHOD=equivol] python 02_contrast_profiles.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import laminar_psc as lps

OUTPUT_DIR = lps.step_dir('02_contrast_profiles')

PLOT_VOXEL_COUNTS = True     # diagnostic panel with voxels per bin
PLOT_BASELINE = True         # laminar profile of the PSC denominator


def plot_baseline_profiles(design):
    """Laminar profile of the PSC denominator, per ROI set and per ROI.

    One panel per ROI set (baseline / mirroring / spacing): the masks differ,
    so they select different voxels and the baseline-signal profile is not
    necessarily the same.

    Row 1: normalised shape (profile divided by its mean over bins), because
    absolute intensity varies across subjects with receive gain. If the curve
    is flat, dividing by the denominator is a rescaling and does not touch the
    shape of the PSC profiles. If it is not, it does.
    Row 2: raw values.
    """
    print(f"\n{'=' * 50}\n Diagnostics: PSC denominator\n{'=' * 50}")

    families = list(lps.TASK.keys())          # baseline, mirroring, spacing
    fig, axes = plt.subplots(2, len(families),
                             figsize=(5.5 * len(families), 10),
                             squeeze=False, sharex=True)
    x_bins = np.arange(1, lps.NR_LAYERS + 1)
    has_data = False
    summary = {}

    for col, family in enumerate(families):
        ax_shape, ax_raw = axes[0, col], axes[1, col]
        roi_key = lps.TASK[family]
        # any contrast of this family: only needed for the paths
        contrast = next(c for c, f in lps.MAPPING.items() if f == family)

        for region in lps.ROIS:
            shapes, raws = [], []
            for sub in lps.SUBJECTS:
                metric_path, roi_path = lps.subject_paths(sub, contrast, region)
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    continue
                try:
                    b_path = lps.get_subject_baseline_upsampled(sub, design)
                except Exception as e:
                    print(f"    [SKIP] {sub}: {e}")
                    continue

                prof, _ = lps.sample_map(b_path, metric_path, roi_path)
                if prof is None:
                    continue
                raws.append(prof)
                m = np.nanmean(prof)
                shapes.append(prof / m if m and np.isfinite(m) else prof * np.nan)

            if not shapes:
                print(f"  [WARNING] no data for {family} / {region}")
                continue

            color = lps.COLOR_ROI.get(region, 'black')
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
            ax.set_xlim((1, lps.NR_LAYERS))
            ax.set_xticks(x_bins)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc="best", frameon=True, fontsize=9)

    if not has_data:
        plt.close(fig)
        return

    axes[0, 0].set_ylabel("Normalised denominator (bin mean = 1)",
                          fontsize=11)
    axes[1, 0].set_ylabel("Constant beta (arbitrary units)", fontsize=11)
    fig.suptitle("Shape of the PSC denominator across depth, "
                 f"per ROI set (metric: {lps.METHOD})\nflat = normalisation does not alter the "
                 "shape of the profiles", fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "diagnostic_psc_denominator.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f" -> saved: {save_path}")

    with open(os.path.join(OUTPUT_DIR,
                           'psc_denominator_profiles.json'), 'w') as f:
        json.dump(summary, f, indent=2)



# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------

def main():
    design = lps.load_design()

    # Each subject has a different number of nuisance regressors, so the
    # constant columns end up at different indices: iB is read per subject.
    print("\nPer-subject designs:")
    for sub in lps.SUBJECTS:
        try:
            lps.get_subject_design(sub, design)
        except Exception as e:
            print(f"  {sub}: [ERROR reading SPM.mat] {e}")

    with open(os.path.join(OUTPUT_DIR, 'design_summary.json'), 'w') as f:
        json.dump({'reference_subject': lps.REFERENCE_SUBJECT,
                   'method': lps.METHOD,
                   'condition_sf': design['cond_sf'],
                   'contrast_sf': {str(k): v for k, v in design['sf'].items()},
                   'norm_factors': {str(k): v
                                    for k, v in design['norm'].items()},
                   'per_subject': {s: {'n_cols': d['n_cols'],
                                       'iB': d['iB'].tolist()}
                                   for s, d in lps._DESIGN_CACHE.items()}},
                  f, indent=2)

    if PLOT_BASELINE:
        plot_baseline_profiles(design)

    for contrast in lps.MAPPING.keys():
        print(f"\n{'=' * 50}\n Contrast: {contrast} ({lps.MAPPING[contrast]})\n{'=' * 50}")

        n_panels = 2 if PLOT_VOXEL_COUNTS else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(9, 6 * n_panels),
                                 squeeze=False)
        ax = axes[0, 0]
        ax_n = axes[1, 0] if PLOT_VOXEL_COUNTS else None

        x_bins = np.arange(1, lps.NR_LAYERS + 1)
        has_data = False

        for region in lps.ROIS:
            print(f" -> ROI: {region}")
            profiles, counts_all = [], []

            for sub in lps.SUBJECTS:
                metric_path, roi_path = lps.subject_paths(sub, contrast, region)
                if not (os.path.exists(metric_path) and os.path.exists(roi_path)):
                    print(f"    [SKIP] {sub}: missing metric or ROI")
                    continue
                try:
                    prof, counts = lps.sample_psc_profile(
                        sub, contrast, metric_path, roi_path, design)
                except Exception as e:
                    print(f"    [SKIP] {sub}: {e}")
                    continue
                if prof is None:
                    print(f"    [SKIP] {sub}: no valid voxel")
                    continue
                profiles.append(prof)
                counts_all.append(counts)

            if not profiles:
                print(f"  [WARNING] no valid subject for {region}")
                continue

            M = np.array(profiles)
            n_valid = np.sum(~np.isnan(M), axis=0)
            mean_profile = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1)
            se = sd / np.sqrt(np.maximum(n_valid, 1))

            color = lps.COLOR_ROI.get(region, 'black')
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
            print(f" [WARNING] no data for {contrast}")
            continue

        is_abs = contrast in lps.ABSOLUTE_CONTRASTS
        ylab = ("Percent signal change (%)" if is_abs
                else "$\\Delta$ percent signal change (%)")
        subtitle = ("normal word reading > implicit rest" if is_abs
                    else "manipulated reading > normal word reading")

        ax.axhline(0, color='0.5', linewidth=1, linestyle='-')
        ax.set_xlim((1, lps.NR_LAYERS))
        ax.set_xticks(x_bins)
        ax.set_xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"Group mean layer profile ($\\pm$ SE)\n"
                     f"{contrast} ({lps.MAPPING[contrast]}) - {subtitle}\n"
                     f"metric: {lps.METHOD}",
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc="best", frameon=True)

        if ax_n is not None:
            ax_n.set_xlim((1, lps.NR_LAYERS))
            ax_n.set_xticks(x_bins)
            ax_n.set_xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
            ax_n.set_ylabel("Voxels per bin (mean over subjects)", fontsize=11)
            ax_n.set_title("Diagnostics: number of voxels sampled per bin",
                           fontsize=11)
            ax_n.grid(True, linestyle='--', alpha=0.4)
            ax_n.legend(loc="best", frameon=True)

        fig.tight_layout()
        save_path = os.path.join(
            OUTPUT_DIR,
            f"group_psc_{contrast}_{lps.MAPPING[contrast]}_{lps.METHOD}.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f" -> saved: {save_path}")


if __name__ == '__main__':
    main()