"""
01 - Exports the laminar PSC profiles to the single CSV.

    <ROOT_DIR>/laminar_psc/<METHOD>/01_profiles/profiles_<METHOD>.csv

Long format: subject, region, family, roi_key, condition ("level" 1..3 or
"baseline" = con_0001), contrast, level, bin, psc, n_voxel, x, method.
It is the only profiles file: it is read by 03_level_profiles.py,
04_laminar_models.R and 05_ff_fb_compartments.py.

Usage:
    python 01_export_profiles.py                          # equivol
    LAYER_METHOD=equidist python 01_export_profiles.py     # equidist
    python 01_export_profiles.py --if-missing             # only if the CSV does not exist
"""
import os
import sys

import laminar_psc as lps

if __name__ == '__main__':
    if '--if-missing' in sys.argv[1:] and os.path.exists(lps.PROFILES_CSV):
        print(f"Already exists: {lps.PROFILES_CSV} (nothing to do)")
    else:
        lps.export_profiles()
    print(f"Next: Rscript 04_laminar_models.R {lps.PROFILES_CSV}")