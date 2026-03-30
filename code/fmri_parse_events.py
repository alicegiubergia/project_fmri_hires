#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:33:05 2025

@author: alicegiubergia
"""

import pandas as pd
import glob

def process_logfile(logfile_path, output_path):
    trialinfo = pd.read_table(logfile_path, sep="\t", skiprows=2, skipfooter=17, engine='python')
    trialinfo = trialinfo[(trialinfo['Event Type'] == 'Picture') & (trialinfo['Code'] != 'END') & (trialinfo.Trial != 0)]
    trialinfo["Time"] = trialinfo["Time"] / 10000.0 # converti in secondi
    trialinfo["Duration"] = trialinfo["Duration"] / 10000.0 # converti in secondi

    # remove rows of dummy scans
    trialinfo = trialinfo[trialinfo.Trial != 0]
    #trialinfo.loc[(trialinfo.Code == 'Esegui'), 'Code'] = 'Apri-chiudi' # in questo caso le due condizioni sono sempre state apri-chiudi
    t0 = trialinfo.Time.values[0]
    trialinfo.Time = trialinfo.Time -t0
    trialinfo.to_csv(f"{output_path}")


if __name__ == '__main__':
    import os
    import re
    file_list = glob.glob('/home/local/IMAGO7/agiubergia/medea/progetto_reading/sourcedata/raw/sub-00/func/*log')
    for file in file_list:
        output_fn = os.path.basename(file)
        output_folder = os.path.dirname(file).replace('sourcedata/raw', 'derivatives')
        # Get number of run 
        match = re.search(r'run-(\d+)', file)
        if match:
            num_run = match.group(1)
            output_folder = f"{output_folder}/run-{num_run}"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        output_file = f"{output_folder}/{output_fn}"    
        if not os.path.isfile(output_file):
            process_logfile(file, output_file)
