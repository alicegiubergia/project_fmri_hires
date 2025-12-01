#!/usr/bin/env python3
"""
=========================================================
Script Name: convert_physio_to_bids.py
Author: Alice Giubergia
Date: 2025-10-08
Version: 1.0
=========================================================

Description:
------------
This script converts physiological data (cardiac PPG and respiratory RESP signals) 
from raw scanner exports (in `sourcedata`) into BIDS-compliant physiological files 
(`.tsv` + `.json`), stored under the `nifti` directory.

For each participant, physiological data are matched to their corresponding BIDS 
subject IDs using the `participants.tsv` file, which must contain at least:
    - exam (scanner exam ID)
    - participant_id (BIDS subject label)

Each run produces:
    - A cardiac (PPG) file:  <subj>_task-reading_run-XX_recording-cardiac_physio.tsv/json/html
    - A respiratory (RESP) file: <subj>_task-reading_run-XX_recording-respiratory_physio.tsv/json/html

The script also:
    - Removes dummy scans from the start of the recording
    - Generates Plotly interactive HTML plots for visual inspection
    - Stores metadata in BIDS-style JSON sidecars

---------------------------------------------------------
Input Directory Structure:
--------------------------
./sourcedata/
    ├── <exam_id1>/
    │     └── gating/
    │           ├── PPGData_*.txt
    │           └── RESPData_*.txt
    ├── <exam_id2>/
          └── gating/ ...

Output Directory Structure:
---------------------------
./data/
    ├── sub-001/
    │     └── func/
    │           ├── sub-001_task-reading_run-01_recording-cardiac_physio.tsv
    │           ├── sub-001_task-reading_run-01_recording-cardiac_physio.json
    │           ├── sub-001_task-reading_run-01_recording-cardiac_physio.html
    │           ├── sub-001_task-reading_run-01_recording-respiratory_physio.tsv
    │           ├── sub-001_task-reading_run-01_recording-respiratory_physio.json
    │           └── sub-001_task-reading_run-01_recording-respiratory_physio.html
    └── sub-002/ ...

---------------------------------------------------------
Dependencies:
-------------
    - Python 3.8+
    - pandas
    - numpy
    - plotly
    - json
    - glob
    - os

Install dependencies (if needed):
    pip install pandas numpy plotly

---------------------------------------------------------
Usage:
------
1. Prepare your directories:
       ./sourcedata/
       ./data/
       ./participants.tsv

2. Edit acquisition parameters if needed (sampling rates, TR, dummy scans):
       PPG_SR, RESP_SR, TR, N_DUMMY

3. Run:
       python convert_physio_to_bids.py

---------------------------------------------------------
Notes:
------
- Dummy scans are automatically removed from the beginning of each signal.
- The script assumes the number of PPGData and RESPData files match per exam.
- Each run is numbered incrementally (run-01, run-02, ...).
- Plotly HTML visualizations allow quick QC of cardiac and respiratory signals.

=========================================================
"""
import os
import glob
import pandas as pd
import json
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# === PARAMETRI ===
SRC_DIR = "./sourcedata"
BIDS_DIR = "./data"
PARTICIPANTS_TSV = f"{BIDS_DIR}/participants.tsv"

# Parametri di acquisizione
PPG_SR = 100.0    # Hz (10 ms)
RESP_SR = 25.0    # Hz (40 ms)
TR = 3.0          # sec
N_DUMMY = 4       # numero dummy scans da rimuovere
REMOVE_SEC = 30   # rimuovere primi 30 secondi

# Filtri PhysioPy
PPG_LOW, PPG_HIGH = 0.5, 5.0     # Hz
RESP_LOW, RESP_HIGH = 0.1, 1.0   # Hz

# === LEGGI PARTICIPANTS FILE ===
df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", dtype={"participant_id": str})
exam_to_sub = dict(zip(df["exam"].astype(str), df["participant_id"]))

# === LOOP SOGGETTI ===
for subj_dir in sorted(glob.glob(os.path.join(SRC_DIR, "*"))):
    exam_id = os.path.basename(subj_dir)
    if exam_id not in exam_to_sub:
        print(f"⚠️  Nessun participant_id trovato per exam {exam_id}, salto.")
        continue

    subj = exam_to_sub[exam_id]
    gating_dir = os.path.join(subj_dir, "gating")
    if not os.path.isdir(gating_dir):
        continue

    print(f"🔹 Processing {subj} (exam {exam_id})")

    ppg_files = sorted(glob.glob(os.path.join(gating_dir, "PPGData*")))
    resp_files = sorted(glob.glob(os.path.join(gating_dir, "RESPData*")))

    for i, (ppg, resp) in enumerate(zip(ppg_files, resp_files), start=1):
        bids_func_dir = os.path.join(BIDS_DIR, f"sub-{subj}", "func")
        os.makedirs(bids_func_dir, exist_ok=True)
        run_prefix = f"sub-{subj}_task-reading_run-{i:02d}"

        # --- PPG (Cardiac) ---
        ppg_data = np.loadtxt(ppg)
        ppg_start = int(N_DUMMY * TR * PPG_SR + REMOVE_SEC * PPG_SR)
        ppg_data = ppg_data[ppg_start:]
        ppg_time = np.arange(len(ppg_data)) / PPG_SR

        ppg_tsv = os.path.join(bids_func_dir, f"{run_prefix}_recording-cardiac_physio.tsv")
        pd.DataFrame({"cardiac": ppg_data}).to_csv(ppg_tsv, sep="\t", index=False)

        ppg_json = os.path.join(bids_func_dir, f"{run_prefix}_recording-cardiac_physio.json")
        ppg_metadata = {
            "SamplingFrequency": PPG_SR,
            "StartTime": 0,
            "Columns": ["cardiac"],
            "DummyScansRemoved": N_DUMMY,
            "RemoveFirstSec": REMOVE_SEC,
            "TR": TR,
            "Manufacturer": "GE",
            "Description": "Cardiac (PPG) signal acquired during EPI sequence. Trimmed to exclude dummy scans and pre-sequence buffer.",
            "DurationSec": round(len(ppg_data) / PPG_SR, 2)
        }



        # --- RESP (Respiratory) ---
        resp_data = np.loadtxt(resp)
        resp_start = int(N_DUMMY * TR * RESP_SR + REMOVE_SEC * RESP_SR)
        resp_data = resp_data[resp_start:]
        resp_time = np.arange(len(resp_data)) / RESP_SR

        resp_tsv = os.path.join(bids_func_dir, f"{run_prefix}_recording-respiratory_physio.tsv")
        pd.DataFrame({"respiratory": resp_data}).to_csv(resp_tsv, sep="\t", index=False)

        resp_json = os.path.join(bids_func_dir, f"{run_prefix}_recording-respiratory_physio.json")
        resp_metadata = {
            "SamplingFrequency": RESP_SR,
            "StartTime": 0,
            "Columns": ["respiratory"],
            "DummyScansRemoved": N_DUMMY,
            "TR": TR,
            "Manufacturer": "GE",
            "Description": "Respiratory (RESP) signal acquired during EPI sequence. Trimmed to exclude dummy scans and pre-sequence buffer.",
            "DurationSec": round(len(resp_data) / RESP_SR, 2)
        }

        parts = ["mag", "real", "imag"]
        intended_files = [f"func/{run_prefix}_part-{p}_bold.nii.gz" for p in parts]

        ppg_metadata["IntendedFor"] = [intended_files]
        resp_metadata["IntendedFor"] = [intended_files] 

        with open(ppg_json, "w") as f:
            json.dump(ppg_metadata, f, indent=4)
        with open(resp_json, "w") as f:
            json.dump(resp_metadata, f, indent=4)


        print(f"  ✔ Run {i} salvato (cardiac + respiratory)")

        # --- PLOTLY VISUALIZATION ---

        # --- Combined Cardiac + Respiratory Plot ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                            vertical_spacing=0.2,
                            subplot_titles=("Cardiac (PPG)", "Respiratory (RESP)", "Respiratory Amplitude Distribution"))

        # Cardiac trace
        fig.add_trace(go.Scatter(
            x=ppg_time, y=ppg_data, mode="lines", line=dict(color="red"),
            name="Cardiac (PPG)"
        ), row=1, col=1)

        # Respiratory trace
        fig.add_trace(go.Scatter(
            x=resp_time, y=resp_data, mode="lines", line=dict(color="blue"),
            name="Respiratory (RESP)"
        ), row=2, col=1)

        # Respiratory amplitude distribution
        fig.add_trace(go.Histogram(
            x=resp_data,
            nbinsx=60,
            marker_color="skyblue",
            name="Respiratory amplitude",
            opacity=0.75
        ),
        row=3, col=1
        )
        
        fig.update_layout(
            title_text=f"{run_prefix} — Physiological signals",
            xaxis3_title="Amplitude (a.u.)",
            yaxis3_title="Count",
            xaxis2_title="Time (s)",
            yaxis1_title="Amplitude",
            #template="plotly_white",
            autosize=True 
        )

        combined_html = os.path.join(bids_func_dir, f"{run_prefix}_recording-physio_combined.html")
        fig.write_html(combined_html)

print("✅ Tutti i file fisiologici convertiti e salvati!")
