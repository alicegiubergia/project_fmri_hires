#!/usr/bin/env python3
"""
Script per copiare i file eventi e aggiornare i JSON dei NIfTI in BIDS.

- Copia i file .log da EVENTS_DIR → BIDS_DIR/sub-XX/func/ come sub-XX_task-reading_run-0X_events.tsv
- Aggiorna i JSON dei file NIfTI corrispondenti aggiungendo/modificando il campo:
  "Events": "<path relativo del file events.tsv>"
"""

import os
import json
import shutil
import pandas as pd
import glob

# ============================================
# CONFIGURAZIONE
# ============================================

SRC_DIR = "./sourcedata"
BIDS_DIR = "./data"
PARTICIPANTS_TSV = f"{BIDS_DIR}/participants.tsv"

RUNS = [1, 2, 3]
TASK = "reading"
PART = "mag"  # parte NIfTI: mag / phase ecc.

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
    print(f"🔹 Processing {subj} (exam {exam_id})")

    for run in RUNS:
        # --------------------------
        # File eventi originale
        # --------------------------
        pattern = f"sub-{subj}_*run-{run:02d}*.log"
        source_files = [f for f in os.listdir(subj_dir) if f.startswith(f"sub-{subj}") and f"run-{run:02d}" in f]
        if not source_files:
            print(f"⚠️ File eventi mancante per sub-{subj}, run-{run:02d}, salto.")
            continue

        source_file = os.path.join(
            subj_dir,
            next(f for f in source_files if f.endswith(".log"))
        )

        # --------------------------
        # File di destinazione BIDS
        # --------------------------
        dest_dir = os.path.join(BIDS_DIR, f"sub-{subj}", "func")
        os.makedirs(dest_dir, exist_ok=True)

        dest_file = os.path.join(dest_dir, f"sub-{subj}_task-{TASK}_run-{run:02d}_events.tsv")
        shutil.copy2(source_file, dest_file)
        print(f"✅ Copiato: {source_file} → {dest_file}")

        # --------------------------
        # Aggiornamento JSON NIfTI
        # --------------------------
        nifti_file = os.path.join(dest_dir, f"sub-{subj}_task-{TASK}_run-{run:02d}_part-{PART}_bold.nii.gz")
        json_file = nifti_file.replace(".nii.gz", ".json")

        if not os.path.exists(nifti_file):
            print(f"⚠️ NIfTI mancante: {nifti_file}")
            continue

        # Leggi JSON esistente o crea uno nuovo
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Aggiorna campo Events
        metadata["Events"] = os.path.relpath(dest_file, BIDS_DIR)

        # Salva JSON aggiornato
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"📝 Aggiornato JSON: {json_file} → Events: {metadata['Events']}")

print("\n🎯 Tutti i run processati con successo.")
