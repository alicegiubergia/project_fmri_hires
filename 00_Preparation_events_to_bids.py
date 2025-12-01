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

# ============================================
# CONFIGURAZIONE
# ============================================

EVENTS_DIR = "/home/local/IMAGO7/agiubergia/medea/progetto_reading/logfiles"
BIDS_DIR = "/mnt/store2/projects/progetto_layer/data"
SUBJECTS = ["001"]  # lista soggetti
RUNS = [1, 2, 3]
TASK = "reading"
PART = "mag"  # parte NIfTI: mag / phase ecc.

# ============================================
# MAIN LOOP
# ============================================

for sub in SUBJECTS:
    for run in RUNS:
        # --------------------------
        # File eventi originale
        # --------------------------
        pattern = f"sub-{sub}_*run-{run}*.log"
        source_files = [f for f in os.listdir(EVENTS_DIR) if f.startswith(f"sub-{sub}") and f"run-{run}" in f]
        if not source_files:
            print(f"⚠️ File eventi mancante per sub-{sub}, run-{run}")
            continue

        source_file = os.path.join(EVENTS_DIR, source_files[0])

        # --------------------------
        # File di destinazione BIDS
        # --------------------------
        dest_dir = os.path.join(BIDS_DIR, f"sub-{sub}", "func")
        os.makedirs(dest_dir, exist_ok=True)

        dest_file = os.path.join(dest_dir, f"sub-{sub}_task-{TASK}_run-0{run}_events.tsv")
        shutil.copy2(source_file, dest_file)
        print(f"✅ Copiato: {source_file} → {dest_file}")

        # --------------------------
        # Aggiornamento JSON NIfTI
        # --------------------------
        nifti_file = os.path.join(dest_dir, f"sub-{sub}_task-{TASK}_run-0{run}_part-{PART}_bold.nii.gz")
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
