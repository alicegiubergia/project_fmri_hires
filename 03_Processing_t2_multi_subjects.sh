#!/bin/bash
set -e  # stop on error
set -u  # treat unset variables as error

# === CONFIGURAZIONE ===
TEMPLATE_DIR="./templates"
INPUT_DIR="./data/"
OUTPUT_DIR="./data/derivatives/mypipeline"

# === LOOP SU SOGGETTI ===
for i in $(seq -f "%03g" 5 8); do
    SUB="sub-${i}"
    echo "============================="
    echo ">>> Processing $SUB"
    echo "============================="
    
    T1_proc="${OUTPUT_DIR}/${SUB}/anat/${SUB}_desc-N4_T1w.nii.gz"
    T1_mask="${OUTPUT_DIR}/${SUB}/anat/${SUB}_desc-brain_mask.nii.gz"
    T2="${INPUT_DIR}/${SUB}/anat/${SUB}_T2w.nii.gz"
    
    if [ ! -f "$T2" ]; then
        echo "⚠️  File non trovato: $T2"
        continue
    fi

    # Lancia lo script Python
    python ./code/preprocess_t2.py \
        --t2 "$T2" \
        --sub "$i" \
        --t1 "$T1_proc" \
        --t1_mask "$T1_mask" \
        --output_dir "$OUTPUT_DIR" &

done

wait || { echo "❌ Uno o più processi hanno fallito."; exit 1; }
echo "✅ Tutti i soggetti completati!"

