#!/bin/bash
set -e  # stop on error
set -u  # treat unset variables as error

# === CONFIGURAZIONE ===
TEMPLATE_DIR="../templates"
INPUT_DIR="../data/"
OUTPUT_DIR="../data/derivatives/hires_pipeline"

# === LOOP SU SOGGETTI ===
for i in $(seq -f "%03g" 7 8); do
    SUB="sub-${i}"
    echo "============================="
    echo ">>> Processing $SUB"
    echo "============================="

    T1="${INPUT_DIR}/${SUB}/anat/${SUB}_T1w.nii.gz"
    
    if [ ! -f "$T1" ]; then
        echo "⚠️  File non trovato: $T1"
        continue
    fi

    # Lancia lo script Python
    python -m preprocessing_scripts.preprocess_t1 \
        --input "$T1" \
        --sub "$i" \
        --output_dir "$OUTPUT_DIR" 

    echo "[DONE] subject "$SUB""
done

echo "✅ Tutti i soggetti completati!"



