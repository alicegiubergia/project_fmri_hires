#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

BASE_DIR="/mnt/store2/projects/progetto_layer/data/derivatives/"

FS_DIR="${BASE_DIR}/freesurfer"
PIPE_DIR="${BASE_DIR}/mypipeline"
OUT_ROOT="${BASE_DIR}/cerebrum7t"

for SUB_DIR in "${FS_DIR}"/sub-*; do

  # Salta se non è una directory
  [[ -d "${SUB_DIR}" ]] || continue

  SUB=$(basename "${SUB_DIR}")

  ASEG="${SUB_DIR}/mri/aseg.mgz"
  T1="${PIPE_DIR}/${SUB}/anat/${SUB}_desc-N4_T1w.nii.gz"
  LTA="${PIPE_DIR}/${SUB}/anat/${SUB}_desc-N4_T1w_2fs.lta"
  OUT_DIR="${OUT_ROOT}/${SUB}/anat"
  OUT_FILE="${OUT_DIR}/${SUB}_desc-aseg_space-anat_seg.nii.gz"

  # Controlli di esistenza
  if [[ ! -f "${ASEG}" ]]; then
    echo "ASEG non trovato per ${SUB}, salto."
    continue
  fi

  if [[ ! -f "${T1}" ]]; then
    echo "T1 non trovata per ${SUB}, salto."
    continue
  fi

  if [[ ! -f "${LTA}" ]]; then
    echo "LTA non trovata per ${SUB}, salto."
    continue
  fi

  mkdir -p "${OUT_DIR}"

  echo "Processo ${SUB}..."

  mri_vol2vol \
    --mov "${ASEG}" \
    --targ "${T1}" \
    --o "${OUT_FILE}" \
    --lta "${LTA}" \
    --inv \
    --interp nearest

done