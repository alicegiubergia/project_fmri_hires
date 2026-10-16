#!/bin/bash
# ============================================================
# Script to run FreeSurfer 8.0 (via Docker) for preprocessing
# Step 3: resume recon-all (autorecon2-wm + autorecon3)
# ============================================================

# ----------- USER VARIABLES -----------
# Controllo che sia stato fornito un argomento
if [ $# -ne 1 ]; then
    echo "Usage: $0 <SUBJECT_ID>"
    exit 1
fi

SUBJECT_ID=$1
PROJECT_DIR="/mnt/store2/projects/progetto_layer"
FS_DOCKER_IMAGE="cesne/freesurfer:8.0"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"

# Input files
T1_FILE="${SUBJECT_ID}_desc-N4_T1w.nii.gz"
T2_FILE="${SUBJECT_ID}_desc-T1space_T2w.nii.gz"


# Output directories
FS_DIR="data/derivatives/freesurfer"

# ============================================================
# Uncomment STEP 3 only AFTER manual refinement is done!
# ============================================================

 echo ">>> STEP 3: Resuming recon-all (autorecon2-wm + autorecon3)..."
 docker run --rm -d \
   --user ${USER_ID}:${GROUP_ID} \
   --name ${SUBJECT_ID}_fs8_recon2 \
   -v ${PROJECT_DIR}:/monta \
   -e SUBJECTS_DIR=/monta/$FS_DIR \
   ${FS_DOCKER_IMAGE} recon-all \
     -autorecon2-wm \
     -autorecon3 \
     -subjid ${SUBJECT_ID}

echo ">>> Script finished for ${SUBJECT_ID}."
echo ">>> Remember: run STEP 3 ONLY after manual refinement."
