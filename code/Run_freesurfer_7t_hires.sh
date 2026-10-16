#!/bin/bash
# =============================================================================
# FreeSurfer v8 -- Pipeline 7T hires con upsample + superfici grezze a 1mm
#
# Workflow:
#   1.  Upsample T1 e T2 a 1mm isotropico (mri_convert)
#   2a. recon-all -autorecon1 a 1mm  (skull-strip watershed provvisorio)
#   2b. [PAUSA] controllo manuale in Freeview, poi Invio per continuare
#   2c. recon-all -autorecon2 a 1mm: superfici "grezze" di partenza
#   3.  Copia superfici 1mm: soggetto hires
#   4a. recon-all -autorecon1 -hires con dati nativi 7T
#   4b. [PAUSA] controllo brainmask hires in Freeview, poi Invio per continuare
#   4c. recon-all -autorecon2 -autorecon3 -hires + expert.opts
#
# Dipendenze: Docker, FreeSurfer license file
# Immagine:   cesne/freesurfer:8.0
#
# Uso:
#   bash run_freesurfer_7T_hires.sh \
#       --t1  /path/to/T1.nii.gz \
#       --t2  /path/to/T2.nii.gz \
#       --sub sub-01 \
#       --sd  /path/to/SUBJECTS_DIR \
#       --expert /path/expert.opts 
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parsing argomenti
# ---------------------------------------------------------------------------
usage() {
  echo "Uso: $0 --t1 <T1.nii.gz> --t2 <T2.nii.gz> --sub <nome_soggetto> --sd <SUBJECTS_DIR> [--expert expert.opts]"
  exit 1
}

T1="" ; T2="" ; SUBJECT="" ; SUBJECTS_DIR="" 
EXPERT_OPTS=""; 

while [[ $# -gt 0 ]]; do
  case "$1" in
    --t1)         T1=$(realpath "$2");              shift 2 ;;
    --t2)         T2=$(realpath "$2");              shift 2 ;;
    --sub)        SUBJECT="$2";                     shift 2 ;;
    --sd)         SUBJECTS_DIR=$(realpath "$2");    shift 2 ;;
    --expert)     EXPERT_OPTS=$(realpath "$2");     shift 2 ;;

    *) echo "Unknown argument: $1"; usage ;;
  esac
done

[[ -z "$T1" || -z "$T2" || -z "$SUBJECT" || -z "$SUBJECTS_DIR" ]] && usage

SCRIPT_DIR=$(dirname "$(realpath "$0")")

## ---------------------------------------------------------------------------
# Configurazione Docker
# ---------------------------------------------------------------------------
DOCKER_IMAGE="cesne/freesurfer:8.0"

# Tutte le path montate nel container sotto /data
DATA_DIR=$(dirname "$T1")   # assumiamo T1 e T2 nella stessa cartella
T1_NAME=$(basename "$T1")
T2_NAME=$(basename "$T2")

# Gestione expert.opts hires
EXPERT_MOUNT_FLAG=""
EXPERT_CONTAINER_PATH=""
if [[ -n "$EXPERT_OPTS" ]]; then
  EXPERT_DIR=$(dirname "$EXPERT_OPTS")
  EXPERT_NAME=$(basename "$EXPERT_OPTS")
  EXPERT_MOUNT_FLAG="-v ${EXPERT_DIR}:/expert_opts"
  EXPERT_CONTAINER_PATH="/expert_opts/${EXPERT_NAME}"
fi


# Directory di lavoro per il volume upsampled (dentro SUBJECTS_DIR per semplicità)
UPSAMP_DIR="${SUBJECTS_DIR}/${SUBJECT}_upsamp"
mkdir -p "$UPSAMP_DIR" "$SUBJECTS_DIR"

echo "============================================================"
echo " Subject     : $SUBJECT"
echo " T1 native   : $T1"
echo " T2 native   : $T2"
echo " SUBJECTS_DIR: $SUBJECTS_DIR"
echo " Upsampled   : $UPSAMP_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: esegue un comando dentro il container Docker
# ---------------------------------------------------------------------------
docker_run() {
  docker run --rm --user $(id -u):$(id -g) \
    -v "${DATA_DIR}:/data_in" \
    -v "${SUBJECTS_DIR}:/subjects" \
    ${EXPERT_MOUNT_FLAG:-} \
    -e "SUBJECTS_DIR=/subjects" \
    "$DOCKER_IMAGE" \
    "$@"
}

# ---------------------------------------------------------------------------
# STEP 1 -- Upsample a 1mm isotropico con interpolazione sinc
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 1: Upsample T1 e T2 a 1mm isotropico ..."
 
docker_run mri_convert \
  --voxsize 1 1 1 \
  --resample_type cubic \
  "/data_in/${T1_NAME}" \
  "/subjects/${SUBJECT}_upsamp/T1_1mm.nii.gz"
 
docker_run mri_convert \
  --voxsize 1 1 1 \
  --resample_type cubic \
  "/data_in/${T2_NAME}" \
  "/subjects/${SUBJECT}_upsamp/T2_1mm.nii.gz"
 
echo "    Upsample completato."
 
 
# ---------------------------------------------------------------------------
# STEP 2a — recon-all -autorecon1 a 1mm
#
#   Esegue solo il primo blocco: motion correction, talairach, skull-strip
# --------------------------------------------------------------------------- 

echo ""
echo ">>> STEP 2a: recon-all -autorecon1 a 1mm ..."

 
docker_run recon-all \
  -s "$SUBJ_1MM" \
  -i "/subjects/${SUBJECT}_upsamp/T1_1mm.nii.gz" \
  -T2 "/subjects/${SUBJECT}_upsamp/T2_1mm.nii.gz" \
  -autorecon1 \
  -noappend \
  -mprage \
  -gcareg \
  -canorm

echo "    autorecon1 completato."



# ---------------------------------------------------------------------------
# STEP 2b — ri-run mri_synthstrip --no-csf → sovrascrive brainmask.mgz
#
#   FS8 chiama synthstrip internamente ma senza --no-csf.
#   Lo richiamiamo manualmente passando T1.mgz come input
#   e sovrascrivendo brainmask.mgz con la mask risultante.
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 2b: mri_synthstrip --no-csf su dati 1mm ..."
 
docker_run mri_synthstrip \
  --no-csf \
  -i "/subjects/${SUBJ_1MM}/mri/T1.mgz" \
  -o "/subjects/${SUBJ_1MM}/mri/synthstrip-nocsf.mgz"

cp ${SUBJECTS_DIR}/${SUBJ_1MM}/mri/synthstrip-nocsf.mgz  ${SUBJECTS_DIR}/${SUBJ_1MM}/mri/brainmask.mgz  
echo "    brainmask.mgz 1mm aggiornata con --no-csf."
 
# ---------------------------------------------------------------------------
# STEP 2c -- [PAUSA] Controllo manuale della brainmask in Freeview
#
#   Lo script si ferma qui e mostra le istruzioni per aprire Freeview.
#   Premi Invio per continuare, oppure Ctrl+C per uscire e correggere
#   manualmente brainmask.mgz prima di rilanciare con --skip-to-autorecon2.
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [PAUSA -- CONTROLLO BRAINMASK]"
echo ""
echo "  Apri Freeview per verificare la brainmask synthstrip:"
echo ""
echo "  freeview \\"
echo "    ${SUBJECTS_DIR}/${SUBJ_1MM}/mri/T1.mgz \\"
echo "    ${SUBJECTS_DIR}/${SUBJ_1MM}/mri/brainmask.mgz:colormap=heat:opacity=0.4"
echo ""
echo "  Se la mask è OK  → premi Invio per continuare"
echo "  Se va corretta   → premi Ctrl+C, modifica brainmask.mgz,"
echo "                      poi rilancia con --skip-to-autorecon2"
echo "============================================================"
read -rp "  Premi Invio per continuare con autorecon2 ..."

# ---------------------------------------------------------------------------
# STEP 2d -- recon-all -autorecon2 a 1mm (superfici grezze di partenza)
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 2c: recon-all -autorecon2 a 1mm (superfici grezze) ..."
 
docker_run recon-all \
  -s "$SUBJ_1MM" \
  -mprage \
  -autorecon2 
echo "    recon-all 1mm completato."

docker_run recon-all \
  -s "$SUBJ_1MM" \
  -T2 "/subjects/${SUBJECT}_upsamp/T2_1mm.nii.gz" \
  -T2pial \
  -mprage \
  -autorecon-pial
echo "    recon-all 1mm completato."


# ---------------------------------------------------------------------------
# STEP 3 -- recon-all -autorecon1 -hires con i dati nativi 7T
#
#   -hires  : abilita la modalità subvoxel (FreeSurfer ≥ 6)
#   -T2pial : usa il T2 per affinare la superficie pial
#   I volumi nativi vengono passati direttamente (0.65×0.4375×0.4375 mm)
# ---------------------------------------------------------------------------

echo ""
echo ">>> STEP 3: recon-all -autorecon1 -hires con dati nativi 7T ..."
 
docker_run recon-all \
  -s "$SUBJ_HIRES" \
  -all \
  -hires \
  -i "/data_in/${T1_NAME}" \
  -T2 "/data_in/${T2_NAME}" \
  -T2pial \
  -mprage \
  -expert "${EXPERT_CONTAINER_PATH}"

echo "    autorecon1 hires completato."

#docker_run mri_normalize \
#  -g 1 \
#  -mprage \
#  "/subjects/${SUBJ_HIRES}/mri/nu.mgz" \
#  "/subjects/${SUBJ_HIRES}/mri/T1.mgz"

# ---------------------------------------------------------------------------
# STEP 3 -- Copia le superfici 1mm come punto di partenza per il soggetto hires
#
#   FreeSurfer -hires cerca le superfici iniziali in <SUBJECTS_DIR>/<subj>/surf/
#   Se le trova già, le usa come warm-start invece di partire da zero.
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 3: Upsampling of the formerly downsampled dataset and placing it in the according folder of the high resolution data."

for vol in aseg.auto_noCCseg aseg; do
  docker_run mri_convert \
    -rl "/subjects/${SUBJ_HIRES}/mri/orig.mgz" \
    -rt nearest \
    "/subjects/${SUBJ_1MM}/mri/${vol}.mgz" \
    "/subjects/${SUBJ_HIRES}/mri/${vol}.mgz"
  echo "    upsampled: ${vol}.mgz"
done

docker_run mri_convert \
  -rl "/subjects/${SUBJ_HIRES}/mri/orig.mgz" \
  -rt nearest \
  "/subjects/${SUBJ_1MM}/mri/brainmask.mgz" \
  "/subjects/${SUBJ_HIRES}/mri/brainmask.hires.mgz" \
  && echo "    upsampled: brainmask.mgz >> brainmask.hires.mgz"

docker_run mri_mask \
  "/subjects/${SUBJ_HIRES}/mri/T1.mgz" \
  "/subjects/${SUBJ_HIRES}/mri/brainmask.hires.mgz" \
  "/subjects/${SUBJ_HIRES}/mri/brainmask.mgz" \
  && echo "    brainmask.hires.mgz >> brainmask.mgz (mascherato su T1.mgz)"



 
# ---------------------------------------------------------------------------
# STEP 4 -- [PAUSA] Controllo brainmask hires in Freeview
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [PAUSA -- CONTROLLO BRAINMASK HIRES]"
echo ""
echo "  Apri Freeview per verificare la brainmask synthstrip hires:"
echo ""
echo "  freeview \\"
echo "    ${SUBJECTS_DIR}/${SUBJ_HIRES}/mri/T1.mgz \\"
echo "    ${SUBJECTS_DIR}/${SUBJ_HIRES}/mri/brainmask.mgz:colormap=heat:opacity=0.4"
echo ""
echo "  Se la mask è OK  → premi Invio per continuare"
echo "  Se va corretta   → premi Ctrl+C, modifica brainmask.mgz,"
echo "                      poi rilancia con --skip-to-autorecon2-hires"
echo "============================================================"
read -rp "  Premi Invio per continuare con autorecon2+3 hires ..."

# ---------------------------------------------------------------------------
# STEP 5 -- recon-all -autorecon2 -autorecon3 -hires con i dati nativi 7T
# ---------------------------------------------------------------------------

docker_run recon-all \
  -s "$SUBJ_HIRES" \
  -T2pial \
  -mprage \
  -hires \
#  -autorecon2 \
#  -autorecon3 

echo "    recon-all hires completato."

echo ""
echo "============================================================"
echo " Pipeline completata!"
echo " Superfici 1mm (grezze) : ${SUBJECTS_DIR}/${SUBJ_1MM}"
echo " Superfici hires        : ${SUBJECTS_DIR}/${SUBJ_HIRES}"
echo "============================================================"
 