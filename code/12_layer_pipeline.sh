#!/bin/bash
set -euo pipefail

# =====================================================================================
# USAGE
# =====================================================================================
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sub_id>"
    echo "Example: $0 sub-001"
    exit 1
fi

SUB=$1

# =====================================================================================
# CONFIGURAZIONE
# =====================================================================================
# Path
BASE_DIR="/mnt/store1/projects/progetto_layer/data/derivatives"
WD="${BASE_DIR}/layer_fmri/${SUB}"
ROI_DIR="${BASE_DIR}/rois/${SUB}"

# Parameters
UPSAMP_SPACING="0.2x0.2x0.2"
NR_LAYERS=10
SPMT=("spmT_0001" "spmT_0008" "spmT_0009")
ROIS=("visual" "ventral" "dorsal")

mkdir -p "${WD}"
cd "${WD}" # nella cartella del soggetto 

echo "====================================================="
echo " Subject: ${SUB}"
echo " Working directory: ${WD}"
echo "====================================================="

# =====================================================================================
# FUNZIONE: RESAMPLING CON ANTs
# =====================================================================================
# INTERP:
#   0 = Linear (T1, EPI, beta / contrast)
#   1 = NN     (segmentazioni, ROI, RIM)      

resample_ants() {

    DIM=$1
    IN=$2
    OUT=$3
    SPACING=$4
    INTERP=$5


    echo "Resampling:"
    echo "  IN  : ${IN}"
    echo "  OUT : ${OUT}"
    echo "  SP  : ${SPACING}"
    echo "  INT : ${INTERP}"

    ResampleImage "${DIM}" \
        "${IN}" \
        "${OUT}" \
        "${SPACING}" \
        0 \
        "${INTERP}"
}

# =====================================================================================
# INPUT FILES
# =====================================================================================

T1="${BASE_DIR}/hires_pipeline/${SUB}/anat/${SUB}_desc-N4_T1w.nii.gz"
#EPI="${BASE_DIR}/mypipeline/${SUB}/func/run-01/${SUB}_run-01_desc-preproc_space-anat_bold.nii.gz"
RIBBON_MGZ="${BASE_DIR}/fs/${SUB}/mri/ribbon.mgz"
MGZ_NATIVE="${BASE_DIR}/fs/${SUB}/mri/rawavg.mgz"

# =====================================================================================
# OUTPUT FILES
# =====================================================================================

RIBBON_NII="${WD}/ribbon.nii.gz"
T1_UP="${WD}/${SUB}_res-up_T1w.nii.gz"
RIBBON_RIM="${WD}/${SUB}_space-fs_label-rim_dseg.nii.gz"
RIBBON_NATIVE="${WD}/${SUB}_space-native_label-rim_dseg.nii.gz"
RIBBON_NATIVE_UP="${WD}/${SUB}_space-native_res-up_label-rim_dseg.nii.gz"

# =====================================================================================
# 1. UPSAMPLING T1 ed EPI (LINEAR)
# =====================================================================================

echo "=== [1] Upsampling T1 ==="
#EPI_UP="${WD}/${SUB}_epi_upsampled.nii.gz"

resample_ants 3 "${T1}"  "${T1_UP}"  "${UPSAMP_SPACING}" 0
#resample_ants 4 "${EPI}" "${EPI_UP}" "${UPSAMP_SPACING}x3" 0

# =====================================================================================
# 2. STEDI – CORREZIONE ERRORI DI INTERPOLAZIONE
# =====================================================================================

echo "=== [2] STEDI correction ==="
segmentator_filters "${T1_UP}" \
  --feature_scale 2.0 \
  --save_every 5 \
  --nr_iterations 10

candidate_file=$(ls ${WD}/${SUB}_res-up_T1w_STEDI_n10*.nii.gz)
T1_STEDI="${WD}/${SUB}_res-up_desc-STEDI_T1w.nii.gz"

cp $candidate_file $T1_STEDI

# Remove intermediate files
rm ${WD}/*_g1.nii.gz

# =====================================================================================
# 3. GRADIENT MAGNITUDE
# =====================================================================================

echo "=== [3] Gradient magnitude ==="

LN2_GRAMAG -input "${T1_STEDI}"

# =====================================================================================
# 4. SEGMENTAZIONE (FreeSurfer → RIM → spazio T1)
# =====================================================================================

echo "=== [4] FreeSurfer segmentation to RIM ==="

#python /mnt/store2/projects/progetto_layer/code/freesurfer_segmentation_to_rim.py "${ASEG}"

mri_convert ${RIBBON_MGZ} ${RIBBON_NII}
LN2_RIMIFY -input "${RIBBON_NII}" -innergm 2 -outergm 0 -gm 3 -output "${WD}/tmp_left.nii.gz"
LN2_RIMIFY -input "${RIBBON_NII}" -innergm 41 -outergm 0 -gm 42 -output "${WD}/tmp_right.nii.gz"

fslmaths "${WD}/tmp_left.nii.gz" -add "${WD}/tmp_right.nii.gz" -mas ${RIBBON_NII} ${RIBBON_RIM}
fslmaths ${RIBBON_NII} -binv ${WD}/tmp_ribbon_inv.nii.gz
fslmaths "${WD}/tmp_left.nii.gz" -mas ${WD}/tmp_ribbon_inv.nii.gz ${WD}/tmp_out.nii.gz
fslmaths ${RIBBON_RIM} -add ${WD}/tmp_out.nii.gz ${RIBBON_RIM}

# Back to native space
mri_vol2vol --nearest \
  --mov "${RIBBON_RIM}" \
  --targ "${MGZ_NATIVE}" \
  --regheader \
  --o ${RIBBON_NATIVE} \
  --no-save-reg

# Remove temporary files
rm ${WD}/tmp_*.nii.gz

# =====================================================================================
# 5. UPSAMPLING RIM e ROI (NN)
# =====================================================================================

echo "=== [5a] Upsampling RIM ==="

resample_ants 3 "${RIBBON_NATIVE}" "${RIBBON_NATIVE_UP}" "${UPSAMP_SPACING}" 1

echo "=== [5b] Upsampling ROIS ==="

for contrast in "${SPMT[@]}"; do
  for region in "${ROIS[@]}"; do
    echo "Upsampling: $contrast - $region"
    resample_ants 3 "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_mask.nii.gz" "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_res-up_mask.nii.gz" "${UPSAMP_SPACING}" 1
  done
done

# =====================================================================================
# 6. MASKING DELLA SEGMENTAZIONE
# =====================================================================================

echo "=== [6] Masking segmentation to ROI ==="
  
for contrast in "${SPMT[@]}"; do
  for region in "${ROIS[@]}"; do
    echo "Masking segmentation: $contrast - $region"
    fslmaths ${RIBBON_NATIVE_UP} -mul "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_res-up_mask.nii.gz" "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_res-up_dseg.nii.gz"
  done
done

# =====================================================================================
# 7. RIM POLISH (AUTOMATICO)
# =====================================================================================

echo "=== [7] RIM polish (automatic) ==="
  
for contrast in "${SPMT[@]}"; do
  for region in "${ROIS[@]}"; do
    echo "Polishing: $contrast - $region"
    LN2_RIM_POLISH -rim "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_res-up_dseg.nii.gz" -output "${ROI_DIR}/${SUB}_con-${contrast}_desc-polish_label-${region}_res-up_dseg.nii.gz"
  done
done
  
# =====================================================================================
# 8. CORREZIONE MANUALE
# =====================================================================================

echo ""
echo ">>> MANUAL INTERVENTION REQUIRED <<<"
echo "itksnap ${ROI_DIR}/${SUB}_con-${contrast}_desc-polish_label-${region}_res-up_dseg.nii.gz"
echo ""

read -p "Press ENTER when manual correction is done"


# =====================================================================================
# 9. RIM POLISH (POST-MANUALE)
# =====================================================================================

echo "=== [9] RIM polish (post-manual) ==="

for contrast in "${SPMT[@]}"; do
  for region in "${ROIS[@]}"; do
    echo "Final polishing: $contrast - $region"
    LN2_RIM_POLISH -rim "${ROI_DIR}/${SUB}_con-${contrast}_desc-manual_label-${region}_res-up_dseg.nii.gz" -output "${ROI_DIR}/${SUB}_con-${contrast}_desc-polishFinal_label-${region}_res-up_dseg.nii.gz"
  done
done
  
  
for contrast in "${SPMT[@]}"; do
  for region in "${ROIS[@]}"; do
    echo "Cleaned RIM: $contrast - $region"
    fslmaths "${ROI_DIR}/${SUB}_con-${contrast}_desc-polishFinal_label-${region}_res-up_dseg.nii.gz" -mas "${ROI_DIR}/${SUB}_con-${contrast}_label-${region}_res-up_mask.nii.gz" "${ROI_DIR}/${SUB}_con-${contrast}_desc-polishCleaned_label-${region}_res-up_dseg.nii.gz"
  done
done
    
  
# =====================================================================================
# 10. COSTRUZIONE DEI LAYER (+ EQUIVOLUME)
# =====================================================================================

echo "=== [10] Building equidistant and quivolumetric layers ==="

for contrast in "${SPMT[@]}"; do
for region in "${ROIS[@]}"; do
echo "Processing Layers per: $contrast - $region"
RIM_FINAL="${ROI_DIR}/${SUB}_con-${contrast}_desc-polishCleaned_label-${region}_res-up_dseg.nii.gz"
PREFIX="${SUB}_con-${contrast}_label-${region}"
LN2_LAYERS \
-rim "${RIM_FINAL}" \
-equivol \
-nr_layers "${NR_LAYERS}" \
-curvature \
-output "${WD}/${PREFIX}"
done
done







	  
  
