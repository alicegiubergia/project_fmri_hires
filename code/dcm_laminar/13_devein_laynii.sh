#!/usr/bin/env bash
# =============================================================================
# 13_devein_laynii.sh — deveining delle serie temporali con LayNii
#
# I file di layering sono separati per ROI e per set di ROI (contrasto):
#   <sub>_con-<CON>_label-<roi>_metric_<LAYERING>.nii  (roi = dorsal | ventral | visual; LAYERING = equidist | equivol)
#   CON è ricavato da TASK tramite il dizionario CON_OF (mirroring -> spmT_0008,
#   spacing -> spmT_0009): in configurazione si scrive solo il nome del task.
# quindi ogni ROI viene processata per conto suo, con il suo crop, le sue
# colonne e la sua ALF.
#
# I BOLD sono sulla griglia nativa, i layer su quella upsampled: ogni 4D viene
# ricampionato (antsApplyTransforms -e 3) direttamente sul crop della ROI nella
# griglia dei layer, così non si crea mai il 4D upsampled dell'intero volume.
#
# Per soggetto, per ogni ROI (dorsal, ventral, visual):
#   1. crop dei file di layering al bounding box della ROI (metric > 0, dilatata)
#   2. colonne con LN2_COLUMNS
#   3. ALF (proxy del CBV venoso): calcolata UNA volta sui residui nativi del GLM
#      e poi ricampionata come mappa 3D sul crop (ALF_MODE=residuals, vedi
#      00_write_residuals.m) oppure uniforme (ALF_MODE=uniform)
#   Per run:
#   4. ricampionamento del BOLD sul crop + percent signal change
#   5. LN2_DEVEIN in modalità deconvoluzione (Markuerkiaga et al. 2016)
# Anche la visiva viene deveinata; nel DCM resta un nodo unico (media dei layer deveinati).
#
# Output in ${ROOT}/<sub>/dcm_devein/<TASK>/ , tutti .nii (per SPM):
#   Dors_run-<r>_devein.nii, Dors_metric.nii.gz   (dorsale)
#   Ven_run-<r>_devein.nii,  Ven_metric.nii.gz    (ventrale)
#   Vis_run-<r>_devein.nii,  Vis_metric.nii.gz    (visiva)
#
# Requisiti: LayNii (LN2_COLUMNS, LN2_DEVEIN), FSL e ANTs nel PATH; python3 con nibabel.
# =============================================================================
set -euo pipefail
export FSLOUTPUTTYPE=NIFTI_GZ          # output FSL compressi (li legge SPM direttamente)

# --------------------------- CONFIGURAZIONE ----------------------------------
ROOT="/mnt/store1/projects/progetto_layer/data/derivatives"  
SUBJECTS=(sub-002)               
RUNS=(1 2 3)
TASK="baseline"                                # mirroring | spacing — set di ROI (uno per tutti i modelli)
# Numero di colonne LN2_COLUMNS: letto per soggetto x set di ROI x ROI dalla
# tabella di n_columns.py (colonna n_columns). Per forzare un numero fisso per
# tutte le ROI, scrivi un intero in N_COLUMNS_FIXED (vuoto = usa la tabella).
COL_DIAM="2"                                   # diametro (mm) con cui è stata generata la tabella: n_columns.py area --diam <COL_DIAM>
N_COLUMNS_FIXED=""
declare -A CON_OF=( [mirroring]="spmT_0008" [spacing]="spmT_0009" [baseline]="spmT_0001")
[[ -n "${CON_OF[${TASK}]+x}" ]] || { echo "TASK sconosciuto: ${TASK} (usa: ${!CON_OF[*]})"; exit 1; }
CON="${CON_OF[${TASK}]}"
LAYERING="equidist"                            # equidist | equivol — usa lo stesso delle altre analisi e della definizione dei bin nel DCM
TR="3.0"                                       # TR in secondi
LAMBDA="0.25"                                  # default LayNii (7T, Markuerkiaga 2016); ~+20% se 3T
ALF_MODE="residuals"                           # residuals (ALF dai residui GLM) | uniform (ALF costante: stesso CBV venoso in ogni layer)
INTERP="Linear"                                # stessa interpolazione usata per upsamplare le mappe spmT (es. Linear, BSpline[3], LanczosWindowedSinc)
TRANSFORMS=(-t identity)                       # identity se i layer sono nello spazio del BOLD solo upsampled; altrimenti le tue trasformazioni ANTs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Percorsi — $1 = soggetto, $2 = ROI (dorsal|ventral|visual), $3 = suffisso LayNii
# es.: lay sub-001 dorsal metric_equidist -> .../sub-001_con-spmT_0008_label-dorsal_metric_equidist.nii
lay()        { echo "${ROOT}/layer_fmri/$1/$1_con-${CON}_label-$2_$3.nii"; }
metric_f()   { lay "$1" "$2" metric_${LAYERING}; }
layers_f()   { lay "$1" "$2" layers_${LAYERING}; }   # layer DISCRETI interi (1 = lato WM), output LN2_LAYERS
midgm_f()    { lay "$1" "$2" midGM_${LAYERING}; }    # output LN2_LAYERS
# rim (dseg) in una cartella diversa, input di LN2_LAYERS (1=CSF, 2=WM, 3=GM)
rim_f()      { echo "${ROOT}/rois/$1/$1_con-${CON}_desc-polishCleaned_label-$2_res-up_dseg.nii.gz"; } 
func_run()   { local r; r=$(printf "%02d" "$((10#$2))"); echo "${ROOT}/hires_pipeline/$1/func/$1_task-reading_run-${r}_space-anat_desc-preproc_bold.nii"; }  # 4D preprocessato, griglia NATIVA (quello entrato nel GLM)
alf_source() { echo "${ROOT}/hires_pipeline/$1/stats-masked/$1_residuals_4d.nii.gz"; }      # output di 00_write_residuals.m = <cartella dello SPM.mat>/<sub>_residuals_4d.nii
COLS_TSV="${ROOT}/rois/n_columns_d${COL_DIAM}mm_${LAYERING}.tsv"                      # output di: python n_columns.py area --diam ${COL_DIAM}
# etichetta nel nome del deveinato: N fisso -> desc-<N>col, N dalla tabella -> desc-d<COL_DIAM>mm
if [[ -n "${N_COLUMNS_FIXED}" ]]; then COL_TAG="${N_COLUMNS_FIXED}col"; else COL_TAG="d${COL_DIAM}mm"; fi
# -----------------------------------------------------------------------------

# n_cols <sub> <set di ROI> <roi> -> numero di colonne (dalla tabella, o N_COLUMNS_FIXED)
n_cols() {
  if [[ -n "${N_COLUMNS_FIXED}" ]]; then echo "${N_COLUMNS_FIXED}"; return; fi
  local n
  n=$(awk -F'\t' -v s="$1" -v t="$2" -v r="$3" \
        'NR==1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
         $c["subject"] == s && $c["roi_set"] == t && $c["roi"] == r { print $c["n_columns"]; exit }' \
        "${COLS_TSV}")
  [[ "${n}" =~ ^[1-9][0-9]*$ ]] || { echo "!! numero di colonne non trovato o non valido per $1 $2 $3 in ${COLS_TSV}: '${n}'" >&2; return 1; }
  echo "${n}"; }

# sposta l'output di LayNii (che aggiunge suffissi al nome) sul nome atteso
grab() { local pat="$1" dst="$2" f; f=$(ls ${pat} 2>/dev/null | head -n1)
         [[ -z "${f}" ]] && { echo "!! output non trovato: ${pat}"; }
         if [[ "${f}" == *.gz ]]; then gunzip -c "${f}" > "${dst}"; rm -f "${f}"; else mv "${f}" "${dst}"; fi; }

# to_layer_grid <4D nativo> <riferimento 3D (crop nella griglia dei layer)> <output>
to_layer_grid() {
  antsApplyTransforms -d 3 -e 3 -i "$1" -r "$2" -o "$3" -n "${INTERP}" "${TRANSFORMS[@]}" --float 1; }

bbox() {  # bbox <file metric> -> "xmin xsize ymin ysize zmin zsize" della ROI (metric>0) dilatata
  local tmp="$2/tmp_bbox"
  fslmaths "$1" -thr 0 -bin -dilM -dilM "${tmp}"
  local w; w=($(fslstats "${tmp}.nii.gz" -w)); rm -f "${tmp}.nii.gz"
  echo "${w[0]} ${w[1]} ${w[2]} ${w[3]} ${w[4]} ${w[5]}"; }

if [[ -z "${N_COLUMNS_FIXED}" && ! -f "${COLS_TSV}" ]]; then
  echo "Tabella delle colonne non trovata: ${COLS_TSV}"
  echo "Generala con: python n_columns.py area --diam ${COL_DIAM}  (oppure imposta N_COLUMNS_FIXED)"; exit 1
fi

for SUB in "${SUBJECTS[@]}"; do
  echo "=== ${SUB} (${TASK} = ${CON}) ==="
  OUT="${ROOT}/roi-deveined/${SUB}/${TASK}"
  mkdir -p "${OUT}"

  # ---------------------- deveining per ROI ---------------------------------
  declare -A PREFIX=( [dorsal]=Dors [ventral]=Ven [visual]=Vis )
  for ROI in visual ventral dorsal; do
    P="${PREFIX[${ROI}]}"
    N_COLUMNS=$(n_cols "${SUB}" "${TASK}" "${ROI}")      # diverso per ogni ROI

    # 1. crop
    # solo se manca almeno uno dei quattro file; in quel caso si rifanno
    # tutti e quattro, così restano sempre sullo stesso box
    CROP_OK=1
    for F in metric layers midgm rim; do
      [[ -f "${OUT}/${P}_${F}.nii.gz" ]] || CROP_OK=0
    done
    if (( CROP_OK )); then
      echo "     crop già presente, salto"
    else
      BOX=$(bbox "$(metric_f "${SUB}" ${ROI})" "${OUT}")
      fslroi "$(metric_f "${SUB}" ${ROI})" "${OUT}/${P}_metric" ${BOX}
      fslroi "$(layers_f "${SUB}" ${ROI})" "${OUT}/${P}_layers" ${BOX}
      fslroi "$(midgm_f  "${SUB}" ${ROI})" "${OUT}/${P}_midgm"  ${BOX}
      fslroi "$(rim_f    "${SUB}" ${ROI})" "${OUT}/${P}_rim"    ${BOX}
    fi

    # 2. colonne
    if [[ ! -f "${OUT}/${P}_columns${N_COLUMNS}.nii" ]]; then
      LN2_COLUMNS -rim "${OUT}/${P}_rim.nii.gz" -midgm "${OUT}/${P}_midgm.nii.gz" \
                  -nr_columns ${N_COLUMNS} -output "${OUT}/${P}_cols.nii.gz"
      grab "${OUT}/${P}_cols_columns${N_COLUMNS}.nii.gz" "${OUT}/${P}_columns${N_COLUMNS}.nii" # gunzip for SPM
      rm -f "${OUT}/${P}"_cols*centroids*.nii*
    fi

    # 3. ALF
    if [[ ! -f "${OUT}/${P}_alf.nii.gz" ]]; then
      if [[ "${ALF_MODE}" == "residuals" ]]; then
        # ALF sui residui NATIVI (una volta per soggetto): niente 4D dei residui sul box
        if [[ ! -f "${OUT}/alf_native.nii.gz" ]]; then
          python3 "${SCRIPT_DIR}/psc_alf.py" alf "$(alf_source "${SUB}")" "${OUT}/alf_native.nii.gz" --tr "${TR}"
        fi
        antsApplyTransforms -d 3 -i "${OUT}/alf_native.nii.gz" -r "${OUT}/${P}_metric.nii.gz" \
                            -o "${OUT}/${P}_alf.nii.gz" -n "${INTERP}" "${TRANSFORMS[@]}" --float 1
        fslmaths "${OUT}/${P}_alf" -mas "${OUT}/${P}_layers" "${OUT}/${P}_alf"
      else
        fslmaths "${OUT}/${P}_layers" -bin "${OUT}/${P}_alf"     # 1 dentro la corteccia
      fi
    fi

    for RUN in "${RUNS[@]}"; do
      echo "     run ${RUN}"
      # 4. ricampionamento sul crop della ROI + PSC
      # (il PSC non dipende dalle colonne: si calcola una volta sola)
      if [[ ! -f "${OUT}/${P}_run-$(printf '%02d' "$RUN")_psc.nii" ]]; then
      to_layer_grid "$(func_run "${SUB}" "${RUN}")" "${OUT}/${P}_metric.nii.gz" "${OUT}/tmp_run.nii.gz"
      python3 "${SCRIPT_DIR}/psc_alf.py" psc "${OUT}/tmp_run.nii.gz" "${OUT}/${P}_run-$(printf '%02d' "$RUN")_psc.nii" \
              --mask "${OUT}/${P}_layers.nii.gz"
      rm -f "${OUT}/tmp_run.nii.gz"
      fi

      # 5. deveining
      LN2_DEVEIN -input "${OUT}/${P}_run-$(printf '%02d' "$RUN")_psc.nii" \
                 -layer_file "${OUT}/${P}_layers.nii.gz" \
                 -column_file "${OUT}/${P}_columns${N_COLUMNS}.nii" \
                 -ALF "${OUT}/${P}_alf.nii.gz" \
                 -lambda "${LAMBDA}" \
                 -output "${OUT}/${P}_run-$(printf '%02d' "$RUN")_columns${N_COLUMNS}"
      grab "${OUT}/${P}_run-$(printf '%02d' "$RUN")_columns${N_COLUMNS}_deveinDeconv*.nii*" "${OUT}/${P}_run-$(printf '%02d' "$RUN")_desc-${COL_TAG}_devein.nii"
    done
  done

done
echo "Fatto. Prima del DCM controlla il profilo laminare della baseline prima/dopo il deveining."
