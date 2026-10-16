#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fMRI Preprocessing Pipeline (BIDS-compliant)

Pipeline aggiornata che usa fmri_preprocessing.py:
- Rimuove lo step di rimozione dummies
- Tiene traccia di tutti gli step nel JSON finale
- Salva output in cartelle derivatives BIDS-compliant
"""

import os
import sys
import shutil
import json
from datetime import datetime
from bids import BIDSLayout 
from preprocessing_scripts import fmri_preprocessing as fmri
import myutils
import nibabel as nib
import subprocess


FWHM = None
#SCRIPT_PATH=os.path.dirname(__file__)
BIDS_ROOT = "/mnt/store2/projects/progetto_layer/data"
PIPELINE_NAME = "hires_pipeline"
TPATTERN = 'interleaved'  # Pattern di acquisizione delle slice
DERIVATIVES_ROOT = os.path.join(BIDS_ROOT, 'derivatives', PIPELINE_NAME)

# ---------------- Setup BIDS Derivatives ----------------
# BIDS richiede un dataset_description.json nella root della cartella derivatives
def setup_bids_derivatives(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    desc_file = os.path.join(dest_dir, "dataset_description.json")
    if not os.path.exists(desc_file):
        description = {
            "Name": "Custom anatomical and fMRI Preprocessing Pipeline",
            "BIDSVersion": "1.4.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": PIPELINE_NAME,
                "Version": "1.0",
                "CodeURL": "https://github.com/alicegiubergia/project_fmri_hires"
            }]
        }
        with open(desc_file, 'w') as f:
            json.dump(description, f, indent=4)

setup_bids_derivatives(DERIVATIVES_ROOT)

# ---------------- Setup BIDS ----------------
   
layout = BIDSLayout(BIDS_ROOT, validate=True, derivatives=True)

subjects = ['007', '008'] # Specifica qui i soggetti da processare
# subjects = layout.get_subjects()  # Uncomment to process all subjects
#patterns = ['sub-{subject}/func/sub-{subject}_run-{n_run}_desc-{desc}_space-{space}_{suffix<bold|epi>}.{extension<nii|nii.gz|json>|nii.gz}']

# ---------------- Funzioni log ----------------
PROCESSING_LOG = []

def log_step(name, input_file, output_file, params=None):
    PROCESSING_LOG.append({
        "StepName": name,
        "InputFile": os.path.basename(input_file) if input_file else None,
        "OutputFile": os.path.basename(output_file) if output_file else None,
        "Parameters": params if params else {},
        "Timestamp": datetime.now().isoformat(timespec='seconds')
    })
    
# ----------------- Pipeline -----------------

for subj in subjects: 
    print(f'sub-{subj}')
    subj_dir = os.path.join(DERIVATIVES_ROOT, f"sub-{subj}", "func")
    os.makedirs(subj_dir, exist_ok=True)
    
    # Get files for processing
    runs = layout.get_runs(subject=subj, task='reading')
    iPE_files = layout.get(subject=subj, suffix='epi', extension='nii.gz', datatype='fmap', return_type='file')
    anat_file = layout.get(subject=subj, datatype='anat', suffix='T1w', desc='N4', extension='.nii.gz', return_type='filename')[0]
                           
    for n_run in runs:
        print(f'|--run-{n_run}')

        # Get epi file for processing
        epi_file = layout.get(subject=subj, suffix='bold', extension='nii.gz', 
                            datatype='func',
                            part='mag',
                            task='reading',
                            run = n_run,
                            return_type='file')[0]
        
        try:
            real_file = layout.get(subject=subj, suffix='bold', extension='nii.gz', 
                                datatype='func',
                                part='real',
                                task='reading',
                                run = n_run,
                                return_type='file')[0]
            imag_file = layout.get(subject=subj, suffix='bold', extension='nii.gz', 
                                datatype='func',
                                part='imag',
                                task='reading',
                                run = n_run,
                                return_type='file')[0]
        except IndexError:
            real_file = ''
            imag_file = ''
        
        
        # trova il fieldmap che ha questo BOLD in IntendedFor
        fmap_file_for_run = None
        for fmap in iPE_files:
            json_file = fmap.replace('.nii.gz', '.json')
            with open(json_file, 'r') as f:
                data = json.load(f)
            intended_for = data.get('intendedFor', [])
            intended_for_full = [os.path.join(BIDS_ROOT, p.replace("bids::", "")) for p in intended_for]

            if epi_file in intended_for_full:
                fmap_file_for_run = fmap
                break
        
        if fmap_file_for_run is None:
            print(f"⚠️ Nessuna fieldmap trovata per run {n_run}")
        else:
            print(f"    Run {n_run} -> Fieldmap {os.path.basename(fmap_file_for_run)} ") 

        # Costruisco output filename
        entities = {'subject': subj, 
                    'task': 'reading',
                    'run': n_run,
                    'desc': 'preproc',
                    'suffix':'bold',
                    'extension':'.nii.gz',
                    'space':'anat' }
        output_fn = layout.build_path(entities, validate=False, strict=True, absolute_paths=False).split('/')[-1]
        output_path = os.path.join(subj_dir, output_fn)

        # Directory di lavoro temporanea (per non sporcare la cartella finale)
        work_dir = os.path.join(subj_dir, f"tmp_run-{n_run}")
        os.makedirs(work_dir, exist_ok=True)


        LOGFILE = os.path.join(subj_dir, f"sub-{subj}_run-{n_run}_preprocessing.log")
        myutils.set_log_file(LOGFILE)



        # Backup del file originale
        orig_file = f"{work_dir}/orig_sub-{subj}_run-{n_run}.nii.gz"
        shutil.copy(epi_file, orig_file)

        base_filename = orig_file.replace("orig_", "").split('.nii')[0]
        myutils.log_message(f"Backup creato: {orig_file}")

        # ---------------- Step 1: NORDIC ----------------
        myutils.log_message("--------------------- Step 1")
        myutils.log_message("Run NORDIC denoising")
        nordic_prefix = f"{base_filename}_nordic"
        #fmri.run_NORDIC(real_file, imag_file, epi_file, f'{nordic_prefix}.nii.gz')
        current_basename = nordic_prefix
        current_file = f"{current_basename}.nii.gz"
        log_step("NORDIC", base_filename, nordic_prefix)

        # ---------------- Step 2: Slice Timing ----------------
        myutils.log_message("--------------------- Step 2")
        myutils.log_message("slicetimer")
        params = fmri.get_seq_params(epi_file)
        st_prefix = f"{current_basename}_st"
        multi_band = layout.get_metadata(epi_file).get('MultibandAccelerationFactor', 1)
        n_slices = nib.load(epi_file).shape[2]
        #fmri.run_slicetimer(current_file, st_prefix, n_slices, multi_band, tpattern=TPATTERN)
        log_step("Slice Timing", current_basename, st_prefix, params={'n_slices': n_slices, 'multi_band': multi_band, 'tpattern': TPATTERN})
        current_basename = st_prefix
        current_file = f"{current_basename}.nii.gz"

        # ---------------- Step 3: Distortion Correction ----------------
        myutils.log_message("--------------------- Step 3")
        myutils.log_message("Distortion field correction (AFNI)")

        afni_corrected_prefix = f"{current_basename}_afni"
        #fmri.run_afni_3dQWarp(current_file, fmap_file_for_run, afni_corrected_prefix)
        log_step("Distortion Correction AFNI", current_basename, afni_corrected_prefix, params={'inverse PE': os.path.basename(fmap_file_for_run)})
        current_basename = afni_corrected_prefix
        current_file = f"{current_basename}.nii.gz"

        # ---------------- Step 4: Motion Correction ----------------
        myutils.log_message("--------------------- Step 4")
        myutils.log_message("Motion Correction")
        mcflirt_prefix = f"{current_basename}_mc"
        #fmri.run_moco(current_file, mcflirt_prefix)
        log_step("Motion Correction", current_basename, mcflirt_prefix)
        current_basename = mcflirt_prefix
        current_file = f"{current_basename}.nii.gz"

        # Applica Grand Mean Scaling intensity normalization
        myutils.log_message("--------------------- Step 5")
        myutils.log_message("Grand Mean Scaling intensity normalization")
        norm_prefix = f"{current_basename}_norm"
        #fmri.GMS_norm(current_file, norm_prefix)
        log_step("Intensity Normalization", current_basename, norm_prefix, params={'method': 'grand mean scaling', 'target mean': 100})
        current_basename = norm_prefix
        current_file = f"{current_basename}.nii.gz"

        # ---------------- Step 6: Smoothing ----------------
        if FWHM is not None:
            myutils.log_message("--------------------- Step 6")
            myutils.log_message("Smoothing")
            smooth_prefix = f"{current_basename}_smoothed"
            fmri.smooth(current_file, FWHM, smooth_prefix)
            log_step("Smoothing", current_basename, smooth_prefix, {"FWHM": FWHM})
            current_basename = smooth_prefix
            current_file = f"{current_basename}.nii.gz"


        # Applica fmri_filter.py per filtraggio temporale
        #log_message("--------------------- Step 5")
        #log_message("Filtering")
        #final_filename = f"{base_filename}_st_topup_mc_filtered.nii.gz"
        #final_file = os.path.join(output_dir, final_filename)
        #fmri_filter_command = f"python fmri_filter.py -epi {mcflirt_file} -mas -TR {TR} -out {final_file}"
        #subprocess.run(fmri_filter_command, shell=True, check=True)
        #log_message(fmri_filter_command)

        # ---------------- Step 7: Registration to T1 ----------------
        myutils.log_message("--------------------- Step 7")
        myutils.log_message("Registration to anatomical")
        epi_in_anat_space = f"{current_basename}_2anat"
        anat_in_epi_space = os.path.join(work_dir, f"sub-{subj}_space-epi_desc-resampled_T1w.nii.gz")
        mtx = fmri.normalize_to_struct(current_file, f"{mcflirt_prefix}.nii.gz", anat_file, f"{epi_in_anat_space}.nii.gz", anat_in_epi_space) 
        log_step("Registration to T1", anat_file + "," + current_file, anat_in_epi_space + "," + epi_in_anat_space, {'script': 'code/epiSlabxT1.sh', "moving": anat_file, "fixed": f"{mcflirt_prefix}_mean_reg_N4_reg.nii.gz", 'mat': mtx})

        current_basename = epi_in_anat_space
        current_file = f"{current_basename}.nii.gz"

        # Il file funzionale finale rimane nello spazio NATIVO (senza Step 7 applicato)
        choosen_output = current_file
        shutil.copy(choosen_output, output_path)
        #  Salviamo la T1 coregistrata (utile per controllare la qualità a occhio)
        bids_anat_ref = os.path.join(subj_dir, f"sub-{subj}_task-reading_run-{n_run}_T1w.nii.gz")
        #shutil.copy(anat_in_epi_space, bids_anat_ref)

        # Salviamo il file .par con i parametri di movimento (se esiste)
        mc_par_file = f"{mcflirt_prefix}.par"
        if os.path.exists(mc_par_file):
            dest_par_file = os.path.join(subj_dir, f"sub-{subj}_task-reading_run-{n_run}_motion.par")
            shutil.copy(mc_par_file, dest_par_file)
            myutils.log_message(f"Motion parameters salvati: {dest_par_file}")

        myutils.log_message("=== Preprocessing completato con successo! ===")

        # --- GENERAZIONE JSON SIDECAR ---
        sidecar_json = output_path.replace('.nii.gz', '.json')

        final_report = {
            "PipelineDescription": {
                "Name": "Custom fMRI Preprocessing",
                "Version": "1.0",
                "CodeURL": "https://github.com/alicegiubergia/project_fmri_hires"
            },
            "SubjectID": subj,
            "RawSources": [epi_file],
            "ProcessingSteps": PROCESSING_LOG,
            "SoftwareVersions": {
            "Python": sys.version,
            "Conda environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "FSL": open(os.path.join(os.environ["FSLDIR"],"etc","fslversion")).read().strip(),
            "AFNI": subprocess.check_output(["afni", "-version"], text=True).strip()
            },
        }
        with open(sidecar_json, "w") as f:
            json.dump(final_report, f, indent=4, default=str)
        myutils.log_message(f"JSON report salvato: {sidecar_json}")
        print(f"[sub-{subj}] ✅ Preprocessing completato!")

        # Pulizia
        # shutil.rmtree(work_dir) 
        PROCESSING_LOG.clear()


