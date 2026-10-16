#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anatomical T1 Preprocessing Pipeline

Questo script esegue il preprocessing di un file anatomico. 
Gli step includono:
1. Backup del file originale.
2. Brain extraction con ANTs.
3. Correzione del bias field.
4. Normalizzazione dell'intensità dell'immagine (0-100)
5. Registrazione al template (MNI152) con ANTs.
6. Salvataggio dei file in cartella derivatives BIDS-compliant.
7. Salvataggio JSON report con tutti gli step e versioni software

Autore: Alice Giubergia
Data: 2025-03-17
"""

import os
import json
import nibabel as nib
import numpy as np
import ants
import antspynet
import subprocess
from datetime import datetime
import shutil
from utils import myutils
import argparse
from templateflow import api as tflow
from nipype.interfaces import spm

# Log globale degli step per il JSON finale
PROCESSING_LOG = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def log_step(name, input_file, output_file, params=None):
    """Aggiunge un record di step di preprocessing al log globale."""
    PROCESSING_LOG.append({
        "StepName": name,
        "InputFile": os.path.basename(input_file) if input_file else None,
        "OutputFile": os.path.basename(output_file) if output_file else None,
        "Parameters": params if params else {},
        "Timestamp": datetime.now().isoformat(timespec='seconds')
    })

# Funzione per caricare un'immagine NIfTI
def load_nifti(file_path):
    """Carica un file NIfTI e restituisce l'array e l'header."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header

# Funzione per salvare un'immagine NIfTI
def save_nifti(data, affine, header, output_path):
    """Salva un array come file NIfTI mantenendo l'affine e l'header."""
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, output_path)

def apply_brain_mask(input_image_path, input_mask_path, output_brain_path):
    """
    Maschera la input image.
    """
    if not os.path.isfile(output_brain_path):
        data, affine, header = load_nifti(input_image_path)
        mask, _, _ = load_nifti(input_mask_path)
        new_data = data * mask
        save_nifti(new_data, affine, header, output_brain_path)
    myutils.log_message(f"output: {output_brain_path}")

def run_ants_brain_extraction(t1_path, output_prefix, template_path, template_mask_path):
    """
    Esegue brain extraction con ANTs:
      1. antsBrainExtraction.sh (standard)
      2. antspynet brain extraction (FreeSurfer-flavored e Combined)
    """ 
    # --- Step 1: ANTs BrainExtraction.sh ---
    command = f"antsBrainExtraction.sh -d 3 -a {t1_path} -m {template_mask_path} -e {template_path} -o {output_prefix}"
    myutils.log_message(command)

    if not os.path.isfile(f"{output_prefix}BrainExtractionMask.nii.gz"):
        subprocess.run(command, shell=True, check=True)


    # --- Step 2: ANTsPyNet-based brain extraction ---
    t1 = ants.image_read(t1_path)
    # FreeSurfer-flavored
    seg_fs = antspynet.brain_extraction(t1, modality="t1nobrainer", verbose=True)
    fs_out = f"{output_prefix}seg_FS_flavored.nii.gz"
    ants.image_write(seg_fs, fs_out)
    myutils.log_message(f"ANTsPyNet FreeSurfer-flavored brain extraction output: {fs_out}")
        
    # Combined
    seg_comb = antspynet.brain_extraction(t1, modality="t1combined", verbose=True)
    comb_out = f"{output_prefix}seg_Combined.nii.gz"
    ants.image_write(seg_comb, comb_out)
    myutils.log_message(f"Saved: {comb_out}")

    return {
        "standard_mask": f"{output_prefix}BrainExtractionMask.nii.gz",
        "fs_mask": fs_out,
        "combined_mask": comb_out
        }

def run_synthstrip(t1_path, output_prefix):
    """
    SynthStrip di FreeSurfer — molto robusto, basato su deep learning.
    Usa il container Docker: cesne/freesurfer:8.0
    """    
    uid = os.getuid()
    gid = os.getgid()
    t1_path = os.path.abspath(t1_path)
    output_prefix = os.path.abspath(output_prefix)
    t1_dir = os.path.dirname(t1_path)
    t1_fn = os.path.basename(t1_path)
    output_dir = os.path.dirname(output_prefix)
    output_fn = os.path.basename(output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--user", f"{uid}:{gid}",
        "-v", f"{t1_dir}:/data",
        "-v", f"{output_dir}:/output",
        "cesne/freesurfer:8.0", "mri_synthstrip",
        "-i", f"/data/{t1_fn}",
        "-m", f"/output/{output_fn}mask.nii.gz",
        "-o", f"/output/{output_fn}brain.nii.gz",
        "--no-csf"
    ]
    cmd_string = " ".join(cmd)
    subprocess.run(cmd_string, shell=True, check=True)
    print("SynthStrip completato!")
    return os.path.join(output_dir, f"{output_fn}mask.nii.gz")

# Funzione per la correzione del bias field con N4 (ants)
def n4_bias_correction(input_image_path, input_mask_path, output_image_path):
    """
    Applica la correzione del bias field N4.
    """
    if not os.path.isfile(output_image_path):
        input_image = ants.image_read(input_image_path)
        if input_mask_path is not None:
            input_mask = ants.image_read(input_mask_path)
        else:
            input_mask = None

        corrected_img = ants.n4_bias_field_correction(input_image, mask=input_mask, shrink_factor=4, rescale_intensities=True)
        ants.image_write(corrected_img, output_image_path)
    myutils.log_message(f"{input_image_path} -> {output_image_path}")

# Funzione per la correzione del bias field con N4 (SPM)
def bias_correction_spm(input_image_path, output_image_path, mode=None):
    """
    Applica la correzione del bias field N4 implementata in SPM.
    """
    current_dir = os.path.dirname(input_image_path)
    work_dir = os.path.join(current_dir, 'biasCorrection_spm')
    if mode is None:
        mode_string = ''
    else:
        mode_string=mode
    
    if not os.path.isfile(f"{output_image_path}"):
        os.makedirs(work_dir, exist_ok=True)
        shutil.copy(input_image_path, os.path.join(work_dir, f'{mode_string}_uncorr.nii.gz'))
        os.system(f"gunzip {os.path.join(work_dir, f'{mode_string}_uncorr.nii.gz')}")
        matlab_command = (
            f'matlab -nodisplay -nosplash -nodesktop -r '
            f'"addpath(\'{SCRIPT_DIR}\'); '
            f'Bias_field_spm(\'{os.path.join(work_dir, f"{mode_string}_uncorr.nii")}\'); exit"' 
            )
        myutils.log_message(matlab_command)
        subprocess.run(matlab_command, shell=True, check=True)
        os.system(f"3dcalc -a {os.path.join(work_dir, f'm{mode_string}_uncorr.nii')} -overwrite -expr 'a' -datum short")
        os.system(f"gzip {os.path.join(work_dir, f'm{mode_string}_uncorr.nii')}")
        shutil.copy(os.path.join(work_dir, f'm{mode_string}_uncorr.nii.gz'), output_image_path)

    myutils.log_message(f"{input_image_path} -> {output_image_path}")

# Funzione per normalizzare l'intensità tra 0 e 100
def normalize_intensity(input_image_path, mask_image_path, output_image_path):
    """
    Normalizza l'intensità dell'immagine tra 0 e 100.
    """
    data, affine, header = load_nifti(input_image_path)
    mask, _, _ = load_nifti(mask_image_path)
    mask = mask.astype(bool)
    mean_val = np.mean(data[mask])
    normalized_data = data / mean_val * 100

    save_nifti(normalized_data, affine, header, output_image_path)
    myutils.log_message(f"{input_image_path} -> {output_image_path}")
    return output_image_path

def histogram_matching(input_image_path, template_path):
    """
    Normalizza l'intensità mediante histogram matching con il template.
    """    
    ref = ants.image_read(template_path)
    img = ants.image_read(input_image_path)
    img_matched = ants.histogram_match_image(img, ref)
    fname = os.path.basename(input_image_path).split('.nii')[0]
    dirname = os.path.dirname(input_image_path)
    ants.image_write(img_matched, input_image_path)
    myutils.log_message(f"{input_image_path} -> {input_image_path}")

    return f"{dirname}/{fname}_normalized.nii.gz"

# Funzione per normalizzare l'immagine anatomica con ANTs (registrazione all'MNI)
def normalize_to_mni(moving_path, fixed_path, output_dir, sub_id, transform_type='Rigid'):
    """
    Normalizza un'immagine T1w nello spazio MNI usando ANTs.
    Parametri:
      - moving_path: path della T1 nativa.
      - fixed_path: path del template MNI.
      - output_dir: directory di output.
      - sub_id: es. '01'
      - transform_type: 'Rigid' oppure 'SyN'
    
    Output:
      - Immagine normalizzata
      - File di trasformazione diretta e inversa
    """
    assert transform_type in ["Rigid", "SyN"], "transform_type deve essere 'Rigid' o 'SyN'"

    desc = "rigid" if transform_type == "Rigid" else "syn"
    warped_output = os.path.join(
        output_dir,
        f"sub-{sub_id}_space-MNI_desc-{desc}_T1w.nii.gz"
    )

    prefix = os.path.join(output_dir, f"sub-{sub_id}_from-T1w_to-MNI_{transform_type}")

    if not os.path.isfile(warped_output):
        moving_img = ants.image_read(moving_path)
        fixed_img = ants.image_read(fixed_path)
        myutils.log_message(f"--- Starting {transform_type} registration to MNI ---")

        # 1. Rigid registration
        tx = ants.registration(fixed=fixed_img, 
                                         moving=moving_img, 
                                         type_of_transform=transform_type, 
                                         outprefix=prefix,
                                         verbose=True)
        
        # 2. Creo reference con risoluzione originale
        # Template MNI è ad alta risoluzione, ma voglio mantenere la risoluzione originale della T1
        # Creo un'immagine di riferimento con la stessa risoluzione della T1 ma nello spazio MNI
        resampled_img = ants.resample_image(fixed_img,
                                            moving_img.spacing,
                                            use_voxels=False,
                                            interp_type=0)
        if transform_type == "SyN":
            ref_img = fixed_img
        elif transform_type == "Rigid":
            ref_img = resampled_img

        # 3. Applico la trasformazione al moving_img usando il reference creato                                                                                                                    moving_img)
        warped_img = ants.apply_transforms(fixed=ref_img,
                                            moving=moving_img,
                                            transformlist=tx['fwdtransforms'],
                                            interpolator='linear')
        
        # 4. Salvo l'immagine nello spazio MNI con la risoluzione originale
        ants.image_write(warped_img, warped_output)
     
        myutils.log_message(f"Registrazione completata ({transform_type})")
        myutils.log_message(f"  Warped: {warped_output}")
    else:
        myutils.log_message(f"Registrazione {transform_type} già presente: {warped_output}")

    return warped_output


# Main script
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="T1 preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Path al file T1w.nii.gz NIfTI di input")
    parser.add_argument("--sub", required=True, help="Subject ID (es. '01')")
    parser.add_argument("--output_dir", default="../data/derivatives/mypipeline", help="Output folder (default: derivatives/mypipeline/sub-XX/anat)")
    args = parser.parse_args()
    
    sub_id = args.sub
    INPUT_NIFTI = args.input
 
    # Cartella BIDS-compliant per i risultati
    OUTPUT_DIR = os.path.join(args.output_dir, f"sub-{sub_id}", "anat")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Log
    LOG_FILE = os.path.join(OUTPUT_DIR, "preprocess_t1.log")
    myutils.set_log_file(LOG_FILE)

    myutils.log_message("=== Configurazione TemplateFlow ===")   
    print("Recupero template MNI da TemplateFlow (se non presenti verranno scaricati)...")

    # Recupera i template (tflow restituisce una stringa/Path o una lista di essi)
    t1w_res = tflow.get("MNI152NLin6Asym", resolution=1, desc=None, suffix="T1w", extension=".nii.gz")
    mask_res = tflow.get("MNI152NLin6Asym", resolution=1, desc="brain", suffix="mask", extension=".nii.gz")
    
    # Assicuriamoci di estrarre la stringa corretta nel caso restituisca una lista
    TEMPLATE_MNI_HR = str(t1w_res[0] if isinstance(t1w_res, list) else t1w_res)
    TEMPLATE_MASK = str(mask_res[0] if isinstance(mask_res, list) else mask_res)
    
    myutils.log_message(f"Template T1w: {TEMPLATE_MNI_HR}")
    myutils.log_message(f"Template Mask: {TEMPLATE_MASK}")

    ORIG_FILE = f"{OUTPUT_DIR}/sub-{sub_id}_desc-orig_T1w.nii.gz"
    shutil.copy(INPUT_NIFTI, ORIG_FILE)
    print(f"[sub-{sub_id}] Copying T1 image for backup ...")

    myutils.log_message("=== Inizio Preprocessing ===")

    # Step 0: move to MNI space
    print(f"[sub-{sub_id}] Moving to MNI space...")
    myutils.log_message("--------------------- Step 0")
    myutils.log_message("Moving to MNI space")
    CURRENT_FILE = normalize_to_mni(moving_path=INPUT_NIFTI, 
                                    fixed_path=TEMPLATE_MNI_HR, 
                                    output_dir=OUTPUT_DIR, 
                                    sub_id=sub_id, transform_type='Rigid')

    log_step("Move to MNI space", INPUT_NIFTI, CURRENT_FILE, {"TransformType": "Rigid"})

    print(f"[sub-{sub_id}] N4 Bias Field Correction...")
    
    # Step 1: Bias Field Correction
    myutils.log_message("--------------------- Step 1")
    myutils.log_message("N4 Bias Field Correction")
    #n4_step1 = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-N4step1_T1w.nii.gz")
    n4_step1 = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-N4_T1w.nii.gz")

    #n4_bias_correction(CURRENT_FILE, None, n4_step1)
    bias_correction_spm(CURRENT_FILE, n4_step1, mode='t1')
    log_step("N4 Bias Field Correction", CURRENT_FILE, n4_step1, {"Tool": "spm"})

    # Step 2: Running Brain Extraction tools
    print(f"[sub-{sub_id}] Brain Extraction ANTs...")
    myutils.log_message("--------------------- Step 2")
    myutils.log_message("Brain Extraction ANTs")
    mask_prefix = os.path.join(OUTPUT_DIR, "sub-{sub}_desc-brain_".format(sub=sub_id))
    mask_files = run_ants_brain_extraction(n4_step1, f"{OUTPUT_DIR}/brainExtraction_ants/output_t1_", TEMPLATE_MNI_HR, TEMPLATE_MASK)
    log_step("Brain Extraction", n4_step1, mask_files['standard_mask'], {"Method": "ANTs", "Template": "MNI152NLin6Asym_res-01"})

    print(f"[sub-{sub_id}] Brain Extraction FreeSurfer SynthStrip...")

    mask_files['synthstrip'] = run_synthstrip(n4_step1, f"{OUTPUT_DIR}/synthstrip/output_t1_")

    bids_mask_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-brain_mask.nii.gz")
    shutil.copy(mask_files['synthstrip'], bids_mask_file)

    log_step("Brain Extraction", n4_step1, bids_mask_file, {"Method": "SynthStrip", "Tool": "FreeSurfer"})

    # Step 2: Normalizzazione intensità 0-100
    myutils.log_message("--------------------- Step 3")
    myutils.log_message("Normalizzazione intensità 0-100")
    preproc_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-norm_T1w.nii.gz")
    normalize_intensity(n4_step1, bids_mask_file, preproc_file)
    log_step("Intensity Normalization", n4_step1, preproc_file, {"Range": "0-100"})

    '''
    # Step 3: Bias Field Correction
    print(f"[sub-{sub_id}] N4 Bias Field Correction...")

    myutils.log_message("--------------------- Step 4")
    myutils.log_message("N4 Bias Field Correction")
    n4_step2 = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-N4_T1w.nii.gz")
    n4_bias_correction(preproc_file, bids_mask_file, n4_step2)

    log_step("N4 Bias Field Correction", preproc_file, n4_step2, {"Tool": "ANTs N4"})

    myutils.log_message("Normalizzazione intensità 0-100")
    normalized = normalize_intensity(n4_step2, bids_mask_file, n4_step2)
    '''
    '''
    # Step 4: Histogram Matching
    myutils.log_message("--------------------- Step 4")
    myutils.log_message("Histogram Matching Normalization")
    img_matched = histogram_matching(CURRENT_FILE, TEMPLATE_MNI_HR)
    CURRENT_FILE = f"{OUTPUT_BASENAME}.nii.gz"
    OUTPUT_BASENAME = f"{OUTPUT_BASENAME}"
    '''
    # Step 4: Brain masking
    print(f"[sub-{sub_id}] Brain masking...")

    myutils.log_message("--------------------- Step 5")
    myutils.log_message("Brain masking")
    brain_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-brain_T1w.nii.gz")

    apply_brain_mask(preproc_file, bids_mask_file, brain_file)
    log_step("Brain Masking", preproc_file, brain_file, {"Method": "Masking with SynthStrip Mask"})

    # Step 5: Registrazione all'MNI
    print(f"[sub-{sub_id}] Registering T1 to MNI...")

    myutils.log_message("--------------------- Step 6")
    myutils.log_message("MNI registration")

    #template_file = os.path.join(TEMPLATE_DIR, "mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.gz")
    mni_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_desc-preproc_T1w.nii.gz")
    normalized_mni = normalize_to_mni(preproc_file, TEMPLATE_MNI_HR, OUTPUT_DIR, sub_id, transform_type='SyN')
    log_step("MNI Registration", preproc_file, normalized_mni, {"TransformType": "SyN"})
    print(f"[sub-{sub_id}] ✅ T1 preprocessing completed.")

    myutils.log_message("=== Preprocessing completato con successo! ===")

    # === SAVE FINAL JSON REPORT ===
    json_output = os.path.join(OUTPUT_DIR, "sub-{sub_id}_preprocess_t1.json".format(sub_id=sub_id))

    final_report = {
        "PipelineDescription": {
            "Name": "Custom T1 Anatomical Preprocessing",
            "Version": "1.0",
            "CodeURL": "https://github.com/alicegiubergia/project_fmri_hires"
        },
        "SoftwareVersions": {
            "ANTs": ants.__version__,
            "NiBabel": nib.__version__,
            "NumPy": np.__version__
        },
        "SubjectID": sub_id,
        "Input": os.path.basename(INPUT_NIFTI),
        "OutputDirectory": OUTPUT_DIR,
        "Template": "TEMPLATE_MNI_HR",
        "ProcessingSteps": PROCESSING_LOG
    }

    with open(json_output, "w") as f:
        json.dump(final_report, f, indent=4)

    myutils.log_message(f"=== Report JSON salvato: {json_output} ===")


