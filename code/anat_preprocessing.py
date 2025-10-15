#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anatomical T1 Preprocessing Pipeline

Questo script esegue il preprocessing di un file anatomico. 
Gli step includono:
1. Backup del file originale come orig*.nii.gz
2. Brain extraction con ANTs.
3. Correzione del bias field con N4 (SimpleITK).
4. Normalizzazione dell'intensità dell'immagine (0-100)
5. Registrazione al template (MNI152 - ICBM 2009b Nonlinear Asymmetric) con ANTs.
6. Salvataggio dei file in cartella derivatives BIDS-compliant.

Autore: Alice Giubergia
Data: 2025-03-17
"""

import os
import json
import nibabel as nib
import numpy as np
import ants
import subprocess
from datetime import datetime
import shutil
import myutils

# Log globale degli step per il JSON finale
PROCESSING_LOG = []

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

def save_bids_json(nifti_path, metadata_dict):
    """
    Salva un file JSON con i metadati BIDS associati al file NIfTI.
    """
    json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(metadata_dict, json_file, indent=4)
    myutils.log_message(f"Metadata JSON salvato: {json_path}")


def brain_extraction(input_image_path, input_mask_path, output_brain_path):
    """
    Maschera la input image.
    """
    if not os.path.isfile(output_brain_path):
        data, affine, header = load_nifti(input_image_path)
        mask, _, _ = load_nifti(input_mask_path)
        new_data = data * mask
        save_nifti(new_data, affine, header, output_brain_path)
    myutils.log_message(f"output: {output_brain_path}")

# Funzione per mask computation con ANTs
def mask_computation_ants(input_image_path, output_brain_mask_path):
    """
    Esegue la brain extraction utilizzando ANTs.
    - Usa il template MNI per migliorare l'estrazione.
    """
    input_img = ants.image_read(input_image_path)

    mask = ants.get_mask(input_img, cleanup=4)

    # Salva il risultato
    ants.image_write(mask, output_brain_mask_path)
    myutils.log_message(f"ANTs Mask Computation terminato: {input_image_path} -> {output_brain_mask_path}")
    return output_brain_mask_path

def brain_extraction_ants(t1_path, output_prefix, template_dir):
    template_mask = f"{template_dir}/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii"
    template = f"{template_dir}/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii"
    command = f"antsBrainExtraction.sh -d 3 -a {t1_path} -m {template_mask} -e {template} -o {output_prefix}"
    myutils.log_message(command)

    if not os.path.isfile(f"{output_prefix}BrainExtractionMask.nii.gz"):
        subprocess.run(command, shell=True, check=True)
    return f"{output_prefix}BrainExtractionMask.nii.gz"

# Funzione per la correzione del bias field con N4 (SimpleITK)
def n4_bias_correction(input_image_path, input_mask_path, output_image_path):
    """
    Applica la correzione del bias field N4.
    """
    if not os.path.isfile(output_image_path):
        input_image = ants.image_read(input_image_path)
        input_mask = ants.image_read(input_mask_path)
        corrected_img = ants.n4_bias_field_correction(input_image, mask=input_mask, shrink_factor=4, rescale_intensities=True)
        ants.image_write(corrected_img, output_image_path)
    myutils.log_message(f"{input_image_path} -> {output_image_path}")
    return output_image_path

# Funzione per normalizzare l'intensità tra 0 e 100
def normalize_intensity(input_image_path, mask_image_path, output_image_path):
    """
    Normalizza l'intensità dell'immagine tra 0 e 100.
    """
    data, affine, header = load_nifti(input_image_path)
    mask, _, _ = load_nifti(mask_image_path)
    min_val = np.mean(data)
    max_val = np.max(data)
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

    prefix = os.path.join(output_dir, f"sub-{sub_id}_from-T1w_to-MNI_xfm0{transform_type}")

    if not os.path.isfile(warped_output):
        moving_img = ants.image_read(moving_path)
        fixed_img = ants.image_read(fixed_path)
        myutils.log_message(f"--- Starting {transform_type} registration to MNI ---")

        registration = ants.registration(fixed=fixed_img, 
                                         moving=moving_img, 
                                         type_of_transform=transform_type, 
                                         outprefix=prefix,
                                         verbose=True)

        warped_img = registration['warpedmovout']
        ants.image_write(warped_img, warped_output)

        # Salva i file di trasformazione
        ants.write_transform

        myutils.log_message(f"Registrazione completata ({transform_type})")
        myutils.log_message(f"  Warped: {warped_output}")
    else:
        myutils.log_message(f"Registrazione {transform_type} già presente: {warped_output}")

    return warped_output


# Main script
if __name__ == "__main__":
    
    # Step 0: Set filenames
    TEMPLATE_DIR="../templates"
    TEMPLATE_MNI_HR = f"{TEMPLATE_DIR}/mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.gz"
    INPUT_NIFTI = "../nifti/sub-01/anat/sub-00_T1w.nii.gz"
    sub_id = os.path.basename(os.path.dirname(INPUT_NIFTI)).split("-")[1]

    # Cartella BIDS-compliant per i risultati
    OUTPUT_DIR = os.path.join("derivatives", "mypipeline", f"sub-{sub_id}", "anat")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Log
    LOG_FILE = os.path.join(OUTPUT_DIR, "anat_preprocessing.log")
    myutils.set_log_file(LOG_FILE)

    #INPUT_DIR = os.path.dirname(os.path.abspath(INPUT_NIFTI))
    #OUTPUT_DIR = INPUT_DIR.replace("nifti", "derivatives/mypipeline")

    ORIG_FILE = f"{OUTPUT_DIR}/sub-{sub_id}_orig_T1w.nii.gz"
    shutil.copy(INPUT_NIFTI, ORIG_FILE)
  
    myutils.log_message("=== Inizio Preprocessing ===")

    # Step 0: move to MNI space
    myutils.log_message("--------------------- Step 0")
    myutils.log_message("Moving to MNI space")
    CURRENT_FILE = normalize_to_mni(moving_path=INPUT_NIFTI, 
                                    fixed_path=TEMPLATE_MNI_HR, 
                                    output_dir=OUTPUT_DIR, 
                                    sub_id=sub_id, transform_type='Rigid')

    log_step("Move to MNI space", INPUT_NIFTI, CURRENT_FILE, {"TransformType": "Rigid"})

    # Step 1: Running Brain Extraction ANTs
    myutils.log_message("--------------------- Step 1")
    myutils.log_message("Brain Extraction ANTs")
    mask_prefix = os.path.join(OUTPUT_DIR, "sub-{sub}_desc-brain_".format(sub=sub_id))
    mask_file = brain_extraction_ants(CURRENT_FILE, f"{OUTPUT_DIR}/brainExtraction_ants/output_", TEMPLATE_DIR)
    bids_mask_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_space-MNI_desc-brain_mask.nii.gz")
    shutil.copy(mask_file, bids_mask_file)

    log_step("Brain Extraction", CURRENT_FILE, bids_mask_file, {"Method": "ANTs", "Template": "MNI152NLin2009bAsym"})

    # Step 2: Normalizzazione intensità 0-100
    myutils.log_message("--------------------- Step 2")
    myutils.log_message("Normalizzazione intensità 0-100")
    preproc_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_space-MNI_desc-preproc_T1w.nii.gz")
    normalized = normalize_intensity(CURRENT_FILE, bids_mask_file, preproc_file)
    log_step("Intensity Normalization", CURRENT_FILE, preproc_file, {"Range": "0-100"})

    # Step 3: Bias Field Correction
    myutils.log_message("--------------------- Step 3")
    myutils.log_message("N4 Bias Field Correction")
    n4_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_space-MNI_desc-N4_T1w.nii.gz")
    n4_corrected = n4_bias_correction(normalized, bids_mask_file, n4_file)

    log_step("N4 Bias Field Correction", normalized, n4_corrected, {"Tool": "ANTs N4"})

    myutils.log_message("Normalizzazione intensità 0-100")
    normalized = normalize_intensity(n4_corrected, bids_mask_file, n4_corrected)

    '''
    # Step 4: Histogram Matching
    log_message("--------------------- Step 4")
    log_message("Histogram Matching Normalization")
    img_matched = histogram_matching(CURRENT_FILE, TEMPLATE_MNI_HR)
    CURRENT_FILE = f"{OUTPUT_BASENAME}.nii.gz"
    OUTPUT_BASENAME = f"{OUTPUT_BASENAME}"
    '''
    # Step 4: Brain masking
    myutils.log_message("--------------------- Step 4")
    myutils.log_message("Brain masking")
    brain_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_space-MNI_desc-brain_T1w.nii.gz")

    brain_extraction(normalized, bids_mask_file, brain_file)
    log_step("Brain Masking", normalized, brain_file, {"Method": "Masking with ANTs Mask"})

    # Step 5: Registrazione all'MNI
    myutils.log_message("--------------------- Step 5")
    myutils.log_message("MNI registration")

    template_file = os.path.join(TEMPLATE_DIR, "mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.gz")
    mni_file = os.path.join(OUTPUT_DIR, f"sub-{sub_id}_space-MNI_desc-preproc_T1w.nii.gz")
    normalized_mni = normalize_to_mni(normalized, TEMPLATE_MNI_HR, mni_file, sub_id, transform_type='SyN')
    log_step("MNI Registration", normalized, normalized_mni, {"TransformType": "SyN"})
    myutils.log_message("=== Preprocessing completato con successo! ===")

    # === SAVE FINAL JSON REPORT ===
    json_output = os.path.join(OUTPUT_DIR, f"sub-01_T1w_preproc.json")

    final_report = {
        "PipelineDescription": {
            "Name": "Custom ANTs Anatomical Preprocessing",
            "Version": "1.0",
            "CodeURL": "https://github.com/alicegiubergia/project_fmri_hires"
        },
        "SoftwareVersions": {
            "ANTs": ants.__version__,
            "NiBabel": nib.__version__,
            "NumPy": np.__version__
        },
        "Input": os.path.basename(INPUT_NIFTI),
        "OutputDirectory": OUTPUT_DIR,
        "Template": "MNI152NLin2009bAsym",
        "ProcessingSteps": PROCESSING_LOG
    }

    with open(json_output, "w") as f:
        json.dump(final_report, f, indent=4)

    myutils.log_message(f"=== Report JSON salvato: {json_output} ===")


