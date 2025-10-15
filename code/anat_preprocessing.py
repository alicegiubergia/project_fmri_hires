#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anatomical T1 Preprocessing Pipeline

Questo script esegue il preprocessing di un file anatomico. 
Gli step includono:
1. Backup del file originale come orig*.nii.gz
2. Brain extraction con ANTs.
3. Correzione del bias field con N4 (SimpleITK).
4. Normalizzazione dell’intensità dell’immagine (0-100)
5. Registrazione al template (MNI152 - ICBM 2009b Nonlinear Asymmetric) con ANTs.

Autore: Alice Giubergia
Data: 2025-03-17
"""

import os
import nibabel as nib
import numpy as np
import ants
import subprocess
import datetime
import shutil
import myutils
#import antspynet

# Variabile globale per il log file (inizialmente None)
LOGFILE = None
TEMPLATE_DIR="../templates"

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

# Funzione per normalizzare l'immagine anatomica con ANTs (registrazione all'MNI)
def to_mni_space(moving_path, fixed_path, output_warped_path):
    """
    Esegue la registrazione rigida dell'immagine anatomica al template MNI con ANTs.
    """
    if not os.path.isfile(output_warped_path):
        input_img = ants.image_read(moving_path)
        template_img = ants.image_read(fixed_path)
        prefix = os.path.dirname(output_warped_path)
        registration = ants.registration(fixed=template_img, 
                                         moving=input_img, 
                                         type_of_transform='Rigid', 
                                         outprefix=prefix+'/Rigid', 
                                         verbose=True)

        warped_image = registration['warpedmovout']
        moving_to_fixed = registration['fwdtransforms']
        fixed_to_moving = registration['invtransforms']

        ants.image_write(warped_image, output_warped_path)
        ants.write_transform
    myutils.log_message(f"{moving_path} -> {output_warped_path}")
    return output_warped_path

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

def brain_extraction_ants(t1_path, output_prefix):
    template_mask = f"{TEMPLATE_DIR}/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii"
    template = f"{TEMPLATE_DIR}/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii"
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
        #input_image = sitk.ReadImage(input_image_path)
        #input_mask = sitk.ReadImage(input_mask_path, sitk.sitkUInt8)
        #corrector = sitk.N4BiasFieldCorrectionImageFilter()
        #output_image = corrector.Execute(input_image, input_mask)
        #sitk.WriteImage(output_image, output_image_path)
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
def normalize_to_mni(moving_path, fixed_path, output_warped_path):
    """
    Esegue la registrazione dell'immagine anatomica al template MNI con ANTs.
    """
    if not os.path.isfile(output_warped_path):
        input_img = ants.image_read(moving_path)
        template_img = ants.image_read(fixed_path)
        prefix = os.path.dirname(moving_path)
        registration = ants.registration(fixed=template_img, 
                                         moving=input_img, 
                                         type_of_transform='SyN', 
                                         outprefix=prefix+'/SyN', 
                                         verbose=True)

        warped_image = registration['warpedmovout']
        moving_to_fixed = registration['fwdtransforms']
        fixed_to_moving = registration['invtransforms']

        ants.image_write(warped_image, output_warped_path)
        ants.write_transform
    myutils.log_message(f"{warped_image} -> {output_warped_path}")
    return output_warped_path

# Main script
if __name__ == "__main__":
    
    # Step 0: Set filenames
    TEMPLATE_DIR="../templates"

    INPUT_NIFTI = "../nifti/sub-01/anat/sub-00_T1w.nii.gz"
    TEMPLATE_MNI_HR = f"{TEMPLATE_DIR}/mni_icbm152_nlin_asym_09b_nifti/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.gz"
    
    INPUT_DIR = os.path.dirname(os.path.abspath(INPUT_NIFTI))
    OUTPUT_DIR = INPUT_DIR.replace("nifti", "derivatives/mypipeline")

        
    # Crea la cartella se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    ORIG_FILE = f"{OUTPUT_DIR}/origT1w.nii.gz"
    shutil.copy(INPUT_NIFTI, ORIG_FILE)

    OUTPUT_BASENAME = ORIG_FILE.replace("orig", "").split('.nii')[0]

    LOG_FILE = os.path.join(INPUT_DIR, "anat_preprocessing.log")
    myutils.set_log_file(LOG_FILE)
    
    myutils.log_message("=== Inizio Preprocessing ===")

    # Step 0: move to MNI space
    myutils.log_message("--------------------- Step 0")
    myutils.log_message("Moving to MNI space")
    CURRENT_FILE = to_mni_space(INPUT_NIFTI, TEMPLATE_MNI_HR, f"{OUTPUT_BASENAME}.nii.gz")


    # Step 1: Running Brain Extraction ANTs
    myutils.log_message("--------------------- Step 1")
    myutils.log_message("Brain Extraction ANTs")
    mask = brain_extraction_ants(CURRENT_FILE, f"{OUTPUT_DIR}/brainExtraction_ants/output_")
    shutil.copy(mask, f"{OUTPUT_BASENAME}_mask.nii.gz")

    # Step 2: Normalizzazione intensità 0-100
    myutils.log_message("--------------------- Step 2")
    myutils.log_message("Normalizzazione intensità 0-100")
    normalized_intensity = normalize_intensity(CURRENT_FILE, mask, f"{OUTPUT_BASENAME}.nii.gz")
    CURRENT_FILE = f"{OUTPUT_BASENAME}.nii.gz"

    # Step 3: Bias Field Correction
    myutils.log_message("--------------------- Step 3")
    myutils.log_message("N4 Bias Field Correction")
    bias_corrected = n4_bias_correction(CURRENT_FILE, mask, f"{OUTPUT_BASENAME}_N4.nii.gz")
    CURRENT_FILE = f"{OUTPUT_BASENAME}_N4.nii.gz"
    OUTPUT_BASENAME = f"{OUTPUT_BASENAME}_N4"

    myutils.log_message("Normalizzazione intensità 0-100")
    normalized_intensity = normalize_intensity(CURRENT_FILE, mask, f"{OUTPUT_BASENAME}.nii.gz")
    CURRENT_FILE = f"{OUTPUT_BASENAME}.nii.gz"

    '''
    # Step 4: Histogram Matching
    log_message("--------------------- Step 4")
    log_message("Histogram Matching Normalization")
    img_matched = histogram_matching(CURRENT_FILE, TEMPLATE_MNI_HR)
    CURRENT_FILE = f"{OUTPUT_BASENAME}.nii.gz"
    OUTPUT_BASENAME = f"{OUTPUT_BASENAME}"
    '''
    # Step 4: Brain extraction
    myutils.log_message("--------------------- Step 4")
    myutils.log_message("Brain extraction")
    brain_extraction(CURRENT_FILE, mask, f"{OUTPUT_BASENAME}_brain.nii.gz")

    # Step 5: Registrazione all'MNI
    myutils.log_message("--------------------- Step 5")
    myutils.log_message("MNI registration")
    normalized_mni = normalize_to_mni(bias_corrected, TEMPLATE_MNI_HR, f"{OUTPUT_BASENAME}_2mni.nii.gz")

    myutils.log_message("=== Preprocessing completato con successo! ===")

