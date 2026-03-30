#!/usr/local/miniconda3/envs/medea python
# -*- coding: utf-8 -*-

"""
EPI Preprocessing Pipeline

Questo script esegue il preprocessing di un file EPI utilizzando FSL.
Gli step includono:
1. Copia del file originale orig*.nii.gz
2. Estrazione dei parametri della sequenza (n_slices, TR, n_volumi, voxel_size, slice_thickness)
3. Correzione slice-timing `slicetimer` 
4. Stima del campo di distorsione `topup`
5. Applicazione della correzione con `applytopup`
6. Correzione del movimento `mcflirt`
7. Filtraggio HPF

Autore: agiubergia
Data: 2025-03-17
"""

import os
import shutil
import nibabel as nib
import subprocess
import datetime
import numpy as np
from myutils import log_message


# Variabile globale per il log file (inizialmente None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HB=3

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

def get_seq_params(epi_file):
    # Carica il file NIfTI e ottiene i parametri della sequenza
    img = nib.load(epi_file)
    header = img.header
    n_slices = header.get_data_shape()[2]  # Numero di slice
    tr = header.get_zooms()[3]  # Tempo di ripetizione (TR)
    n_volumes = img.shape[3]  # Numero di volumi
    voxel_size = "x".join(map(str, header.get_zooms()[:2]))
    slice_thick = header.get_zooms()[2]
    log_message(f"\nParametri: \nTR={tr}s\nvoxel_size={voxel_size}mm\nn_slices={n_slices}\nslice thickness={slice_thick:.2f}mm")
    output = {'n_slices': n_slices, 'tr': tr, 'HB':HB}
    return output

def remove_dummy(file_path, num_volumes, output_file_path):
    """Rimuove i primi `num_volumes` volumi da un file NIfTI"""
    if not os.path.isfile(output_file_path):
        fsl_command=f"fslroi {file_path} {output_file_path} {num_volumes} -1"
        subprocess.run(fsl_command, shell=True, check=True)
#        data, affine, header = load_nifti(file_path)
#        data_out = data[..., num_volumes:]
#        save_nifti(data_out, affine, header, output_file_path)

def run_NORDIC(part_real, part_imag, part_mag, output_file):
    import glob
    if not os.path.isfile(f"{output_file}"):
        matlab_command = (
            f'matlab -nodisplay -nosplash -nodesktop -r '
            f'"addpath(\'{SCRIPT_DIR}\'); '
            f'NIFTI_NORDIC_batch(\'{part_real}\', \'{part_imag}\', \'{part_mag}\', \'{output_file}\'); exit"' 
            )
        log_message(matlab_command)
        subprocess.run(matlab_command, shell=True, check=True)

def run_slicetimer(epi_file, output_prefix, n_slices, multi_band, tpattern='interleaved'):
    output_dir = os.path.dirname(epi_file)
    file_command = f"python {SCRIPT_DIR}/get_slicetimer_params.py -out {output_dir} -n_slices {n_slices} -tpattern {tpattern} -MB {multi_band}"
    slicetimer_params = os.path.join(output_dir, "slice_timings.txt")
    slicetimer_command = f"slicetimer -i {epi_file} -o {output_prefix}.nii.gz --tcustom={slicetimer_params}"
    log_message(file_command)
    log_message(slicetimer_command)
    if not os.path.isfile(f"{output_prefix}.nii.gz"):
        subprocess.run(file_command, shell=True, check=True)
        subprocess.run(slicetimer_command, shell=True, check=True)

def run_afni_3dQWarp(epi_file, invPE_epi_file, output_prefix):
    output_dir = os.path.dirname(epi_file)
    # Carica i dati
    pPE = nib.load(epi_file)
    iPE = nib.load(invPE_epi_file)

    if not os.path.exists(f"{output_dir}/pPE.nii.gz") or not os.path.exists(f"{output_dir}/iPE.nii.gz"):
        # Estrai i dati come array NumPy
        pPE_data = pPE.get_fdata()
        iPE_data = iPE.get_fdata()

        # Verifica numero di volumi (assunto 4D)
        n_vols = iPE_data.shape[3]
        print(f"Number of volumes in inverse encoding: {n_vols}")

        # Trimma func_inv per usare stesso numero di volumi (nel caso abbia più volumi)
        func_trimmed = pPE_data[..., :n_vols]

        # Calcola la media temporale (asse 3)
        func_mean = np.mean(func_trimmed, axis=3)
        func_inv_mean = np.mean(iPE_data, axis=3)

        # Salva le immagini medie come NIfTI
        func_mean_img = nib.Nifti1Image(func_mean, affine=pPE.affine, header=pPE.header)
        func_inv_mean_img = nib.Nifti1Image(func_inv_mean, affine=iPE.affine, header=iPE.header)

        nib.save(func_mean_img, f"{output_dir}/pPE.nii.gz")
        nib.save(func_inv_mean_img, f"{output_dir}/iPE.nii.gz")
        #subprocess.run(f"3dWarp -deoblique -prefix {output_dir}/pPE.nii.gz {output_dir}/pPE.nii.gz", shell=True, check=True)
        #subprocess.run(f"3dWarp -deoblique -prefix {output_dir}/iPE.nii.gz {output_dir}/iPE.nii.gz", shell=True, check=True)
        command=f"3dQwarp -prefix {output_dir}/warp_AP_to_PA.nii.gz -base {output_dir}/iPE.nii.gz -source {output_dir}/pPE.nii.gz -noXdis -plusminus"
        subprocess.run(command, shell=True, check=True)
        log_message(command)

    if not os.path.exists(f"{output_prefix}.nii.gz"):
        command=f"3dNwarpApply -nwarp {output_dir}/warp_AP_to_PA_PLUS_WARP.nii.gz -source {epi_file} -master {epi_file} -prefix {output_prefix}.nii.gz"
        subprocess.run(command, shell=True, check=True)
        log_message(command)
        #subprocess.run(f"3drefit -oblique_origin {output_prefix}.nii.gz", shell=True, check=True)

    # File target da modificare
    target_img = nib.load(f"{output_prefix}.nii.gz")
    new_img = nib.Nifti1Image(target_img.get_fdata(), pPE.affine, target_img.header)

    # Salva con affine aggiornata
    nib.save(new_img, f"{output_prefix}.nii.gz")

        
def run_topup(epi_file, invPE_epi_file, topup_output_base):
    output_dir = os.path.dirname(epi_file)
    txtFile = os.path.join(output_dir, "topupOut_pPEiPE_acquisitionPars.txt")
    field_output = topup_output_base + "_fieldmap"
    warp_output = topup_output_base + "_WarpField"
    matrix_output = topup_output_base + "_TransformMatrix"
    if not os.path.isfile(f"{output_dir}/pPEiPE_merged_unwarped.nii.gz"):
        # Load inverse Phase Encoding epi file
        iPE, _,_ = load_nifti(invPE_epi_file)
        n_vols = iPE.shape[-1]
        pPE, affine, header = load_nifti(epi_file)
        iPE_mean = np.mean(iPE, axis=-1)
        pPE_mean = np.mean(pPE[:,:,:,:n_vols], axis=-1)
        merged = np.stack((pPE_mean, iPE_mean), axis=-1)
        slices = merged.shape[2]
        if np.mod(slices,2)!=0:
            config_file="b02b0_1.cnf"
        else:
            config_file="b02b0.cnf"

        save_nifti(merged, affine, header, f"{output_dir}/pPEiPE_merged.nii.gz")
        lines = ["0 1 0 0.024\n"] * 1 + ["0 -1 0 0.024\n"] * 1
        if os.path.isfile(txtFile): os.remove(txtFile)
        with open(txtFile, "w") as f: f.writelines(lines)
        
        topup_command = f"topup --imain={output_dir}/pPEiPE_merged.nii.gz --datain={txtFile} --config={config_file} --out={topup_output_base} --fout={field_output} --iout={output_dir}/pPEiPE_merged_unwarped --dfout={warp_output} --rbmout={matrix_output} --logout={output_dir}/topup_logfile -v"
        log_message(topup_command)
        subprocess.run(topup_command, shell=True, check=True)
    return txtFile

def apply_topup(epi_file, txt_file, topup_output_base, output_prefix):
    applytopup_command = f"applytopup --imain={epi_file} --inindex=1 --datain={txt_file} --topup={topup_output_base} --method=jac --interp=trilinear --out={output_prefix}"
    log_message(applytopup_command)
    if not os.path.isfile(f"{output_prefix}.nii.gz"):
        subprocess.run(applytopup_command, shell=True, check=True)

def run_moco(epi_file, output_prefix):
    mcflirt_command = f"mcflirt -in {epi_file} -out {output_prefix} -plots -stats"
    log_message(mcflirt_command)
    if not os.path.isfile(f"{output_prefix}.nii.gz"):
        subprocess.run(mcflirt_command, shell=True, check=True)

def smooth(epi_file, fwhm, output_smooth_path):
    fsl_command = f"fslmaths {epi_file} -s {fwhm/np.sqrt(8 * np.log(2))} {output_smooth_path}"
    if not os.path.isfile(output_smooth_path):
        subprocess.run(fsl_command, shell=True, check=True)

def GMS_norm(epi_file, output_prefix):
    proc = subprocess.run([f"fslstats {epi_file} -M"], shell=True, capture_output=True, check=True)
    mean_intensity = float(proc.stdout.strip())
    scale_factor = 100 / mean_intensity
    if not os.path.isfile(f"{output_prefix}.nii.gz"):
        subprocess.run([f"fslmaths {epi_file} -mul {str(scale_factor)} {output_prefix}.nii.gz"], shell=True, check=True)
        subprocess.run([f"fslmaths {output_prefix}.nii.gz -thr 0 {output_prefix}.nii.gz"], shell=True, check=True)

    log_message(f"Grand mean scaling applied with factor {scale_factor:.6f}. Output saved to {output_prefix}.nii.gz")


# Funzione per normalizzare l'immagine anatomica con ANTs (registrazione all'MNI)
def normalize_to_struct(epi_image_path, epi_image_4reg, anat_path, epi_in_anat_space_path, anat_in_epi_space_path):
    """
    Esegue la registrazione dell'immagine EPI alla strutturale con ANTs.
    """
    import sys
    from pathlib import Path


    subprocess.run(f"fslmaths {epi_image_4reg} -Tmean {epi_image_4reg.split('.nii')[0]}_mean_reg.nii.gz", shell=True, check=True)
    fixedImage=f"{epi_image_4reg.split('.nii')[0]}_mean_reg.nii.gz"
    movingImage=anat_path
    subprocess.run(f"{SCRIPT_DIR}/epiSlabxT1.sh {fixedImage} {movingImage}", shell=True, check=True)

    folder = Path(os.path.dirname(epi_image_4reg))

    matches = list(folder.glob("*from-T1w_to-epiSlab*.mat"))
    if len(matches) == 0:
        sys.exit(f"❌ Nessun file trovato in {folder}")
    elif len(matches) > 1:
        msg = "\n".join(f"  {m}" for m in matches)
        sys.exit(f"❌ Più di un file trovato:\n{msg}")
    else:
        mtx = matches[0]
        print(f"✅ File trovato: {mtx}")

    reg_command=f"antsApplyTransforms -d 3 -e 3 -i {epi_image_path} -r {anat_path} -o {epi_in_anat_space_path} -n linear -t [{mtx},1] -v"
    subprocess.run(reg_command, shell=True, check=True)
    reg_command=f"antsApplyTransforms -d 3 -i {anat_path} -r {fixedImage} -o {anat_in_epi_space_path} -n linear -t {mtx} -v"
    subprocess.run(reg_command, shell=True, check=True)


    log_message(f"{epi_image_path} -> {epi_in_anat_space_path}")
    log_message(f"{anat_path} -> {anat_in_epi_space_path}")

    return mtx

def normalize_to_mni(epi_image_path, moving_path, template_path, output_warped_path):
    """
    Esegue la registrazione dell'immagine EPI alla strutturale con ANTs.
    """
    import ants

    if not os.path.isfile(output_warped_path):
        subprocess.run(f"fslmaths {moving_path} -Tmean {moving_path.split('.nii')[0]}_mean_reg.nii.gz", shell=True, check=True)

        input_epi_img = ants.image_read(f"{epi_image_path}")
        input_mean_img = ants.image_read(f"{moving_path.split('.nii')[0]}_mean_reg.nii.gz")
        template_img = ants.image_read(template_path)
        prefix = os.path.dirname(epi_image_path)
        epi_mask = ants.get_mask(input_mean_img, cleanup=2)
        registration = ants.registration(fixed=template_img, 
                                         moving=input_mean_img,
                                         type_of_transform='SyN', 
                                         outprefix=prefix+'/SyN_2mni',
                                         moving_mask = epi_mask,
                                         verbose=True)
        ants.image_write(epi_mask, f"{moving_path.split('.nii')[0]}_mean_reg_mask.nii.gz")
        warped_mean_image = registration['warpedmovout']
        moving_to_fixed = registration['fwdtransforms']
        fixed_to_moving = registration['invtransforms']
        warped_image = ants.apply_transforms(fixed=template_img, moving=input_epi_img, 
                                             transformlist=moving_to_fixed, 
                                             interpolator='BSpline', 
                                             imagetype=3, 
                                             verbose=True)
        ants.image_write(warped_image, output_warped_path)
        ants.write_transform

#        mtx=glob.glob(prefix+'/Rigid_2struct*.mat')[0]
#        reg_command=f"antsApplyTransforms -d 3 -e 3 -i {epi_image_path} -r {struct_path} -o {output_warped_path} -n Linear -t {mtx} -v"
#        subprocess.run(reg_command, shell=True, check=True)
    log_message(f"Warped image -> {output_warped_path}")
    return output_warped_path


