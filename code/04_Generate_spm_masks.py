import os
import subprocess
import ants
import antspynet
from bids import BIDSLayout
import argparse
import json
import nibabel as nib
import numpy as np
from scipy import ndimage

def create_bids_description(output_path, pipeline_name="SPM_Masking_Pipeline"):
    """
    Crea il file dataset_description.json obbligatorio per i derivati BIDS.
    """
    description = {
        "Name": pipeline_name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Custom SynthStrip-ANTsPyNet Pipeline",
                "Version": "1.0",
                "Description": "Brain extraction for SPM using SynthStrip and ANTsPyNet refinement"
            }
        ],
        "HowToAcknowledge": "Please cite FreeSurfer SynthStrip and ANTsPyNet."
    }
    
    file_path = os.path.join(output_path, "dataset_description.json")
    
    with open(file_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f"[INFO] Created BIDS description at: {file_path}")



def generate_spm_mask(args):
    # 1. Setup percorsi: Input dalle hires, Output in una cartella dedicata
    bids_root = args.bids_root
    input_deriv = os.path.join(bids_root, "derivatives", "hires_pipeline")
    output_deriv = os.path.join(bids_root, "derivatives", "spm_masks_b-2")
    os.makedirs(output_deriv, exist_ok=True)

    create_bids_description(output_deriv, pipeline_name="spm_masks")
    
    layout = BIDSLayout(input_deriv, validate=False)
    subjects = args.subs if args.subs else layout.get_subjects()

    for sub in subjects:
        print(f"\n[PIPELINE MAKS FOR SPM] >>> Processing sub-{sub}")
        
        # Directory di output specifica per il soggetto
        sub_output_dir = os.path.join(output_deriv, f"sub-{sub}", "anat")
        os.makedirs(sub_output_dir, exist_ok=True)

        # 2. Recupero T1w (BIDS style)
        t1_files = layout.get(subject=sub, suffix='T1w', desc='N4', extension='.nii.gz', return_type='file')
        if not t1_files:
            print(f"[!] T1w non trovato per sub-{sub}. Passo avanti.")
            continue
            
        t1_full_path = t1_files[0]
        t1_dir = os.path.dirname(t1_full_path)
        t1_fn = os.path.basename(t1_full_path)

        # Nomi file definitivi per SPM
        # Usiamo desc-spmMask per renderlo BIDS-compliant ma chiaro
        synth_mask_internal = f"sub-{sub}_temp_synthstrip_mask.nii.gz"
        synth_brain_internal = f"sub-{sub}_temp_synthstrip_brain.nii.gz"
        ants_mask_internal = f"sub-{sub}_temp_ants_mask.nii.gz"
        final_mask_name = f"sub-{sub}_desc-spm_mask.nii.gz"

        # --- STEP 1: SynthStrip (Docker) ---
        # Creiamo una maschera grezza ma robusta
        print(f"[1/2] Estrazione iniziale con SynthStrip...")
        
        docker_cmd = (
            "docker run --rm --user $(id -u):$(id -g) "
            f"-v {t1_dir}:/data -v {sub_output_dir}:/output "
            "cesne/freesurfer:8.0 mri_synthstrip "
            f"-i /data/{t1_fn} "
            f"-m /output/{synth_mask_internal} "
            f"-o /output/{synth_brain_internal} " 
            "--no-csf "
            "-b 2"
        )

        try:
            subprocess.run(docker_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Docker ha fallito su sub-{sub}")
            continue

        # --- STEP 2: ANTsPyNet Refinement ---
        # Usiamo ANTsPyNet per raffinare il risultato e creare la maschera finale
        print(f"[2/2] Raffinamento maschera con ANTsPyNet...")
        
        try:
            # Carichiamo l'immagine originale
            synth_brain_img = ants.image_read(os.path.join(sub_output_dir, synth_brain_internal))
            synth_mask_img = ants.image_read(os.path.join(sub_output_dir,synth_mask_internal))
            
            # ANTsPyNet brain extraction (flavor no-brainer)
            # Produce una maschera di probabilità/segmentazione
            prob_mask = antspynet.brain_extraction(synth_brain_img, modality="t1nobrainer", verbose=False)
            ants.image_write(prob_mask, os.path.join(sub_output_dir, ants_mask_internal))
            
            #final_mask = ants.threshold_image(prob_mask, 2, 2, 1)
            binary_prob = ants.threshold_image(prob_mask, 1, 2, 1, 0)
            final_mask = binary_prob * synth_mask_img
            final_mask_path = os.path.join(sub_output_dir, final_mask_name)
            
            # Salviamo la maschera finale per SPM
            final_mask_path = os.path.join(sub_output_dir, final_mask_name)
            ants.image_write(final_mask, final_mask_path)
            

            
            # Pulizia file temporaneo
            #temp_file = os.path.join(sub_output_dir, synth_mask_internal)
            #if os.path.exists(temp_file):
            #    os.remove(temp_file)
            print(f"[SUCCESS] Maschera SPM creata: {final_mask_name}")

        except Exception as e:
            print(f"[ERROR] Errore ANTsPyNet: {e}")

        try:
            subprocess.run(f"gunzip {final_mask_path}", check=True, shell=True)
        except Exception as e:
                print(f"[ERROR] Errore gunzip: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True)
    parser.add_argument("--subs", nargs="+")
    args = parser.parse_args()
    generate_spm_mask(args)