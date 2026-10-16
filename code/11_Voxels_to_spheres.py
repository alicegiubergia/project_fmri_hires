#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT: 15_voxels_to_spheres.py
DESCRIZIONE:
    Questo script automatizza la creazione di maschere binarie (ROI - Region of Interest) 
    in formato NIfTI a partire da un file CSV contenente coordinate voxel.
    
    Il processo si articola in:
    1. Lettura dinamica del file CSV tramite Pandas.
    2. Raggruppamento dei dati per soggetto e contrasto.
    3. Per ogni gruppo, creazione di sfere di raggio definito (default 1.5cm) 
       centrate sulle coordinate (i, j, k) fornite nel CSV.
    4. Salvataggio delle maschere risultanti come file NIfTI (.nii.gz).

REQUISITI:
    - Pandas, NumPy, Nibabel
    - Le immagini originali (es. spmT_0001.nii) devono essere presenti nella cartella di lavoro.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import os

def create_sphere_mask(reference_img_path, coordinates, diameter_cm=1.5):
    img = nib.load(reference_img_path)
    data_shape = img.shape
    affine = img.affine
    header = img.header
    
    radius_mm = diameter_cm * 10 / 2
    voxel_size = header.get_zooms()[:3]
    radius_voxels = [radius_mm / vs for vs in voxel_size]
    
    mask_data = np.zeros(data_shape, dtype=np.uint8)
    
    for coord in coordinates:
        i, j, k = coord
        
        x_range = np.arange(-int(radius_voxels[0]), int(radius_voxels[0]) + 1)
        y_range = np.arange(-int(radius_voxels[1]), int(radius_voxels[1]) + 1)
        z_range = np.arange(-int(radius_voxels[2]), int(radius_voxels[2]) + 1)
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (x**2 / radius_voxels[0]**2 + 
                        y**2 / radius_voxels[1]**2 + 
                        z**2 / radius_voxels[2]**2) <= 1:
                        ii, jj, kk = int(i + x), int(j + y), int(k + z)
                        if (0 <= ii < data_shape[0] and 
                            0 <= jj < data_shape[1] and 
                            0 <= kk < data_shape[2]):
                            mask_data[ii, jj, kk] = 1
    
    return nib.Nifti1Image(mask_data, affine, header)

def main():
    stats_dir = "/mnt/store1/projects/progetto_layer/data/derivatives/hires_pipeline"
    working_dir = "/mnt/store1/projects/progetto_layer/data/derivatives/rois"
    file_coords = os.path.join(working_dir, "coords.csv")
    diameter_cm = 1.5

    # Carica il CSV
    df = pd.read_csv(file_coords, dtype={'subject': str, 'contrast': str, 'region': str, 'i': int, 'j': int, 'k': int}) 

    # Ottieni la lista delle regioni uniche presenti nel CSV
    regions = df['region'].unique()
    
    # Cicla per ogni soggetto e ogni contrasto presente nel file
    # Questo approccio è più flessibile se hai molti soggetti/contrasti
    grouped = df.groupby(['subject', 'contrast'])

    for (subject, contrast), group in grouped:
        subject = str(subject).zfill(3)
        # Costruisci il nome del file dell'immagine target
        # Esempio: spmT_0001.nii
        img_name = f"{contrast}.nii"
        img_file = os.path.join(stats_dir, f"sub-{subject}", "stats-masked", img_name)
        
        if not os.path.exists(img_file):
            print(f"File {img_name} non trovato, skipping...")
            continue
            
        print(f"Elaborazione: Soggetto {subject}, Contrasto {contrast}")
        
        for region in regions:
            # Filtra le coordinate per la regione corrente
            region_data = group[group['region'] == region]
            coords = list(zip(region_data['i'], region_data['j'], region_data['k']))
            
            if not coords:
                continue
                
            mask_img = create_sphere_mask(img_file, coords, diameter_cm)

            # Salva con nome specifico per soggetto/regione
            fn = f"sub-{subject}_con-{contrast}_label-{region}_mask.nii.gz"
            os.makedirs(os.path.join(working_dir, f"sub-{subject}"), exist_ok=True)
            output_path = os.path.join(working_dir, f"sub-{subject}", fn)
            nib.save(mask_img, output_path)
            print(f"  - Salvata maschera per {region}: {output_path}")

if __name__ == "__main__":
    main()