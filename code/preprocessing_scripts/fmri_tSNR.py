#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:09:29 2023

@author: alicegiubergia
"""

import sys
import argparse
import os
import numpy as np
import nibabel as nib
import nibabel.processing 
import matplotlib.pyplot as plt
from scipy.io import savemat
import nibabel.nifti1 as nifti
import pandas as pd
import seaborn as sns
import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount=0.5):  
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def tSNR(data, affine, filename):
    Tmean = np.mean(data, axis=-1)
    Tsd = np.std(data, axis=-1)
    tsnr = Tmean/Tsd
    tsnr[np.where(np.isinf(tsnr))] = np.nan
    tsnr_save=tsnr.copy()
    tsnr_save[np.isnan(tsnr_save)] = 0
    tSNRimg = nifti.Nifti1Image(tsnr_save,affine)
    nib.save(tSNRimg, filename)
    #savemat(filename.split('.')[0]+'.mat',{'tSNR':tsnr})       
    return tsnr

import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def generate_image_in_memory(image_data, cmap=None, norm=None):
    # Crea una figura in memoria per visualizzare 24 slice
    image_data[np.isnan(image_data)]=0
    mid_slice = image_data.shape[2] // 2
    mid_volume = image_data.shape[-1] // 2
    
    fig, ax = plt.subplots(4, 6, figsize=(18, 12))  # 4 righe e 6 colonne

    slices = np.linspace(2, image_data.shape[2] -3, 24, dtype=int)  # 24 slices centrati sul mezzo

    for i, sl in enumerate(slices):
        row = i // 6  # Righe
        col = i % 6   # Colonne
        if len(image_data.shape) > 3:
            ax[row, col].imshow(np.rot90(image_data[:, :, sl,mid_volume],k=1), cmap=cmap, norm=norm)
        else:
            ax[row, col].imshow(np.rot90(image_data[:, :, sl],k=1), cmap=cmap, norm=norm)
        ax[row, col].axis("off")
        ax[row, col].set_title(f"Slice {sl}")
    
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal')
    
    # Salva l'immagine direttamente in un oggetto BytesIO (in memoria)
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    
    return buf  # Restituisce il buffer in memoria

def generate_histogram_in_memory(data):
    # Crea un istogramma in memoria
    sns.set_style("whitegrid")

    fig, ax=plt.subplots(figsize=(12,6))
    plt.boxplot(np.ravel(snr[~np.isnan(snr)]), vert=False)
    plt.ylabel('tSNR')
    plt.yticks([])
    plt.title("tSNR Brain Box Plot")
    fig.savefig('%s/%s_tSNRbox.png' %(out_folder, fname), dpi=400)
    plt.clf()
    fig, ax=plt.subplots(figsize=(12,6))
    n_bins = 100

    sns.histplot(np.ravel(snr[~np.isnan(snr)]), bins=n_bins, kde=True, ax=ax)
    plt.ylabel('tSNR')
    plt.xlabel('Frequency')
    plt.title("tSNR Distribution")
    max_tsnr = np.nanmax(snr)
    perc = int(np.nanpercentile(snr,99))
    plt.axvline(perc, color='red', linestyle='dashed', linewidth=1)
    plt.annotate(f'{perc:d}', xy=(perc,0.02), xytext=(perc+3,25000),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=12, color='red')
    # Salva l'immagine dell'istogramma in memoria
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    
    return buf  # Restituisce il buffer in memoria

def generate_html_report(EPIimg, snr, description, out_file):
    # Crea le immagini delle slice in memoria
    norm = Normalize(vmin=0, vmax=100)
    epi_img_buf = generate_image_in_memory(EPIimg, cmap="gray")
    
    tsnr_img_buf = generate_image_in_memory(snr, cmap="inferno", norm=norm)
    norm_individual = Normalize(vmin=0, vmax=np.nanpercentile(snr,99))
    tsnr_img_buf_norm_individual = generate_image_in_memory(snr, cmap="inferno", norm=norm_individual)
    # Crea l'istogramma in memoria
    tsnr_hist_buf = generate_histogram_in_memory(np.ravel(snr[~np.isnan(snr)]))
    
    # Converti le immagini e l'istogramma in memoria in stringhe base64
    epi_img_base64 = base64.b64encode(epi_img_buf.read()).decode('utf-8')
    tsnr_img_base64 = base64.b64encode(tsnr_img_buf.read()).decode('utf-8')
    tsnr_img_individual_base64 = base64.b64encode(tsnr_img_buf_norm_individual.read()).decode('utf-8')
    tsnr_hist_base64 = base64.b64encode(tsnr_hist_buf.read()).decode('utf-8')

    # Contenuto HTML con immagini in base64
    html_content = f"""
    <html>
    <head>
        <title>tSNR Report</title>

    </head>
    <body>
        <h1 style="font-family: Calibri">tSNR Analysis Report</h1>
        
        <h2 style="font-family: Calibri">{out_file.split('_')[-1]}</h2>

        <p style="font-family: Calibri">{description}</p>
        <img src='data:image/png;base64,{epi_img_base64}' width='1600'>
        
        <h2 style="font-family: Calibri">tSNR Map</h2>
        <img src='data:image/png;base64,{tsnr_img_base64}' width='1600'>

        <h2 style="font-family: Calibri">tSNR Map - individually normalized</h2>
        <img src='data:image/png;base64,{tsnr_img_individual_base64}' width='1600'>
        
        <h2 style="font-family: Calibri">tSNR Histogram</h2>
        <img src='data:image/png;base64,{tsnr_hist_base64}' width='1600'>
    </body>
    </html>
    """
    
    # Salva il report HTML
    with open(os.path.join(out_file + ".html"), "w") as f:
        f.write(html_content)


 
def arg_summary():
    print("----------------")  
    print("Input arguments:")   
    print("----------------")  
     
    for arg in vars(args):
        print(arg, getattr(args, arg))

    

parser = argparse.ArgumentParser(description='Estimate tSNR of nifti file and generate SNR 3D.')
parser.add_argument('-epi', metavar='FILE', type=str, required=True,
                    help='EPI file.')
parser.add_argument('-mas', metavar='FILE', type=str, required=False,
                    help='MASK file.')
parser.add_argument('-seg', metavar='FILE', type=str,
                    help='SEGMENTATION file.')
parser.add_argument('-out', metavar='FOLDER', type=str,
                    help='Output folder.')
parser.add_argument('--report',  action='store_true')

try:
   args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)


arg_summary()

if args.epi.startswith('/'):
    epi_file = args.epi
elif args.epi.startswith('.'):
    # built path with current directory
    epi_file = os.path.join(os.path.abspath(os.getcwd()), '/'.join(args.epi.split('/')[1:]))
    #print(epi_file)
else:
    epi_file = os.path.join(os.path.abspath(os.getcwd()), args.epi)

if args.seg:
    if args.seg.startswith('/'):
        seg_file = args.seg
    elif args.seg.startswith('.'):
        # built path with current directory
        seg_file = os.path.join(os.path.abspath(os.getcwd()),  '/'.join(args.seg.split('/')[1:]))
    else:
        seg_file = os.path.join(os.path.abspath(os.getcwd()), args.seg)
if args.mas:
    if args.mas.startswith('/'):
        mas_file = args.mas
    elif args.mas.startswith('.'):
        # built path with current directory
        mas_file = os.path.join(os.path.abspath(os.getcwd()),  '/'.join(args.mas.split('/')[1:]))
    else:
        mas_file = os.path.join(os.path.abspath(os.getcwd()), args.mas)

if not os.path.isfile(epi_file):    
    print("EPI file does not exists. Check input.")
    sys.exit(1)

# Make sure you have a place to save the results:
if not args.out:
    out_folder = os.path.join(os.path.dirname(epi_file), 'tSNR')
else:
    if args.out.startswith('/'):
        out_folder = args.out
    elif args.out.startswith('.'):
        # built path with current directory
        out_folder = os.path.join(os.path.abspath(os.getcwd()), '/'.join(args.out.split('/')[1:]))    
    else:
        out_folder = os.path.join(os.path.abspath(os.getcwd()), args.out)

print(f"output_folder: {out_folder}")
os.makedirs(out_folder, exist_ok=True)

if __name__ == "__main__":

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    EPI=nib.load(epi_file)
    EPIimg=EPI.get_fdata()
    EPIhdr=EPI.header
    EPIaffine=EPI.affine
    tr = [EPIhdr.get_zooms()[3] if len(EPIhdr.get_zooms()) > 0 else None][0]
    voxel_size = "x".join(map(str, EPIhdr.get_zooms()[:2]))
    slice_thick = EPIhdr.get_zooms()[2]
    description = f"TR={tr:.1f}s <br /> voxel size={voxel_size}mm <br /> slice thickness={slice_thick}"
    if args.mas:
        if not os.path.isfile(mas_file):    
            print("No MASK file provided. Computing without masking.")
        else:
            MASK=nib.load(mas_file)
            MASKimg=MASK.get_fdata().astype(bool)
            EPIimg[~MASKimg] = 0
                
    
    fname = os.path.split(epi_file)[-1].split('.nii')[0]
    
    
    snr = tSNR(EPIimg, EPIaffine,'%s/%s_tSNR.nii.gz' %(out_folder, fname))
    
    if args.report:
        out_file = os.path.join(out_folder, "report_" + fname)
        generate_html_report(EPIimg, snr, description, out_file)
    sns.set_style("whitegrid")

    fig, ax=plt.subplots(figsize=(12,6))
    plt.boxplot(np.ravel(snr[~np.isnan(snr)]), vert=False)
    plt.ylabel('tSNR')
    plt.yticks([])
    plt.title("tSNR Brain Box Plot")
    #fig.savefig('%s/%s_tSNRbox.png' %(out_folder, fname), dpi=400)
    plt.clf()
    fig, ax=plt.subplots(figsize=(12,6))
    n_bins = 100

    sns.histplot(np.ravel(snr[~np.isnan(snr)]), bins=n_bins, kde=True, ax=ax)
    plt.ylabel('tSNR')
    plt.xlabel('Frequency')
    plt.title("tSNR Distribution")
    max_tsnr = np.nanmax(snr)
    perc = int(np.nanpercentile(snr,99))
    plt.axvline(perc, color='red', linestyle='dashed', linewidth=1)
    plt.annotate(f'{perc:d}', xy=(perc,0.02), xytext=(perc+3,25000),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=12, color='red')
    #fig.savefig('%s/%s_tSNRhist.png' %(out_folder, fname), dpi=400)
    plt.clf()

        
        
        
    if args.seg:
        if not os.path.isfile(seg_file):    
            print(f"SEGMENTATION file {seg_file} doeas not exists. Check input.")
            sys.exit(1)
        SEGM = nib.load(seg_file)
        SEGMimg = SEGM.get_fdata()
        labels = np.unique(SEGMimg)
        s = pd.DataFrame()
        s['tSNR'] = np.ravel(snr)
        s['label'] = np.ravel(SEGMimg)
        s = s[s["label"] != 0]
        sns.set_style("whitegrid")
        label_dict = {1: 'CSF',
                      2: 'GM',
                      3: 'WM',
                      4: 'BASAL GANGLIA',
                      5: 'TRUNK',
                      6: 'CEREBELLUM'} 
        s["label"] = [label_dict[num] for num in s["label"]]
        ax = sns.boxplot(data=s, y='tSNR', x='label', palette='RdYlBu', #RdYlBu
                    linewidth=1.2, saturation=1,
                    flierprops=dict(marker="o", markersize=3))
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        for i,artist in enumerate(ax.artists):
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = lighten_color(artist.get_facecolor(), 1.2)
            artist.set_edgecolor(col)   
            for j in range(i*6,i*6+6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
        sns.despine(left=True, bottom=True)
        plt.title("Segmentation tSNR Box Plot")
        plt.tight_layout()

        plt.savefig('%s/%s_SEGM_tSNRbox.png' %(out_folder, fname), dpi=400)

