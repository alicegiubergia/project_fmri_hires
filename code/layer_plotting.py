import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import ants

# Config
#%matplotlib qt
ROOT_DIR = '/mnt/store2/projects/progetto_layer/data/derivatives'
OUTPUT_DIR = '/mnt/store2/projects/progetto_layer/data/derivatives/layer_plotting'
SUB="sub-001"
ROIS=["dorsal", "ventral", "visual"]
TASK={'baseline':'spmT_0001', 'mirroring':'spmT_0008', 'spacing':'spmT_0009'}
MAPPING = {'spmT_0001':'baseline',
           'spmT_0002':'mirroring',
           'spmT_0003':'mirroring',
           'spmT_0004':'mirroring',
           'spmT_0005':'spacing',
           'spmT_0006':'spacing',
           'spmT_0007':'spacing',
           'spmT_0008':'mirroring',    
           'spmT_0009':'spacing'}

UPSAMP_SPACING = (0.2, 0.2, 0.2) 
NR_LAYERS = 10

def resample_ants(input_path, output_path, spacing, interp_type):
    print("Resampling:")
    print(f"  IN  : {input_path}")
    print(f"  OUT : {output_path}")
    print(f"  SP  : {spacing}")
    print(f"  INT : {interp_type}")
    img = ants.image_read(input_path)
    resampled_img = ants.resample_image(img, spacing, use_voxels=False, interp_type=interp_type)
    ants.image_write(resampled_img, output_path)

# Ciclo principale sui contrasti
for contrast in MAPPING.keys():
    print(f"Processing contrast: {contrast} ({MAPPING[contrast]})")
    
    # Dizionari per accumulare i dati del Line Plot per tutte le ROI di questo contrasto
    line_plot_data = {}
    all_y_values_for_lineplot = []

    # Ciclo sulle regioni
    for region in ROIS:
        print(f"  Processing region: {region}")

        # Cortical depth file
        NII_METRIC = f'{ROOT_DIR}/layer_fmri/{SUB}/{SUB}_con-{TASK[MAPPING[contrast]]}_label-{region}_metric_equidist.nii'

        # ROI mask
        NII_ROI = f'{ROOT_DIR}/rois/{SUB}/{SUB}_con-{TASK[MAPPING[contrast]]}_desc-polishCleaned_label-{region}_res-up_dseg.nii.gz'

        # fMRI activation map
        NII_ACTIVATION = glob.glob(f'{ROOT_DIR}/roi_statistics-masked/{SUB}/func/{SUB}_desc-{contrast}thresh001_statmap.nii.gz')
        NII_ACTIVATION_UP = f'{ROOT_DIR}/roi_statistics-masked/{SUB}/func/{SUB}_desc-{contrast}thresh001_res-up_statmap.nii.gz'
        if not os.path.exists(NII_ACTIVATION_UP):
            resample_ants(NII_ACTIVATION[0], NII_ACTIVATION_UP, UPSAMP_SPACING, interp_type=0)

        # Load mask data
        nii_roi = nib.load(NII_ROI)
        mask = (nii_roi.get_fdata() == 3)

        # Load metric data
        nii_metric = nib.load(NII_METRIC)
        data_x = nii_metric.get_fdata()

        nii_scalar = nib.load(NII_ACTIVATION_UP)
        if len(nii_scalar.shape) > 3:
            data_y = nii_scalar.get_fdata()[:, :, :, 0]
        else:
            data_y = nii_scalar.get_fdata()

        # Apply mask
        data_x = data_x[mask]
        data_y = data_y[mask]

        non_zero_mask = data_y != 0
        data_x = data_x[non_zero_mask]
        data_y = data_y[non_zero_mask]

        if data_y is None or len(data_y) == 0 or data_x is None or len(data_x) == 0:
            print(f"    [WARNING] data_y o data_x vuoti per {region} - {contrast}. Salto i plot individuali.")
            continue

        os.makedirs(os.path.join(OUTPUT_DIR, SUB), exist_ok=True)

        # ================================================================
        # PLOT 1: Scatter plot (Individuale per ROI)
        # ================================================================
        TITLE_SCATTER = "Scatter plot\nCortical depth layer profile\n{} - {}".format(region, contrast)
        plt.figure()
        plt.scatter(data_x, data_y, alpha=0.1, marker=".", color="black")
        plt.xlim((0, 1))
        plt.xlabel('Normalized cortical depth (0=WM, 1=CSF)')
        plt.ylabel('Activation (t-statistic)')
        plt.title(TITLE_SCATTER)
        plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{SUB}/{SUB}_scatter_{region}_{contrast}.png"), dpi=300)
        plt.close()
        # ================================================================
        # PLOT 2: 2D Histogram (Individuale per ROI)
        # ================================================================
        TITLE_HIST = "2D Histogram\nCortical depth layer profile\n{} - {}".format(region, contrast)
        plt.figure()
        plt.hist2d(data_x, data_y, bins=100, cmap='gray_r')
        plt.xlim((0, 1))
        plt.ylim(np.percentile(data_y, (1, 99)))
        plt.xlabel("Normalized cortical depth (0=WM, 1=CSF)")
        plt.ylabel("Voxel value")
        plt.title(TITLE_HIST)

        # Add colorbar
        colorbar = plt.colorbar()
        box = colorbar.ax.get_position()
        width = box.width / 2
        height = box.height / 2
        pad = 0.02
        colorbar.ax.set_position([box.x0, box.y0 + pad, width, height])
        colorbar.set_label('Voxel counts')
        plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{SUB}/{SUB}_hist_{region}_{contrast}.png"), dpi=300)
        plt.close()
        # ================================================================
        # PREPARAZIONE DATI PER PLOT 3 (Line Plot Cumulativo)
        # ================================================================
        # Quantizziamo la metrica in layer interi
        data_x_discrete = (data_x * NR_LAYERS).astype(int)
        
        # Calcoliamo la media per ogni bin di questa ROI
        data_y_mean = np.zeros(NR_LAYERS)
        for i in range(NR_LAYERS):
            if np.any(data_x_discrete == i):
                data_y_mean[i] = np.mean(data_y[data_x_discrete == i])
            else:
                data_y_mean[i] = np.nan
        
        # Salviamo il profilo medio nel dizionario usando la ROI come chiave
        line_plot_data[region] = data_y_mean
        # Accumuliamo i valori y per calcolare un range verticale sensato per il lineplot finale
        all_y_values_for_lineplot.extend(data_y)

    # ================================================================
    # PLOT 3: Line plot cumulativo (Fuori dal ciclo ROI, dentro il contrasto)
    # ================================================================

    if len(all_y_values_for_lineplot) == 0:
        print(f"[WARNING] Nessun dato valido per l'intero contrasto: {contrast}. Salto il Line Plot.")
        continue
    
    TITLE_LINE = f"Line plot\nCortical depth layer profile\nContrast: {contrast}"
    data_x_bins = np.arange(1, NR_LAYERS + 1)
    
    plt.figure(figsize=(8, 6))
    color_roi = {
        'visual': '#0f9d58',
        'ventral': '#ffab40',
        'dorsal': '#4285f4'

    }
    
    # Cicliamo sul dizionario per plottare ogni ROI memorizzata
    for region, y_mean in line_plot_data.items():
        plt.plot(data_x_bins, y_mean, marker='o', linewidth=2, label=region, color=color_roi.get(region, 'black'))
        
    plt.xlim((1, NR_LAYERS))
    plt.xticks(np.arange(1, NR_LAYERS + 1, 1))
    
    if len(all_y_values_for_lineplot) > 0:
        plt.ylim(np.percentile(all_y_values_for_lineplot, (1, 99)))
        
    plt.xlabel("Geometric layer bins (1=WM, 10=CSF)")
    plt.ylabel("Voxel value")
    plt.title(TITLE_LINE)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostra la legenda con i colori associati automaticamente alle ROI
    plt.legend(loc="best")
    plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{SUB}/{SUB}_line_{contrast}.png"), dpi=300)
    plt.close()
    