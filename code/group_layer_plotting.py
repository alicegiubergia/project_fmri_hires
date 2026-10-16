import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ants

ROOT_DIR = '/mnt/store1/projects/progetto_layer/data/derivatives'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'group_layer_plotting')

SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008"] 

ROIS = ["dorsal", "ventral", "visual"]

TASK = {'baseline': 'spmT_0001', 'mirroring': 'spmT_0008', 'spacing': 'spmT_0009'}
MAPPING = {
    'spmT_0001': 'baseline',
    'spmT_0002': 'mirroring',
    'spmT_0003': 'mirroring',
    'spmT_0004': 'mirroring',
    'spmT_0005': 'spacing',
    'spmT_0006': 'spacing',
    'spmT_0007': 'spacing',
    'spmT_0008': 'mirroring',    
    'spmT_0009': 'spacing'
}

UPSAMP_SPACING = (0.2, 0.2, 0.2) 
NR_LAYERS = 10

COLOR_ROI = {
    'visual': '#0f9d58',   # Verde
    'ventral': '#ffab40',  # Arancione
    'dorsal': '#4285f4'    # Blu
}

def resample_ants(input_path, output_path, spacing, interp_type=0):
    """Esegue il resample dell'immagine via ANTs se non esiste già."""
    print(f"  [Resampling ANTs] -> {os.path.basename(output_path)}")
    img = ants.image_read(input_path)
    resampled_img = ants.resample_image(img, spacing, use_voxels=False, interp_type=interp_type)
    ants.image_write(resampled_img, output_path)


def extract_subject_profile(sub, contrast, region):
    """Estrae il profilo di attivazione per layer di un singolo soggetto."""
    task_key = TASK[MAPPING[contrast]]
    
    nii_metric_path = f'{ROOT_DIR}/layer_fmri/{sub}/{sub}_con-{task_key}_label-{region}_metric_equidist.nii'
    nii_roi_path = f'{ROOT_DIR}/rois/{sub}/{sub}_con-{task_key}_desc-polishCleaned_label-{region}_res-up_dseg.nii.gz'
    
    activation_files = glob.glob(f'{ROOT_DIR}/roi_statistics-masked/{sub}/func/{sub}_desc-{contrast}thresh001_statmap.nii.gz')
    if not activation_files:
        print(f"    [WARNING] Mappa attivazione non trovata: {sub} | {contrast} | {region}")
        return None
        
    nii_act_path = activation_files[0]
    nii_act_up_path = f'{ROOT_DIR}/roi_statistics-masked/{sub}/func/{sub}_desc-{contrast}thresh001_res-up_statmap.nii.gz'
    
    # Resampling ANTs della mappa funzionale se non presente
    if not os.path.exists(nii_act_up_path):
        resample_ants(nii_act_path, nii_act_up_path, UPSAMP_SPACING, interp_type=0)

    # Check esistenza file essenziali
    if not os.path.exists(nii_metric_path) or not os.path.exists(nii_roi_path):
        print(f"    [WARNING] File ROI o Metrica mancante: {sub} | {region}")
        return None

    # Caricamento e masking in gray matter
    mask = (nib.load(nii_roi_path).get_fdata() == 3)
    data_x = nib.load(nii_metric_path).get_fdata()[mask]
    
    act_img = nib.load(nii_act_up_path)
    data_y = act_img.get_fdata()[:, :, :, 0][mask] if len(act_img.shape) > 3 else act_img.get_fdata()[mask]

    # Filtraggio valori zero/nulli
    non_zero = data_y != 0
    data_x, data_y = data_x[non_zero], data_y[non_zero]

    if len(data_y) == 0:
        return None

    # Quantizzazione nei Bins di Layer
    data_x_discrete = (data_x * NR_LAYERS).astype(int)
    profile = np.full(NR_LAYERS, np.nan)
    
    for i in range(NR_LAYERS):
        in_layer = (data_x_discrete == i)
        if np.any(in_layer):
            profile[i] = np.mean(data_y[in_layer])

    return profile

# PIPELINE 
os.makedirs(OUTPUT_DIR, exist_ok=True)

for contrast in MAPPING.keys():
    print(f"\n==================================================")
    print(f" Processing contrast: {contrast} ({MAPPING[contrast]})")
    print(f"==================================================")

    plt.figure(figsize=(9, 6))
    data_x_bins = np.arange(1, NR_LAYERS + 1)
    has_plot_data = False

    for region in ROIS:
        print(f" -> Collecting group data for region: {region}")
        
        # Array per raccogliere i profili di tutti i soggetti (N_subjects x NR_LAYERS)
        group_subject_profiles = []

        for sub in SUBJECTS:
            prof = extract_subject_profile(sub, contrast, region)
            if prof is not None:
                group_subject_profiles.append(prof)
            else:
                print(f"    [SKIP] Missing data per {sub}")

        if len(group_subject_profiles) == 0:
            print(f"  [WARNING] Nessun soggetto valido per ROI {region} in {contrast}.")
            continue

        # Conversione in matrice Numpy [N_soggetti, 10]
        group_matrix = np.array(group_subject_profiles)
        
        # Calcolo Statistica di Gruppo (Media e SD per ogni layer)
        mean_profile = np.nanmean(group_matrix, axis=0)

        # SD 
        std_profile = np.nanstd(group_matrix, axis=0, ddof=1) # ddof=1 per la SD campionaria

        # Errore Standard 
        std_profile = std_profile / np.sqrt(np.sum(~np.isnan(group_matrix), axis=0))

        color = COLOR_ROI.get(region, 'black')
        n_subs_effective = group_matrix.shape[0]

        # 1. Plot della linea della Media di Gruppo
        plt.plot(
            data_x_bins, 
            mean_profile, 
            marker='o', 
            linewidth=2.5, 
            label=f"{region} (N={n_subs_effective})", 
            color=color
        )
        
        # 2. Plot dell'area di incertezza (+/- 1 SD)
        plt.fill_between(
            data_x_bins, 
            mean_profile - std_profile, 
            mean_profile + std_profile, 
            color=color, 
            alpha=0.20,  # Trasparenza dell'ombra
            edgecolor='none'
        )
        
        has_plot_data = True

    # Definizione e salvataggio del Grafico Finale di Gruppo
    if has_plot_data:
        TITLE_LINE = f"Group mean layer profile (± SE)\nContrast: {contrast} ({MAPPING[contrast]})"
        plt.xlim((1, NR_LAYERS))
        plt.xticks(np.arange(1, NR_LAYERS + 1, 1))
        plt.xlabel("Geometric layer bins (1=WM, 10=CSF)", fontsize=11)
        plt.ylabel("Activation (t-statistic)", fontsize=11)
        plt.title(TITLE_LINE, fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc="best", frameon=True)
        
        save_path = os.path.join(OUTPUT_DIR, f"group_line_{contrast}_{MAPPING[contrast]}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
        print(f" -> Grafico di gruppo salvato in: {save_path}")
    else:
        plt.close()
        print(f" [WARNING] Nessun dato valido plottato per il contrasto {contrast}.")