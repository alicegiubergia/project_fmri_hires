import subprocess
import nibabel as nib
from scipy import ndimage
import numpy as np

pipeline_path = "/mnt/store2/projects/progetto_layer/data/derivatives/spm_masks"
sub = "sub-001"
t1_dir = f"{pipeline_path}/{sub}/anat"
t1_mask_path = f"{t1_dir}/{sub}_temp_ants_mask.nii.gz"
output_fn = f"{t1_dir}/{sub}_output_clean.nii.gz"

img = nib.load(t1_mask_path)
mask = img.get_fdata().astype(bool)

import nibabel as nib
import numpy as np
from scipy import ndimage

def select_brain_component(labeled, n_components, mask_shape):
    """Seleziona il componente cerebrale usando densità nella sfera centrale."""
    center = np.array(mask_shape) / 2
    radius = min(mask_shape) // 3

    zz, yy, xx = np.ogrid[:mask_shape[0], :mask_shape[1], :mask_shape[2]]
    sphere = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 <= radius**2

    best_label, best_count = None, -1
    for label in range(1, n_components + 1):
        comp = (labeled == label)
        count = np.sum(comp & sphere)
        print(f"  Label {label}: {count} voxel nella sfera centrale")
        if count > best_count:
            best_count = count
            best_label = label

    print(f"[INFO]  Scelto label {best_label} ({best_count} voxel nella sfera)")
    return (labeled == best_label)


def clean_brain_mask(mask_path, output_path, n_erosions_max=20):
    img = nib.load(mask_path)
    mask = img.get_fdata().astype(bool)
    struct = ndimage.generate_binary_structure(3, 2)

    # --- Caso 1: già più componenti, nessuna erosione necessaria ---
    labeled, n = ndimage.label(mask)
    print(f"[INFO]  Componenti iniziali: {n}")

    if n >= 2:
        print("→ Caso 1: componenti già separati, seleziono direttamente.")
        clean_mask = select_brain_component(labeled, n, mask.shape)

    # --- Caso 2: tutto connesso, erodo finché non si separano ---
    else:
        print("→ Caso 2: unico componente, erodo per separare cervello e corona...")
        clean_mask = None

        for i in range(1, n_erosions_max + 1):
            eroded = ndimage.binary_erosion(mask, structure=struct, iterations=i)
            labeled_e, n_e = ndimage.label(eroded)
            print(f"  Erosione {i:2d}: {n_e} componenti, {eroded.sum()} voxel rimasti")

            if n_e >= 2:
                print(f"→ Separazione ottenuta a erosione {i}!")
                seed_mask = select_brain_component(labeled_e, n_e, mask.shape)

                # Ridilata per recuperare il volume originale
                clean_mask = ndimage.binary_dilation(seed_mask, structure=struct, iterations=i)
                clean_mask = ndimage.binary_fill_holes(clean_mask)
                break

        if clean_mask is None:
            print("[ERROR]  Impossibile separare i componenti anche con erosione massima.")
            print("         Prova ad aumentare n_erosions_max.")
            return None

    # Fill holes finali
    clean_mask = ndimage.binary_fill_holes(clean_mask)

    # Salva
    out_img = nib.Nifti1Image(clean_mask.astype(np.uint8), img.affine, img.header)
    nib.save(out_img, output_path)

    removed = mask.sum() - clean_mask.sum()
    print(f"\n[INFO]  Voxel rimossi: {removed} ({100*removed/mask.sum():.1f}%)")
    print(f"[SUCCESS]  Salvato in: {output_path}")
    return clean_mask


# Uso
clean_brain_mask("brain_mask.nii.gz", "brain_mask_clean.nii.gz", n_erosions_max=20)