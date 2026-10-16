#! /bin/sh

baseDirectory=${PWD}

#fixedImage=${baseDirectory}/epi_mean.nii.gz
#movingImage=${baseDirectory}/../../anat/T1w_N4.nii.gz
#maskImage=${baseDirectory}/epi_mask.nii.gz

fixedImage=$1
movingImage=$2
SUBJECT_ID=$(basename "$(dirname "$(dirname "$movingImage")")")


fixedImageN4=${fixedImage%.nii.gz}_N4_reg.nii
maskImage=${fixedImage%.nii.gz}_mask_reg.nii.gz
outDirectory=$(dirname ${fixedImage})

# make mask of slab image
#python getMask.py ${fixedImage} 
N4BiasFieldCorrection -d 3 -i $fixedImage -o $fixedImageN4 -b [100] -c [200x200x200x200,0] -v 1 -s 4
ThresholdImage 3 ${fixedImageN4} ${maskImage} 1000 Inf

# Register T1 to EPI slab with mask
outputPrefix=${outDirectory}/${SUBJECT_ID}_from-T1w_to-epiSlab

antsRegistration --verbose 1 \
                 --dimensionality 3 \
                 --float 0 \
                 --interpolation Linear \
                 --use-histogram-matching 0 \
                 --winsorize-image-intensities [0.005,0.995] \
                 --output [${outputPrefix},${outputPrefix}Warped.nii.gz,${outputPrefix}InverseWarped.nii.gz] \
                 --collapse-output-transforms 1 \
                 --initial-moving-transform [${fixedImageN4},${movingImage},1] \
                 --transform translation[0.2] \
                   --metric MI[${fixedImageN4},${movingImage},1,32,Random,0.5] \
                   --convergence [400x200x100,1e-6,10] \
                   --shrink-factors 8x4x2 \
                   --smoothing-sigmas 0x0x0vox \
                   --masks [${maskImageN4},NULL] \
                 --transform Rigid[0.2] \
                   --metric MI[${fixedImageN4},${movingImage},1,32,Random,0.5] \
                   --convergence [500x500x250x50,1e-6,10] \
                   --shrink-factors 6x4x2x1 \
                   --smoothing-sigmas 2x1x0x0vox \
                   --masks [${maskImage},NULL]
