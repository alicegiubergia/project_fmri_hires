#! /bin/sh


baseDirectory=~/medea/progetto_reading/derivatives/mypipeline/sub-00/func/run-01

fixedImage=${baseDirectory}/epi_st_topup_mc_mean_reg.nii.gz
movingImage=$(dirname ${fixedImage})/T2xEPIslab/t2_masked.nii.gz
fixedImageN4=${fixedImage%.nii.gz}_N4.nii
movingImageN4=$(dirname ${fixedImage})/T2xEPIslab/t2_masked_N4.nii.gz

fixedMask=${fixedImage%.nii.gz}_mask.nii.gz


outDirectory=$(dirname ${fixedImage})/T2xEPIslab

# make mask of slab image
#python getMask.py ${fixedImage} 
[[ ! -f $fixedImageN4 ]] && N4BiasFieldCorrection -d 3 -i $fixedImage -o $fixedImageN4 -b [100] -c [200x200x200x200,0] -v 1 -s 4
ThresholdImage 3 ${fixedImageN4} ${fixedMask} 1000 Inf

[[ ! -f $movingImageN4 ]] && N4BiasFieldCorrection -d 3 -i $movingImage -o $movingImageN4 -b [100] -c [200x200x200x200,0] -v 1 -s 4

# Register T1 to EPI slab with mask
outputPrefix=${outDirectory}/T2xEPIslab

antsRegistration --verbose 1 \
                 --dimensionality 3 \
                 --float 0 \
                 --interpolation Linear \
                 --use-histogram-matching 0 \
                 --winsorize-image-intensities [0.005,0.995] \
                 --output [${outputPrefix},${outputPrefix}Warped.nii.gz,${outputPrefix}InverseWarped.nii.gz] \
                 --initial-moving-transform [${fixedImageN4},${movingImageN4},1] \
                 --transform translation[0.2] \
                   --metric MI[${fixedImageN4},${movingImageN4},1,32,Random,0.5] \
                   --convergence [400x200x100,1e-6,10] \
                   --shrink-factors 8x4x2 \
                   --smoothing-sigmas 0x0x0vox \
                   --masks [${fixedMask},] \
                 --transform Rigid[0.2] \
                   --metric MI[${fixedImageN4},${movingImageN4},1,32,Random,0.5] \
                   --convergence [500x500x250x50,1e-6,10] \
                   --shrink-factors 6x4x2x1 \
                   --smoothing-sigmas 2x1x0x0vox \
                   --masks [${fixedMask},]
