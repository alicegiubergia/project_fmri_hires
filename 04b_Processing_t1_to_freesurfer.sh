#!/bin/bash
###############################################################################
# This script aligns a native-space T1-weighted image to the subject’s 
# FreeSurfer anatomical space (T1.mgz) using FreeSurfer 8 inside Docker.
#
# Workflow:
# 1. bbregister estimates a linear registration (native T1 → FreeSurfer space)
#    and stores it as a .dat transform.
# 2. lta_convert converts the .dat file into an .lta file so that it can be
#    used with mri_vol2vol.
# 3. mri_vol2vol applies the transform to bring the skull-stripped T1 mask 
#    into FreeSurfer space, producing a .mgz output.
# 4. The original brainmask.mgz is backed-up, and the transformed mask is 
#    incorporated using mri_mask.
#
# The script is compatible with BIDS-derivatives conventions and allows 
# integrating a native-space T1 (NIfTI) into the subject’s FreeSurfer directory.
###############################################################################

# Controllo che sia stato fornito un argomento
if [ $# -ne 1 ]; then
    echo "Usage: $0 <subject_id>"
    exit 1
fi


FS_DOCKER_IMAGE="cesne/freesurfer:8.0"
SUBJECTS_DIR=/mnt/store2/projects/progetto_layer/data/derivatives/freesurfer
MYPIPELINE_DIR=/mnt/store2/projects/progetto_layer/data/derivatives/mypipeline
subject=$1

docker run --user $(id -u):$(id -g)  \
	-v ${SUBJECTS_DIR}:/subjects \
	-v ${MYPIPELINE_DIR}:/data \
	-e SUBJECTS_DIR=/subjects \
	${FS_DOCKER_IMAGE} bbregister \
	--s $subject \
	--mov /data/$subject/anat/${subject}_desc-N4_T1w.nii.gz \
	--reg /data/$subject/anat/${subject}_desc-N4_T1w_2fs.dat \
	--init-coreg \
	--t1

docker run --user $(id -u):$(id -g)  \
	-v ${SUBJECTS_DIR}:/subjects \
	-v ${MYPIPELINE_DIR}:/data \
	${FS_DOCKER_IMAGE} lta_convert \
	--src /data/$subject/anat/${subject}_desc-N4_T1w.nii.gz \
	--trg /subjects/$subject/mri/T1.mgz \
	--inreg /data/$subject/anat/${subject}_desc-N4_T1w_2fs.dat \
	--outlta /data/$subject/anat/${subject}_desc-N4_T1w_2fs.lta \


docker run --user $(id -u):$(id -g)  \
	-v ${SUBJECTS_DIR}:/subjects \
	-v ${MYPIPELINE_DIR}:/data \
	${FS_DOCKER_IMAGE} mri_vol2vol \
	--mov /data/$subject/anat/${subject}_desc-brain_mask.nii.gz \
	--targ /subjects/$subject/mri/T1.mgz \
    --o /data/$subject/anat/${subject}_desc-brain_mask_in_fs.mgz \
    --lta /data/$subject/anat/${subject}_desc-N4_T1w_2fs.lta \
    --interp nearest

 
cp ${SUBJECTS_DIR}/$subject/mri/brainmask.mgz ${SUBJECTS_DIR}/$subject/mri/brainmask_before_edits.mgz 


docker run --user $(id -u):$(id -g)  \
	-v ${SUBJECTS_DIR}:/subjects \
	-v ${MYPIPELINE_DIR}:/data \
	${FS_DOCKER_IMAGE} mri_mask \
	/subjects/$subject/mri/brainmask.mgz \
	/data/$subject/anat/${subject}_desc-brain_mask_in_fs.mgz \
	/subjects/$subject/mri/brainmask.mgz 

