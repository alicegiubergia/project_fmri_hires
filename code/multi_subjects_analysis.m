% subject independent information such as study paths
% the scripts assume this setup for your data. 

addpath('/home/matuser/matlab_toolbox/spm12')
addpath('/home/matuser/matlab_toolbox/ART')
addpath('/mnt/store2/projects/progetto_layer/code')
addpath('/mnt/store2/projects/progetto_layer/code/batch_files/')

sub(1).id='001';
sub(1).path = ['/mnt/store2/projects/progetto_layer/data/derivatives/mypipeline/sub-' sub(1).id];
sub(1).log{1} = ['/mnt/store2/projects/progetto_layer/data/sub-' sub(1).id '/func/sub-' sub(1).id '_task-reading_run-01_events.tsv'];
sub(1).log{2} = ['/mnt/store2/projects/progetto_layer/data/sub-' sub(1).id '/func/sub-' sub(1).id '_task-reading_run-02_events.tsv'];
sub(1).log{3} = ['/mnt/store2/projects/progetto_layer/data/sub-' sub(1).id '/func/sub-' sub(1).id '_task-reading_run-03_events.tsv'];
sub(1).mask = ['/mnt/store2/projects/progetto_layer/data/derivatives/mypipeline/sub-' sub(1).id '/anat/' sub(1).id '_desc-brain_mask.nii'];

for i = 1:length(sub)
    spm('defaults','fmri');
    spm_jobman('initcfg');
        
    %% First Level Analysis ('maecr')
    fprintf('  --- First level analysis: GLM --- \n');
    first_level_analysis('r', sub(i)); 
end