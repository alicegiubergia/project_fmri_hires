% subject independent information such as study paths
% the scripts assume this setup for your data. 

%% Setup Paths
base_path = '/mnt/store1/projects/progetto_layer';
addpath('/home/matuser/matlab_toolbox/spm12');
addpath('/home/matuser/matlab_toolbox/ART');
addpath(fullfile(base_path, 'code/statistics_scripts'));
addpath(fullfile(base_path, 'code/statistics_scripts/batch_files/'));

%% Subject Configuration
% List of subject IDs to process
subject_ids = {'001'};
num_runs = 3; % Number of functional runs per subject

% Initialize subject structure
sub = struct();

for s = 1:length(subject_ids)
    id = subject_ids{s};

    % Main subject info
    sub(s).id = id;
    
    % Derivative path 
    sub(s).path = fullfile(base_path, 'data/derivatives/hires_pipeline', ['sub-', id]);

    % Physio data path
    sub(s).physio = fullfile(base_path, 'data/derivatives/physio', ['sub-', id]);

    % Anatomical mask
    sub(s).mask = fullfile(base_path, 'data/derivatives/spm_masks_b-2', ['sub-', id], 'anat', ['sub-', id, '_desc-spm_mask.nii']);

    % Loop through runs to find event log files (.tsv)
    for r = 1:num_runs
        run_str = sprintf('%02d', r);
        sub(s).log{r} = fullfile(base_path, 'data', ['sub-', id], 'func', ['sub-', id, '_task-reading_run-', run_str, '_events.tsv']);
    end
end

%% Parallel Setup (2 Workers)
if isempty(gcp('nocreate'))
    parpool('local', 2); % Limits to 2 subjects at a time to save RAM
end

%% Execute Analysis

fprintf('\n--- Starting Batch Processing for %d subjects ---\n', length(sub));

parfor i = 1:length(sub)
    % Initialize SPM for each parallel worker
    spm('defaults','fmri');
    spm_jobman('initcfg');

    fprintf('\n🚀 Elaborazione Soggetto sub-%s (%d di %d)\n', sub(i).id, i, length(sub));
    
    % Sanity check: verify if the subject directory exists
    if ~exist(sub(i).path, 'dir')
        warning('Directory not found for sub-%s. Skipping ...', sub(i).id)
        continue;
    end

    try
        % Launching the GLM function
        fprintf('  --- First level analysis: GLM --- \n');
        first_level_analysis('c', sub(i)); 
        fprintf('✅ Successfully processed sub-%s\n', sub(i).id);
    catch ME
        % Error handling: if one subject fails, the script continues with
        % the next
        fprintf('❌ Error processing sub-%s: %s\n', sub(i).id, ME.message);
    end
end
delete(gcp('nocreate')); % Close parallel pool
fprintf('\n--- Batch processing completed ---\n');
exit;