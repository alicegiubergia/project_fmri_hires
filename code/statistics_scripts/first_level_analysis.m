function first_level_analysis(mode, sub)

% MAKE AND ESTIMATE MODEL
% Make model using experimental logfiles/timings
% Estimate model to create SPM file for first level analysis
% Create contrasts
% Results manager

% INPUT ARGUMENTS:

% 'mode'
% m = make model (Basic GLM)
% a = artifact detection
% e = estimate model    
% c = contrasts manager 
% r = results report    

spm('Defaults', 'FMRI');
currentFolder = fileparts(mfilename('fullpath'));
batch_files = fullfile(currentFolder, 'batch_files');

id = sub.id;
D = sub.path;
func_dir = fullfile(D, 'func');
physio_dir = fullfile(sub.physio, 'func');
stats_dir = fullfile(D,'stats-masked');

% get number of runs
list_files = dir(fullfile(func_dir, '*_run-*_bold.json'));
num_runs = length(list_files);
fprintf('Identificate %d run BIDS.\n', num_runs);
nii_files = cellstr(spm_select('FPList', func_dir, ['^sub-' id '.*_run-.*_space-anat_desc-preproc_bold.nii$']));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make model: Basic GLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode,'m')
    % if folder for stats doesnt exist, create the folder
    if ~exist(stats_dir,'dir'); cd(D); mkdir(stats_dir); end    

    load(fullfile(batch_files, 'design.mat'), 'matlabbatch');

    %%%%%% Global setup for model
    matlabbatch{1}.spm.stats.fmri_spec.dir = {stats_dir}; %Output Directory
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 3; % Repetition Time
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    % set explicit mask 
    matlabbatch{1}.spm.stats.fmri_spec.mask = {sub.mask};

    if num_runs > 1
        for i=2:num_runs
            matlabbatch{1}.spm.stats.fmri_spec.sess(i) = matlabbatch{1}.spm.stats.fmri_spec.sess(1);
            matlabbatch{1}.spm.stats.fmri_spec.sess(i).scans = [];
            matlabbatch{1}.spm.stats.fmri_spec.sess(i).cond = [];
        end
    end
    %%%%%% Filling in scans and conditions for each session
    for ss=1:num_runs %number of sessions
        
        % Loading up pre-processed scans 
        P = cellstr(spm_select('ExtFPList', func_dir, ['^sub-' id '.*_run-' sprintf('%02d', ss) '.*_space-anat_desc-preproc_bold.nii$'], Inf));
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(ss).scans={};
        for ii=1:size(P, 1)
            matlabbatch{1}.spm.stats.fmri_spec.sess(ss).scans{ii,1} = P{ii};
        end
        clear P;       

        ev = events_onsets_duration(sub.log{ss}, func_dir);       
        matlabbatch{1}.spm.stats.fmri_spec.sess(ss).cond = ev;
        clear ev
    end
    spm_jobman('run', matlabbatch);
end

if contains(mode, 'a')
    statsfile = fullfile(stats_dir, 'SPM.mat');
    stats_exists = exist(statsfile, 'file');
    
    % ART
    cd(stats_dir)
    %art_fmri(statsfile, sub.mask)
    par_files = cellstr(spm_select('FPList', func_dir, '^sub-.*_task-reading_run-.*_motion.par'));
    
    %%%%%%%%%%%% Configure ART settings %%%%%%%%%%%%
    cfg = [];
    cfg.P = nii_files;
    cfg.M = par_files;
    cfg.num_sess = num_runs;
    cfg.global_mean = 1; % global mean type (1: Standard 2: User-defined Mask)
    cfg.motion_file_type = 1; %  motion file type (0: SPM .txt file 1: FSL .par file 2:Siemens .txt file)
    cfg.motion_threshold = 1.0; % threshold for outlier detection based on motion estimates
    cfg.global_threshold = 2.0; % threshold for outlier detection based on global signal
    cfg.use_diff_motion = 1; % 1: uses scan-to-scan motion to determine outliers; 0: uses absolute motion
    cfg.use_diff_global = 1; % 1: uses scan-to-scan global signal change to determine outliers; 0: uses absolute global signal values
    cfg.use_norms = 1; % 1: uses composite motion measure (largest voxel movement) to determine outliers; 0: uses raw motion measures (translation/rotation parameters) 
    if stats_exists
        cfg.spm_file = statsfile;
    end
    cfg.mask_file = sub.mask;
    cfg.output_dir = func_dir; 
    cfg.motion_fname_from_image_fname = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    art('sess_file', cfg)

    if stats_exists
        movefile(statsfile, fullfile(stats_dir,'SPM_no_art.mat'));
    
        % Update matlabbatch
        for ss=1:num_runs %number of sessions
            run_str = sprintf('%02d', ss);
    
            % Il punto '.' prima di 'run' serve a catturare qualsiasi carattere (spesso un underscore)
            pattern = ['^art.*outliers_and_movement.*run-', run_str, '.*\.mat$'];
    
            % 1. Locate the movement and outlier regressors file created by ART
            art_file = spm_select('FPList', func_dir, pattern);
            [~, art_filename, ~] = fileparts(art_file);
            bids_core = regexp(art_filename, 'sub-.*_task-.*_run-\d+', 'match', 'once');
            out_name = fullfile(func_dir, [bids_core, '_desc-confounds.mat']);
            
            % 2. Locate the CSV files containing additional regressors (e.g., physio)·
            hrv_file = spm_select('FPList', physio_dir, ['^sub-.*run-',run_str , '.*cardiac.HRV_resampled.*\.csv$']);
            rv_file = spm_select('FPList', physio_dir, ['^sub-.*run-',run_str , '.*respiratory.RV_resampled.*\.csv$']);
            
            if ~isempty(hrv_file) && ~isempty(rv_file)
                % Note: ART and SPM typically store the regressor matrix in a variable named 'R'
                data_mat = load(art_file);
                R_art = data_mat.R;
                % Load numerical data from the file
                R_hrv = readmatrix(hrv_file);
                R_rv = readmatrix(rv_file);
    
                % --- DIMENSION CHECK ---
                % Ensure both matrices have the same number of rows (fMRI volumes)
                if (size(R_art,1) == size(R_hrv,1)) &  size(R_hrv,1) == size(R_rv,1)
                    % Horizontally concatenate ART regressors and CSV columns
                    R = [R_art, R_hrv, R_rv];
                    % Define the output path for the combined regressor file
                    save(out_name, 'R');
                    P = {out_name};
                else
                    % Stop execution if the number of volumes does not match
                    error('Session %d Error: HRV (%d rows), RV (%d rows) and ART (%d rows) length mismatch!', ...
                          ss, size(R_hrv, 1), size(R_rv, 1), size(R_art, 1));
                end
            else
                warning('Missing files in session %d. Checking for ART .mat and CSV...', ss);
                P = cellstr(art_file);
            end
    
            matlabbatch{1}.spm.stats.fmri_spec.sess(ss).multi_reg = P;
    
            clear P R R_art R_hrv R_rv data_mat out_name;  
    
        end
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf;
        save(fullfile(stats_dir, 'design.mat'), 'matlabbatch');
    
        spm_jobman('run', matlabbatch);
    end
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode,'e')
    
    statsfile = fullfile(stats_dir, 'SPM.mat');
    
    load(fullfile(batch_files, 'estimate.mat'), 'matlabbatch');
    
    % Set the right SPM.mat file
    matlabbatch{1}.spm.stats.fmri_est.spmmat={statsfile};
    save(fullfile(stats_dir, 'estimate.mat'), 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contrast Manager
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if contains(mode,'c')
    
    statsfile=fullfile(stats_dir,'SPM.mat');    
    
    load(fullfile(batch_files,'contrast_manager.mat'), 'matlabbatch');
    
    matlabbatch{1}.spm.stats.con.spmmat={statsfile};
    
    load(statsfile, 'SPM');
    [Ntt, Nregr] = size(SPM.xX.X);
    
    baseline_idx = find(contains(SPM.xX.name, 'baseline'));
    flip_idx = find(contains(SPM.xX.name, 'flip'));
    spacing_idx = find(contains(SPM.xX.name, 'spacing'));
    
    % baseline > 0
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'Normally displayed words';         % t Contrast (f contrast = fcon)
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights(baseline_idx) = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none'; 
    
    % Mirroring LRUD > baseline
    w_flip_LRUD = contains(SPM.xX.name, 'flip_LRUD*')/sum(contains(SPM.xX.name, 'flip_LRUD*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'Mirroring LRUD > baseline';         % t Contrast (f contrast = fcon)
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = w_flip_LRUD;
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';  
    
    % Mirroring LR > baseline
    w_flip_LR = contains(SPM.xX.name, 'flip_LR*')/sum(contains(SPM.xX.name, 'flip_LR*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));    
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'Mirroring LR > baseline';              
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = w_flip_LR;    
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';  
    
    % Mirroring UD > baseline
    w_flip_UD = contains(SPM.xX.name, 'flip_UD*')/sum(contains(SPM.xX.name, 'flip_UD*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));   
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'Mirroring UD > baseline';              
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = w_flip_UD;    
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';  
    
    % Spacing 2 > baseline
    w_space_2 = contains(SPM.xX.name, 'spacing_2*')/sum(contains(SPM.xX.name, 'spacing_2*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'Spacing 2 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = w_space_2;
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';

    
    % Spacing 4 > baseline
    w_space_4 = contains(SPM.xX.name, 'spacing_4*')/sum(contains(SPM.xX.name, 'spacing_4*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'Spacing 4 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = w_space_4;
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';

    % Spacing 6 > baseline
    w_space_6 = contains(SPM.xX.name, 'spacing_6*')/sum(contains(SPM.xX.name, 'spacing_6*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'Spacing 6 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = w_space_6;   
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';  
    
    matlabbatch{1}.spm.stats.con.consess={};
    % Mirroring (cumulative) > baseline
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'Mirroring (all) > baseline';              
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights(flip_idx) = 1/length(flip_idx);
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights(baseline_idx) = -1/length(baseline_idx);
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none'; 

    % Spacing (cumulative) > baseline
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'Spacing (all) > baseline';              
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights(spacing_idx) = 1/length(spacing_idx);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights(baseline_idx) = -1/length(baseline_idx);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none'; 

    spm_jobman('run',matlabbatch);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Results Report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode, 'r')
    
    statsfile=fullfile(stats_dir,'SPM.mat');

    load(fullfile(batch_files,'results_report.mat'), 'matlabbatch');
    
    matlabbatch{1}.spm.stats.results.spmmat={statsfile};
    load(statsfile, 'SPM')

    %% --- 1. NON CORRETTO: p < 0.001 voxel-level ---
    for i=1:length(SPM.xCon)
        % Which contrast files to report
        matlabbatch{1}.spm.stats.results.conspec(i).titlestr = [SPM.xCon(i).name ' - uncorrected'];
        matlabbatch{1}.spm.stats.results.conspec(i).contrasts = i;
        matlabbatch{1}.spm.stats.results.conspec(i).conjunction = 1;
        matlabbatch{1}.spm.stats.results.conspec(i).threshdesc = 'none';    % Test type
        matlabbatch{1}.spm.stats.results.conspec(i).thresh = 0.001;         % P Value
        matlabbatch{1}.spm.stats.results.conspec(i).extent = 0;             % Number of voxels per cluster
        matlabbatch{1}.spm.stats.results.conspec(i).mask.none = 1;

    end
    
    matlabbatch{1}.spm.stats.results.units = 1;                             
    matlabbatch{1}.spm.stats.results.export{1}.ps = 1;                      % Export results as postscript
    matlabbatch{1}.spm.stats.results.export{2}.csv = true;                  % Export results as csv
    matlabbatch{1}.spm.stats.results.export{3}.tspm.basename = 'uncorrected';

    save(fullfile(stats_dir, 'results_report.mat'), 'matlabbatch');
    spm_jobman('run',matlabbatch(1));

    %% --- 2. FWE CORRETTO: p < 0.05 cluster-level ---
    matlabbatch{2}.spm.stats.results.spmmat={statsfile};

    csvFile = dir(fullfile(stats_dir, "*.csv"));
    

    for i=1:length(SPM.xCon)
        T = readtable(fullfile(stats_dir, csvFile(i).name), 'NumHeaderLines',1);
        k_fwe = min(T(T.p_FWE_corr_<0.05,"equivk")).equivk;
        if isempty(k_fwe); k_fwe = Inf;end

        % Which contrast files to report
        matlabbatch{2}.spm.stats.results.conspec(i).titlestr = [SPM.xCon(i).name ' - FWE_cluster'];
        matlabbatch{2}.spm.stats.results.conspec(i).contrasts = i;
        matlabbatch{2}.spm.stats.results.conspec(i).conjunction = 1;
        matlabbatch{2}.spm.stats.results.conspec(i).threshdesc = 'none';    % Test type
        matlabbatch{2}.spm.stats.results.conspec(i).thresh = 0.001;         % P Value
        matlabbatch{2}.spm.stats.results.conspec(i).extent = k_fwe;         % Number of voxels per cluster
        matlabbatch{2}.spm.stats.results.conspec(i).mask.none = 1;

    end
    
    matlabbatch{2}.spm.stats.results.units = 1;                             
    matlabbatch{2}.spm.stats.results.export{1}.ps = 1;                      % Export results as postscript
    matlabbatch{2}.spm.stats.results.export{2}.csv = false;                  % Export results as csv
    matlabbatch{2}.spm.stats.results.export{3}.nary.basename = 'bin'; 
    matlabbatch{2}.spm.stats.results.export{4}.tspm.basename = 'FWEc';

    save(fullfile(stats_dir, 'results_report.mat'), 'matlabbatch');
    spm_jobman('run',matlabbatch(2));

end 

return

    


