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
sesspath = dir(fullfile(D, 'func','run-*'));
numsess = length(sesspath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make model: Basic GLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode,'m')
    % if folder for stats doesnt exist, create the folder
    if ~exist(fullfile(D,'stats'),'dir'); cd(D); mkdir('stats'); end    
    stats_directory=fullfile(D,'stats');

    load(fullfile(batch_files, 'design.mat'), 'matlabbatch');

    %%%%%% Global setup for model
    matlabbatch{1}.spm.stats.fmri_spec.dir = {stats_directory}; %Output Directory
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 3; % Repetition Time
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    if numsess > 1
        for i=2:numsess
            matlabbatch{1}.spm.stats.fmri_spec.sess(i) = matlabbatch{1}.spm.stats.fmri_spec.sess(1);
            matlabbatch{1}.spm.stats.fmri_spec.sess(i).scans = [];
            matlabbatch{1}.spm.stats.fmri_spec.sess(i).cond = [];
        end
    end
    %%%%%% Filling in scans and conditions for each session
    for ss=1:numsess %number of sessions
        directory = fullfile(D, 'func', sesspath(ss).name);
        fprintf(directory);
        
        % Loading up pre-processed scans 
        P = cellstr(spm_select('ExtFPList', directory, ['^sub-' id '_run-' sprintf('%02d', ss) '_desc-preproc_space-anat_bold.nii$'], Inf));
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(ss).scans={};
        for ii=1:size(P, 1)
            matlabbatch{1}.spm.stats.fmri_spec.sess(ss).scans{ii,1} = P{ii};
        end
        clear P;       

        ev = events_onsets_duration(sub.log{ss}, directory);       
        matlabbatch{1}.spm.stats.fmri_spec.sess(ss).cond = ev;
        clear ev
    end
    spm_jobman('run', matlabbatch);
end

if contains(mode, 'a')
    stats_directory = fullfile(D, 'stats');
    statsfile = fullfile(stats_directory, 'SPM.mat');
    
    % ART
    cd(stats_directory)
    art_fmri(statsfile, sub.mask)
 
    movefile(statsfile, fullfile(stats_directory,'SPM_no_art.mat'));

    % Update matlabbatch
    for ss=1:numsess %number of sessions
        directory = fullfile(D, 'func', sesspath(ss).name);

        % Loading up regressors 
        P = cellstr(spm_select('FPList', directory,'^art.*outliers_and_movement.*\.mat$'));

        matlabbatch{1}.spm.stats.fmri_spec.sess(ss).multi_reg = P;

        clear P;       

    end
    
    save(fullfile(stats_directory, 'design.mat'), 'matlabbatch');

    spm_jobman('run', matlabbatch);
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode,'e')
    
    stats_directory = fullfile(D, 'stats');
    statsfile = fullfile(stats_directory, 'SPM.mat');
    
    load(fullfile(batch_files, 'estimate.mat'), 'matlabbatch');
    
    % Set the right SPM.mat file
    matlabbatch{1}.spm.stats.fmri_est.spmmat={statsfile};
    save(fullfile(stats_directory, 'estimate.mat'), 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contrast Manager
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if contains(mode,'c')
    
    stats_directory=fullfile(D,'stats');
    statsfile=fullfile(stats_directory,'SPM.mat');    
    
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
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.conjoint = 1;
    
    % Mirroring LRUD > baseline
    w_flip_LRUD = contains(SPM.xX.name, 'flip_LRUD*')/sum(contains(SPM.xX.name, 'flip_LRUD*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'Mirroring LRUD > baseline';         % t Contrast (f contrast = fcon)
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = w_flip_LRUD;
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';  
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.conjoint = 1;
    
    % Mirroring LR > baseline
    w_flip_LR = contains(SPM.xX.name, 'flip_LR*')/sum(contains(SPM.xX.name, 'flip_LR*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));    
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'Mirroring LR > baseline';              
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = w_flip_LR;    
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';  
    matlabbatch{1}.spm.stats.con.consess{3}.tcon.conjoint = 1;
    
    % Mirroring UD > baseline
    w_flip_UD = contains(SPM.xX.name, 'flip_UD*')/sum(contains(SPM.xX.name, 'flip_UD*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));   
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'Mirroring UD > baseline';              
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = w_flip_UD;    
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';  
    matlabbatch{1}.spm.stats.con.consess{4}.tcon.conjoint = 1;
    
    % Spacing 2 > baseline
    w_space_2 = contains(SPM.xX.name, 'spacing_2*')/sum(contains(SPM.xX.name, 'spacing_2*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'Spacing 2 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = w_space_2;
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.consess{5}.tcon.conjoint = 1;

    
    % Spacing 4 > baseline
    w_space_4 = contains(SPM.xX.name, 'spacing_4*')/sum(contains(SPM.xX.name, 'spacing_4*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'Spacing 4 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = w_space_4;
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.consess{6}.tcon.conjoint = 1;

    % Spacing 6 > baseline
    w_space_6 = contains(SPM.xX.name, 'spacing_6*')/sum(contains(SPM.xX.name, 'spacing_6*')) - ...
        contains(SPM.xX.name, 'baseline')/sum(contains(SPM.xX.name, 'baseline'));
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'Spacing 6 > baseline';              
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = zeros(1, Nregr);
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = w_space_6;   
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';  
    matlabbatch{1}.spm.stats.con.consess{7}.tcon.conjoint = 1;



    spm_jobman('run',matlabbatch);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Results Report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if contains(mode, 'r')
    
    stats_directory=fullfile(D,'stats');
    statsfile=fullfile(stats_directory,'SPM.mat');

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

    save(fullfile(stats_directory, 'results_report.mat'), 'matlabbatch');
    spm_jobman('run',matlabbatch(1));

    %% --- 2. FWE CORRETTO: p < 0.05 cluster-level ---
    matlabbatch{2}.spm.stats.results.spmmat={statsfile};

    csvFile = dir(fullfile(stats_directory, "*.csv"));
    

    for i=1:length(SPM.xCon)
        T = readtable(fullfile(stats_directory, csvFile(i).name), 'NumHeaderLines',1);
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
    matlabbatch{2}.spm.stats.results.export{4}.tspm.basename = 'FWE_cluster';

    save(fullfile(stats_directory, 'results_report.mat'), 'matlabbatch');
    spm_jobman('run',matlabbatch(2));

end 

return

    


