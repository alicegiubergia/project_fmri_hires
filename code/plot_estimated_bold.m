clear; clc;

%% === Setup ===
condition_name = 'baseline';  % ← Cambia qui il nome della condizione
xyz_mm = [-30 -86 2]';      % ← Coordinate MNI
spm_folder = '/mnt/store2/projects/progetto_ME-fMRI/progetto_reading/derivatives/mypipeline/sub-00/stats/';
output_dir = fullfile(spm_folder, 'figures'); % Salva nella stessa cartella SPM
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

filename = fullfile(output_dir, ['voxel_prediction_' strrep(condition_name, ' ', '_') '.png']);

%% === Funzione per ottenere segnale predetto da SPM ===
get_model_prediction = @(folder, xyz_mm, condition_name) ...
    deal_func(folder, xyz_mm, condition_name);

%% === HRF ===
[y, model_hrf] = get_model_prediction(spm_folder, xyz_mm, condition_name);
bold_z = y;
model_hrf_z = model_hrf;

%% === Plot ===
figure;
hold on;
plot(bold_z, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', .6);
plot(model_hrf_z, 'b', 'LineWidth', .8);
legend({['Raw data: ' condition_name], ['Predicted HRF']}, ...
       'FontSize', 8, 'Location', 'northeast');
title('GLM Prediction at voxel for Mirroring UD', 'FontSize',10);
xlabel('Scan', 'FontSize', 9);
ylabel('BOLD', 'FontSize', 9);

set(gca, 'FontSize', 8); % assi

%% === Funzione interna ===
function [y, yhat] = deal_func(folder, xyz_mm, condition_name)
    cd(folder);
    load('SPM.mat');

    % Coordinate voxel
    xyz_vox = round(SPM.xVol.M \ [xyz_mm; 1]);
    xyz_vox = xyz_vox(1:3);

    % Estrai betas
    n_betas = length(SPM.Vbeta);
    betas = zeros(n_betas, 1);
    for i = 1:n_betas
        betas(i) = spm_sample_vol(SPM.Vbeta(i), xyz_vox(1), xyz_vox(2), xyz_vox(3), 0);
    end

    % Design matrix e colonne condizione
    X = SPM.xX.X;
    cond_cols = find(contains(SPM.xX.name, condition_name) | contains(SPM.xX.name, 'constant'));
    beta_cond = betas(cond_cols);
    X_cond = X(:, cond_cols);

    % Segnale stimato
    yhat = X_cond * beta_cond;

    % Segnale osservato (solo la prima volta)
    if nargout > 1
        y = spm_get_data(SPM.xY.P, xyz_vox);
    else
        y = [];
    end
end

% === Salva figura ===
set(gcf, 'Units', 'centimeters');
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 16 8]); % stampa

set(gca, 'FontSize', 8, 'TickDir', 'out', 'Box', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [5 5 16 8]);  % più larga per farci stare la legenda
exportgraphics(gcf, filename, 'Resolution', 300);  
