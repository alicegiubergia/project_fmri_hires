%% write_residuals.m — residui del GLM di primo livello, come 4D, per l'ALF
%
% Serve solo se in 01_devein_laynii.sh ALF_MODE="residuals".
% Usa lo SPM.mat già stimato: non rifà la stima. SPM ricostruisce i residui
% dai beta e dai volumi originali, quindi i file in SPM.xY.VY (i BOLD entrati
% nel GLM) devono essere ancora raggiungibili ai loro percorsi.
%
% Output: <SPM.swd>/<sub>_residuals_4d.nii (griglia nativa, tutti i run
% concatenati). I file Res_XXXX.nii temporanei vengono cancellati.
%

cfg.root      = '/mnt/store1/projects/progetto_layer/data/derivatives/hires_pipeline';                        
cfg.subjects  = {'sub-001','sub-002','sub-003','sub-004', ...
                 'sub-005','sub-006','sub-007','sub-008'};
cfg.spm_first = @(sub) fullfile(cfg.root, sub, 'stats-masked', 'SPM.mat'); 

spm('defaults','fmri');
for s = 1:numel(cfg.subjects)
    sub = cfg.subjects{s};
    fprintf('=== %s ===\n', sub);
    load(cfg.spm_first(sub), 'SPM');
    old = pwd; cd(SPM.swd);

    % Ic = NaN: rimuove tutti gli effetti del modello (task + nuisance + filtro)
    VRes = spm_write_residuals(SPM, NaN);

    out = fullfile(SPM.swd, sprintf('%s_residuals_4d.nii', sub));
    spm_file_merge(VRes, out, 0);                    % 4D, float
    for k = 1:numel(VRes), delete(VRes(k).fname); end
    fprintf('  %d volumi -> %s\n', numel(VRes), out);
    cd(old);
end
