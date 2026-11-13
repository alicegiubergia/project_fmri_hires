function layer_profiles_baseline_devein_vs_raw()
% LAMINAR PROFILES of baseline > 0: deveined vs non-deveined BOLD.
%
% Same pipeline as layer_profiles_roiwise, restricted to one contrast.
% For each subject, ROI set and region:
%   1. average over the GM voxels of each layer (1..N_LAYERS), run by run,
%      both the deveined time series (_devein) and the non-deveined one
%      (_psc), on the same layer masks
%   2. fit both with the design matrix, high-pass filter and whitening of
%      the NON-deveined whole-brain GLM (same onsets, same confounds):
%      any difference between the two profiles comes from the deveining only
%   3. baseline > 0 per layer -> long-format table + comparison figure
%
% Voxel selection matches the PSC maps of group_layer_plotting_psc.py:
%   - inside the SPM analysis mask (mask.nii of the first level)
%   - mean of the constant betas >= MIN_SESSION_MEAN
%   - native mask resampled trilinearly on the crop grid and kept only
%     where ~1, i.e. all native neighbours valid (mimics the NaN spread of
%     the linear upsampling of the PSC maps)
%   - voxels left at 0 for the whole run by psc_alf.py are excluded
% Note: for the deveined data this restricts the averaging only; LN2_DEVEIN
% has already used all voxels of each column.
%
% Outputs: same BIDS-derivatives folder as layer_profiles_roiwise, own desc
%   sub-XXX/func/sub-XXX_task-<TASK>_desc-<DESC>_layerprofiles.tsv (+ .json)
%   group/task-<TASK>_desc-<DESC>_layerprofiles.tsv (+ .json)   all subjects
%   group/task-<TASK>_desc-<DESC>_layerprofiles.png             mean +- SEM
% TSV columns: subject, roi_set, region, layer, n_vox, data, primary, con, t
%   data = 'deveined' | 'nondeveined'

%% ---------------- CONFIG ----------------
BASE        = '/mnt/store1/projects/progetto_layer';
DEVEIN_ROOT = fullfile(BASE, 'data', 'derivatives', 'roi-deveined');
SPM_ROOT    = fullfile(BASE, 'data', 'derivatives', 'hires_pipeline');
SPM_SUBDIR  = 'stats-masked';

ROI_SETS = {'mirroring', 'spacing', 'baseline'};   % add 'baseline' if that set exists
REGIONS  = {'Dors', 'Ven', 'Vis'};
N_LAYERS = 10;
N_RUNS   = 3;
GM_VALUE = 3;              % GM value in the LayNii rim file
DATA     = {'deveined', 'nondeveined'};
MIN_SESSION_MEAN = 1.0;    % same threshold as group_layer_plotting_psc

% output pipeline (BIDS derivatives)
OUT_ROOT      = fullfile(BASE, 'data', 'derivatives', 'layer-profiles');
TASK          = 'reading';
IN_FILE_DESC  = 'd2mm';
OUT_FILE_DESC = [IN_FILE_DESC 'DeveinVsRaw'];   % BIDS labels: alphanumeric only

addpath('/home/matuser/matlab_toolbox/spm12');
spm('defaults', 'fmri');
init_derivatives(OUT_ROOT, DEVEIN_ROOT);

%% ---------------- LOOP ----------------
subs = dir(fullfile(DEVEIN_ROOT, 'sub-*'));
subs = {subs([subs.isdir]).name};
fprintf('Subjects found: %d\n', numel(subs));

rows = {};
for s = 1:numel(subs)
    sub = subs{s};
    spm_dir  = fullfile(SPM_ROOT, sub, SPM_SUBDIR);
    spm_file = fullfile(spm_dir, 'SPM.mat');
    if ~exist(spm_file, 'file')
        warning('%s: SPM.mat not found (%s), skipping', sub, spm_file);
        continue
    end
    S = load(spm_file, 'SPM');  SPM = S.SPM;
    assert(numel(SPM.nscan) == N_RUNS, '%s: %d sessions in SPM.mat, expected %d', ...
        sub, numel(SPM.nscan), N_RUNS);
    cb = baseline_contrast(SPM.xX.name);
    [mnat, Mnat] = native_valid(SPM, spm_dir, MIN_SESSION_MEAN);
    fprintf('\n=== %s ===\n', sub);

    for rs = 1:numel(ROI_SETS)
        for rg = 1:numel(REGIONS)
            prefix = fullfile(DEVEIN_ROOT, sub, ROI_SETS{rs}, REGIONS{rg});
            if ~exist([prefix '_layers.nii.gz'], 'file')
                warning('%s %s %s: layer file missing, skipping', sub, ROI_SETS{rs}, REGIONS{rg});
                continue
            end

            % layer masks (rim GM only, SPM-valid voxels), shared by both data types
            [L, VL] = read_vol([prefix '_layers.nii.gz']);
            R   = read_vol([prefix '_rim.nii.gz']);
            dim = size(L);
            ok_spm = to_crop(mnat, Mnat, VL.mat, dim);
            masks = cell(1, N_LAYERS);
            for l = 1:N_LAYERS
                masks{l} = (round(L) == l) & (round(R) == GM_VALUE) & ok_spm;
            end

            % mean time series per layer, runs concatenated
            Ydv  = zeros(sum(SPM.nscan), N_LAYERS);
            Yraw = zeros(sum(SPM.nscan), N_LAYERS);
            nvox_run = zeros(N_RUNS, N_LAYERS);
            row0 = 0;  ok = true;
            for r = 1:N_RUNS
                f_dv  = sprintf('%s_run-%02d_desc-%s_devein.nii', prefix, r, IN_FILE_DESC);
                f_raw = sprintf('%s_run-%02d_psc.nii', prefix, r);
                if ~exist(f_raw, 'file')
                    warning('%s: non-deveined file missing, skipping %s %s', ...
                        f_raw, ROI_SETS{rs}, REGIONS{rg});
                    ok = false;  break
                end
                ydv               = layer_means(f_dv, masks, dim);
                [yraw, nvox_run(r, :)] = layer_means(f_raw, masks, dim);
                assert(size(ydv, 1) == SPM.nscan(r) && size(yraw, 1) == SPM.nscan(r), ...
                    '%s run %d: %d deveined / %d non-deveined volumes vs %d in SPM.mat', ...
                    prefix, r, size(ydv, 1), size(yraw, 1), SPM.nscan(r));
                rr = row0 + (1:SPM.nscan(r));
                Ydv(rr, :)  = ydv;
                Yraw(rr, :) = yraw;
                row0 = row0 + SPM.nscan(r);
            end
            if ~ok; continue; end
            nvox = min(nvox_run, [], 1);   % voxels valid in every run

            % GLM with the whole-brain model's design, filter and whitening
            [con_dv,  t_dv]  = fit_layers(SPM.xX, Ydv,  cb);
            [con_raw, t_raw] = fit_layers(SPM.xX, Yraw, cb);
            con = [con_dv', con_raw'];     % columns follow DATA
            t   = [t_dv',   t_raw'];

            is_primary = strcmp(ROI_SETS{rs}, 'baseline');
            for l = 1:N_LAYERS
                for d = 1:numel(DATA)
                    rows(end + 1, :) = {sub, ROI_SETS{rs}, REGIONS{rg}, l, ...
                        nvox(l), DATA{d}, is_primary, con(l, d), t(l, d)}; %#ok<AGROW>
                end
            end
            fprintf('  %-9s %-4s ok (voxels per layer: %s)\n', ROI_SETS{rs}, ...
                REGIONS{rg}, num2str(nvox));
        end
    end
end

T = cell2table(rows, 'VariableNames', {'subject', 'roi_set', 'region', ...
    'layer', 'n_vox', 'data', 'primary', 'con', 't'});
stem = sprintf('task-%s_desc-%s_%s_layerprofiles', TASK, OUT_FILE_DESC, IN_FILE_DESC);

% per-subject tables
for sub = unique(T.subject)'
    d = fullfile(OUT_ROOT, sub{1}, 'func');
    if ~exist(d, 'dir'); mkdir(d); end
    write_tsv(T(strcmp(T.subject, sub{1}), :), fullfile(d, [sub{1} '_' stem]));
end

% group table + figure
gdir = fullfile(OUT_ROOT, 'group');
if ~exist(gdir, 'dir'); mkdir(gdir); end
write_tsv(T, fullfile(gdir, stem));
fprintf('\nOutputs in: %s\n', OUT_ROOT);

plot_compare(T, ROI_SETS, REGIONS, N_LAYERS, DATA, fullfile(gdir, [stem '.png']));
end


%% ================= FUNCTIONS =================

function cb = baseline_contrast(xname)
% baseline > 0: mean of the baseline*bf(1) regressors over runs
% (same row as con_0001 of the whole-brain first level)
p  = numel(xname);
ib = find(contains(xname, 'baseline*bf(1)'));
assert(~isempty(ib), 'baseline regressors not found in SPM.xX.name');
cb = full(sparse(1, ib, 1 / numel(ib), 1, p));
end


function [m, M] = native_valid(SPM, spm_dir, min_mean)
% Native-grid voxels kept by the PSC maps of group_layer_plotting_psc:
% inside the SPM analysis mask and mean constant beta >= min_mean
VM = spm_vol(in_dir(spm_dir, SPM.VM.fname));
m  = spm_read_vols(VM) > 0;
iB = SPM.xX.iB;
B  = zeros(VM.dim);
for i = iB
    B = B + spm_read_vols(spm_vol(in_dir(spm_dir, SPM.Vbeta(i).fname)));
end
m = m & (B / numel(iB) >= min_mean);      % NaN >= x is false
M = VM.mat;
end


function ok = to_crop(m, Mnat, Mcrop, dim)
% Native mask resampled on the crop grid (trilinear): keep only voxels
% whose native neighbours are all valid, mimicking the NaN spread of the
% linear upsampling of the PSC maps. Outside the native FOV -> excluded.
[i, j, k] = ndgrid(1:dim(1), 1:dim(2), 1:dim(3));
xyz = Mnat \ Mcrop * [i(:)'; j(:)'; k(:)'; ones(1, numel(i))];
s   = spm_sample_vol(double(m), xyz(1, :)', xyz(2, :)', xyz(3, :)', 1);
ok  = reshape(s > 0.999, dim);
end


function f = in_dir(d, f0)
% file name stored in SPM.mat, relocated to the stats folder
[~, n, e] = fileparts(f0);
f = fullfile(d, [n e]);
end


function [con, t] = fit_layers(xX, Y, C)
% Same estimation as spm_spm: filter K, whitening W, pinv of filtered design
if isfield(xX, 'W') && ~isempty(xX.W); W = xX.W; else; W = speye(size(Y, 1)); end
KWY   = spm_filter(xX.K, W * Y);
beta  = xX.pKX * KWY;
res   = spm_sp('r', xX.xKXs, KWY);
ResMS = sum(res .^ 2, 1) / xX.trRV;
con   = C * beta;
se    = sqrt(diag(C * xX.Bcov * C') * ResMS);
t     = con ./ se;
end


function [y, nvox] = layer_means(f, masks, dim)
% Per-layer mean of the time series.
% Excludes voxels that psc_alf.py left at 0 for the whole run
% (std = 0, e.g. outside the native FOV after resampling) and any NaN/Inf.
V = spm_vol(f);
assert(isequal(V(1).dim, dim), '%s: grid differs from the layer file', f);
anym = false(dim);
for l = 1:numel(masks); anym = anym | masks{l}; end
D = zeros(numel(V), nnz(anym));               % volumes x voxels of the ROI
for k = 1:numel(V)
    v = spm_read_vols(V(k));
    D(k, :) = v(anym);
end
valid = all(isfinite(D), 1) & any(D ~= 0, 1);
nl   = numel(masks);
y    = zeros(numel(V), nl);
nvox = zeros(1, nl);
for l = 1:nl
    sel = masks{l}(anym)' & valid;
    y(:, l) = mean(D(:, sel), 2);
    nvox(l) = nnz(sel);
end
end


function [v, V] = read_vol(f)
% Read a 3D NIfTI, including .nii.gz (decompressed to a temporary folder)
if endsWith(f, '.gz')
    tmp = tempname; mkdir(tmp);
    g = gunzip(f, tmp);
    V = spm_vol(g{1}); v = spm_read_vols(V);
    rmdir(tmp, 's');
else
    V = spm_vol(f); v = spm_read_vols(V);
end
end


function init_derivatives(out_root, source_dir)
% Create the pipeline folder and its dataset_description.json
% (shared with layer_profiles_roiwise: left untouched if it already exists)
if ~exist(out_root, 'dir'); mkdir(out_root); end
f = fullfile(out_root, 'dataset_description.json');
if exist(f, 'file'); return; end
dd = struct( ...
    'Name', 'Laminar profiles from ROI-wise deveined BOLD', ...
    'BIDSVersion', '1.9.0', ...
    'DatasetType', 'derivative', ...
    'GeneratedBy', {{struct('Name', 'layer_profiles_devein_roiwise', ...
        'Description', ['Layer-averaged deveined and non-deveined time series ' ...
        'fitted with the whole-brain first-level design (SPM12)'])}}, ...
    'SourceDatasets', {{struct('URL', ['file://' source_dir])}});
fid = fopen(f, 'w');
fprintf(fid, '%s', jsonencode(dd, 'PrettyPrint', true));
fclose(fid);
end


function write_tsv(T, stem_path)
% Write <stem>.tsv and a <stem>.json sidecar describing the columns
writetable(T, [stem_path '.tsv'], 'FileType', 'text', 'Delimiter', '\t');
cols = struct( ...
    'subject',  struct('Description', 'Subject label'), ...
    'roi_set',  struct('Description', 'Contrast used to define the ROI (baseline, mirroring, spacing)'), ...
    'region',   struct('Description', 'Region (Dors, Ven, Vis)'), ...
    'layer',    struct('Description', 'Cortical depth bin from the LayNii layer file (1 = WM side, 10 = CSF side)'), ...
    'n_vox',    struct('Description', ['Number of GM voxels in the layer used for the mean: ' ...
        'inside the SPM mask, mean constant beta >= threshold, nonzero in every run']), ...
    'data',     struct('Description', ['Time series fitted: deveined (_devein) or ' ...
        'non-deveined (_psc), same layer masks and same GLM']), ...
    'primary',  struct('Description', ['1 if baseline > 0 is the contrast that defined ' ...
        'the ROI set (selected on the same runs: amplitude is inflated)']), ...
    'con',      struct('Description', ['baseline > 0 estimate on the layer-mean time series, ' ...
        'per unit regressor height (multiply by the regressor peak SF to get % signal change)']), ...
    't',        struct('Description', 't statistic of baseline > 0'));
fid = fopen([stem_path '.json'], 'w');
fprintf(fid, '%s', jsonencode(cols, 'PrettyPrint', true));
fclose(fid);
end


function plot_compare(T, roi_sets, regions, nl, data_lbl, out_png)
% Mean +- SEM baseline > 0 profiles across subjects:
% rows = ROI set, columns = region; deveined in black, non-deveined in grey
COLS = [0 0 0; 0.6 0.6 0.6];
nr  = numel(roi_sets);  ng = numel(regions);
fig = figure('Color', 'w', 'Position', [100 100 1150 300 * nr + 120]);
tlo = tiledlayout(fig, nr, ng, 'TileSpacing', 'compact', 'Padding', 'compact');
for rs = 1:nr
    for rg = 1:ng
        ax = nexttile(tlo); hold(ax, 'on')
        for d = 1:numel(data_lbl)
            sel = strcmp(T.roi_set, roi_sets{rs}) & strcmp(T.region, regions{rg}) & ...
                  strcmp(T.data, data_lbl{d});
            if ~any(sel); continue; end
            Ts   = T(sel, :);
            subj = unique(Ts.subject);
            M = nan(numel(subj), nl);
            for i = 1:numel(subj)
                ti = Ts(strcmp(Ts.subject, subj{i}), :);
                M(i, ti.layer) = ti.con;
            end
            m  = mean(M, 1, 'omitnan');
            se = std(M, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(M), 1));
            fill(ax, [1:nl, nl:-1:1], [m + se, fliplr(m - se)], COLS(d, :), ...
                 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(ax, 1:nl, m, '-o', 'Color', COLS(d, :), 'LineWidth', 1.8, ...
                 'MarkerFaceColor', COLS(d, :), 'MarkerSize', 4, ...
                 'DisplayName', data_lbl{d});
        end
        yline(ax, 0, ':', 'HandleVisibility', 'off');
        xlim(ax, [0.5 nl + 0.5]); xticks(ax, 1:nl); box(ax, 'off')
        title(ax, sprintf('%s ROI (%s)', regions{rg}, roi_sets{rs}), 'Interpreter', 'none');
        if rs == nr; xlabel(ax, 'layer (1 = WM \rightarrow CSF)'); end
        if rg == 1; ylabel(ax, {'baseline > 0', 'beta (per unit regressor)'}); end
    end
end
lgd = legend(ax, 'Orientation', 'horizontal', 'Interpreter', 'none', 'Box', 'off');
lgd.Layout.Tile = 'north';
title(tlo, sprintf(['Laminar profiles of baseline > 0 - deveined vs non-deveined ' ...
    '(mean \\pm SEM, N = %d)'], numel(unique(T.subject))), 'FontWeight', 'bold');
exportgraphics(fig, out_png, 'Resolution', 300);
fprintf('Figure: %s\n', out_png);
end