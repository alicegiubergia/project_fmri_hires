function layer_profiles_roiwise()
% LAMINAR PROFILES from ROI-wise deveined BOLD.
%
% For each subject, ROI set (baseline / mirroring / spacing) and region
% (Dors / Ven / Vis):
%   1. average the deveined time series (already in PSC) over the GM voxels
%      of each layer (1..N_LAYERS), run by run
%   2. GLM on the N_LAYERS time series using the design matrix, high-pass
%      filter and whitening of the NON-deveined whole-brain GLM
%      (same onsets, same ART + physio confounds)
%   3. per-layer contrasts -> long-format table + profile figure
%
% Outputs: separate BIDS-derivatives pipeline folder (OUT_ROOT)
%   dataset_description.json
%   sub-XXX/func/sub-XXX_task-<TASK>_desc-<DESC>_layerprofiles.tsv (+ .json)
%   group/task-<TASK>_desc-<DESC>_layerprofiles.tsv (+ .json)   all subjects
%   group/task-<TASK>_desc-<DESC>_layerprofiles.png             mean +- SEM
% TSV columns: subject, roi_set, region, layer, n_vox, std_psc, contrast,
%              spm_con, primary, con, t
% Contrasts: the same 9 of the whole-brain GLM (see make_contrasts);
% spm_con = their index (con_000X / spmT_000X).
% primary = 1 for the contrast that defined the ROI set (amplitude inflated
% by circular selection: interpret its profile shape, not its size).

%% ---------------- CONFIG ----------------
BASE        = '/mnt/store1/projects/progetto_layer';
DEVEIN_ROOT = fullfile(BASE, 'data', 'derivatives', 'roi-deveined');
SPM_ROOT    = fullfile(BASE, 'data', 'derivatives', 'hires_pipeline');
SPM_SUBDIR  = 'stats-masked';

ROI_SETS = {'baseline', 'mirroring', 'spacing'};
REGIONS  = {'Dors', 'Ven', 'Vis'};
N_LAYERS = 10;
N_RUNS   = 3;
GM_VALUE = 3;              % GM value in the LayNii rim file
% contrast that defined each ROI set (flagged as primary in the TSV)
PRIMARY  = struct('baseline', 'baseline', ...
                  'mirroring', 'mirroring>baseline', ...
                  'spacing', 'spacing>baseline');
% output pipeline (BIDS derivatives)
OUT_ROOT = fullfile(BASE, 'data', 'derivatives', 'layer-profiles');
TASK     = 'reading';
DESC     = 'deveinroiwise';    % BIDS labels: alphanumeric only

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
    spm_file = fullfile(SPM_ROOT, sub, SPM_SUBDIR, 'SPM.mat');
    if ~exist(spm_file, 'file')
        warning('%s: SPM.mat not found (%s), skipping', sub, spm_file);
        continue
    end
    S = load(spm_file, 'SPM');  SPM = S.SPM;
    assert(numel(SPM.nscan) == N_RUNS, '%s: %d sessions in SPM.mat, expected %d', ...
        sub, numel(SPM.nscan), N_RUNS);
    [C, cnames] = make_contrasts(SPM.xX.name);
    fprintf('\n=== %s (%s) ===\n', sub, strjoin(cnames, ', '));

    for rs = 1:numel(ROI_SETS)
        for rg = 1:numel(REGIONS)
            d   = fullfile(DEVEIN_ROOT, sub, ROI_SETS{rs});
            prefix = fullfile(d, REGIONS{rg});
            if ~exist([prefix '_layers.nii.gz'], 'file')
                warning('%s %s %s: layer file missing, skipping', sub, ROI_SETS{rs}, REGIONS{rg});
                continue
            end

            % layer masks (rim GM only)
            L   = read_vol([prefix '_layers.nii.gz']);
            R   = read_vol([prefix '_rim.nii.gz']);
            dim = size(L);
            masks = cell(1, N_LAYERS);
            for l = 1:N_LAYERS
                masks{l} = (round(L) == l) & (round(R) == GM_VALUE);
            end

            % mean time series per layer, runs concatenated
            Y = zeros(sum(SPM.nscan), N_LAYERS);
            Ypsc_std = zeros(N_RUNS, N_LAYERS);
            row0 = 0;
            for r = 1:N_RUNS
                f_dv  = sprintf('%s_run-%02d_devein.nii', prefix, r);
                f_psc = sprintf('%s_run-%02d_psc.nii', prefix, r);
                [y, nvox] = layer_means(f_dv, masks, dim);
                assert(size(y, 1) == SPM.nscan(r), ...
                    '%s run %d: %d deveined volumes vs %d in SPM.mat', ...
                    f_dv, r, size(y, 1), SPM.nscan(r));
                Y(row0 + (1:size(y, 1)), :) = y;
                row0 = row0 + size(y, 1);
                % QC: std of the layer-mean time series before deveining
                if exist(f_psc, 'file')
                    Ypsc_std(r, :) = std(layer_means(f_psc, masks, dim), 0, 1);
                end
            end

            % GLM with the whole-brain model's design, filter and whitening
            [con, t] = fit_layers(SPM.xX, Y, C);

            for l = 1:N_LAYERS
                for c = 1:numel(cnames)
                    is_primary = strcmp(cnames{c}, PRIMARY.(ROI_SETS{rs}));
                    rows(end + 1, :) = {sub, ROI_SETS{rs}, REGIONS{rg}, l, ...
                        nvox(l), mean(Ypsc_std(:, l)), c, cnames{c}, is_primary, ...
                        con(c, l), t(c, l)}; %#ok<AGROW>
                end
            end
            fprintf('  %-9s %-4s ok (voxels per layer: %s)\n', ROI_SETS{rs}, ...
                REGIONS{rg}, num2str(nvox));
        end
    end
end

T = cell2table(rows, 'VariableNames', {'subject', 'roi_set', 'region', ...
    'layer', 'n_vox', 'std_psc', 'spm_con', 'contrast', 'primary', 'con', 't'});
stem = sprintf('task-%s_desc-%s_layerprofiles', TASK, DESC);

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

plot_profiles(T, ROI_SETS, REGIONS, N_LAYERS, fullfile(gdir, [stem '.png']));
end


function init_derivatives(out_root, source_dir)
% Create the pipeline folder and its dataset_description.json
if ~exist(out_root, 'dir'); mkdir(out_root); end
f = fullfile(out_root, 'dataset_description.json');
if exist(f, 'file'); return; end
dd = struct( ...
    'Name', 'Laminar profiles from ROI-wise deveined BOLD', ...
    'BIDSVersion', '1.9.0', ...
    'DatasetType', 'derivative', ...
    'GeneratedBy', {{struct('Name', 'layer_profiles_devein_roiwise', ...
        'Description', ['Layer-averaged deveined time series fitted with the ' ...
        'whole-brain first-level design (SPM12)'])}}, ...
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
    'n_vox',    struct('Description', 'Number of GM voxels in the layer'), ...
    'std_psc',  struct('Description', 'Temporal std of the layer-mean time series before deveining, averaged over runs', 'Units', '%'), ...
    'spm_con',  struct('Description', ['Contrast index in the whole-brain ' ...
        'first-level GLM (con_000X / spmT_000X)']), ...
    'contrast', struct('Description', ['Contrast name: baseline = baseline > 0; ' ...
        '<level>>baseline; mirroring/spacing = all levels pooled']), ...
    'primary',  struct('Description', ['1 if this is the contrast that defined ' ...
        'the ROI set (selected on the same runs: amplitude is inflated)']), ...
    'con',      struct('Description', 'Contrast estimate on the layer-mean deveined time series', 'Units', '% signal change'), ...
    't',        struct('Description', 't statistic of the contrast'));
fid = fopen([stem_path '.json'], 'w');
fprintf(fid, '%s', jsonencode(cols, 'PrettyPrint', true));
fclose(fid);
end


%% ================= FUNCTIONS =================

function [C, names] = make_contrasts(xname)
% Same 9 contrasts, same order, as the whole-brain first level
% (row k = con_000k / spmT_000k):
%   1 baseline > 0            ('Normally displayed words')
%   2 flip_LRUD > baseline    3 flip_LR > baseline    4 flip_UD > baseline
%   5 spacing_2 > baseline    6 spacing_4 > baseline  7 spacing_6 > baseline
%   8 mirroring (all) > baseline                      9 spacing (all) > baseline
% Only bf(1) regressors. Matching '<level>*bf(1)' keeps flip_LR distinct
% from flip_LRUD.
p   = numel(xname);
avg = @(i) full(sparse(1, i, 1 / numel(i), 1, p));
bf1 = contains(xname, '*bf(1)');
idx = @(lvl) find(contains(xname, [lvl '*bf(1)']));

ib = idx('baseline');
assert(~isempty(ib), 'baseline regressors not found in SPM.xX.name');
cb = avg(ib);
names = {'baseline'};  C = cb;

for lvl = {'flip_LRUD', 'flip_LR', 'flip_UD', 'spacing_2', 'spacing_4', 'spacing_6'}
    i = idx(lvl{1});
    assert(~isempty(i), '%s regressors not found in SPM.xX.name', lvl{1});
    names{end + 1} = [lvl{1} '>baseline']; %#ok<AGROW>
    C = [C; avg(i) - cb]; %#ok<AGROW>
end

im = bf1 & contains(xname, 'flip');
is = bf1 & contains(xname, 'spacing');
names(end + (1:2)) = {'mirroring>baseline', 'spacing>baseline'};
C = [C; avg(im) - cb; avg(is) - cb];
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
% Per-layer mean, volume by volume (does not load the whole 4D in memory);
% excludes any NaN/Inf
V = spm_vol(f);
assert(isequal(V(1).dim, dim), '%s: grid differs from the layer file', f);
nl = numel(masks);
y  = zeros(numel(V), nl);
for k = 1:numel(V)
    v = spm_read_vols(V(k));
    for l = 1:nl
        x = v(masks{l});
        y(k, l) = mean(x(isfinite(x)));
    end
end
nvox = cellfun(@nnz, masks);
end


function v = read_vol(f)
% Read a 3D NIfTI, including .nii.gz (decompressed to a temporary folder)
if endsWith(f, '.gz')
    tmp = tempname; mkdir(tmp);
    g = gunzip(f, tmp); f = g{1};
    v = spm_read_vols(spm_vol(f));
    rmdir(tmp, 's');
else
    v = spm_read_vols(spm_vol(f));
end
end


function plot_profiles(T, roi_sets, regions, nl, out_png)
% Mean +- SEM profiles across subjects: rows = ROI set, columns = region.
% Each row shows the contrasts of its own ROI set: the primary (defining)
% contrast in black, its levels in colour.
SHOW.baseline  = {'baseline'};
SHOW.mirroring = {'mirroring>baseline', 'flip_LR>baseline', 'flip_UD>baseline', 'flip_LRUD>baseline'};
SHOW.spacing   = {'spacing>baseline', 'spacing_2>baseline', 'spacing_4>baseline', 'spacing_6>baseline'};
COLS.baseline  = [0 0 0];
COLS.mirroring = [0 0 0; 0.99 0.68 0.38; 0.90 0.33 0.05; 0.60 0.15 0.00];
COLS.spacing   = [0 0 0; 0.55 0.75 0.95; 0.13 0.47 0.80; 0.03 0.25 0.55];
fig  = figure('Color', 'w', 'Position', [100 100 1150 950]);
for rs = 1:numel(roi_sets)
    show = SHOW.(roi_sets{rs});
    cols = COLS.(roi_sets{rs});
    for rg = 1:numel(regions)
        subplot(numel(roi_sets), numel(regions), (rs - 1) * numel(regions) + rg); hold on
        for c = 1:numel(show)
            sel = strcmp(T.roi_set, roi_sets{rs}) & strcmp(T.region, regions{rg}) & ...
                  strcmp(T.contrast, show{c});
            if ~any(sel); continue; end
            Ts = T(sel, :);
            subj = unique(Ts.subject);
            M = nan(numel(subj), nl);
            for i = 1:numel(subj)
                ti = Ts(strcmp(Ts.subject, subj{i}), :);
                M(i, ti.layer) = ti.con;
            end
            m  = mean(M, 1, 'omitnan');
            se = std(M, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(M), 1));
            fill([1:nl, nl:-1:1], [m + se, fliplr(m - se)], cols(c, :), ...
                 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            lw = 1.2 + 1.0 * (c == 1);                % primary drawn thicker
            plot(1:nl, m, '-o', 'Color', cols(c, :), 'LineWidth', lw, ...
                 'MarkerFaceColor', cols(c, :), 'MarkerSize', 4, ...
                 'DisplayName', strrep(show{c}, '>baseline', ''));
        end
        yline(0, ':', 'HandleVisibility', 'off');
        xlim([0.5 nl + 0.5]); xticks(1:nl); box off
        title(sprintf('%s ROI (%s), deveined', regions{rg}, roi_sets{rs}), 'Interpreter', 'none');
        if rs == numel(roi_sets); xlabel('layer (1 = WM \rightarrow CSF)'); end
        if rg == 1
            if strcmp(roi_sets{rs}, 'baseline'); ylabel({'baseline > 0', 'beta (% signal change)'});
            else; ylabel({'vs baseline', '\Delta beta (% signal change)'}); end
        end
        if rg == numel(regions)
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'Box', 'off');
        end
    end
end
sgtitle(sprintf(['Laminar profiles - deveined BOLD, ROI-wise GLM ' ...
    '(mean \\pm SEM, N = %d)'], numel(unique(T.subject))), 'FontWeight', 'bold');
exportgraphics(fig, out_png, 'Resolution', 200);
fprintf('Figure: %s\n', out_png);
end