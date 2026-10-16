function gppi_laminar(cfg)
% GPPI_LAMINAR  gPPI di primo livello per l'analisi laminare FF/FB fra ROI.
%               Versione multi-sessione (piu' run per soggetto).
%
% Costruisce, per ogni soggetto e ogni ROI seed, un GLM di primo livello che
% contiene i regressori originali del task piu':
%   - la serie temporale del seed (regressore fisiologico)
%   - le serie delle altre due ROI come fisiologici di non interesse
%   - i termini di interazione PPI
% tutto replicato sessione per sessione.
%
% Due modelli, selezionati da cfg.model:
%   1  ('family')  3 interazioni: seed x baseline, seed x mirroring, seed x spacing
%                  -> risponde a "quale ROI guida quale"
%   2  ('trend')   5 interazioni: le tre sopra + trend lineare entro ciascuna
%                  famiglia -> risponde a "il bilancio FF/FB cambia con la
%                  degradazione"
%
% Nessuno dei due richiede di rispecificare il primo livello con dei pmod:
% il trend si ottiene dando pesi [-1 0 +1] ai tre livelli dentro spm_peb_ppi,
% che e' gia' centrato sulla media per costruzione.
%
% MULTI-SESSIONE. VOI, deconvoluzione e termini PPI sono specifici di ogni
% run: spm_peb_ppi lavora su una sessione per volta, e i regressori risultanti
% hanno la lunghezza di quella sessione. I contrasti sommano le colonne
% omologhe delle sessioni dividendo per il loro numero, cioe' fanno la media
% fra run.
%
% Le mappe con_* che escono entrano nella pipeline laminare esistente
% (upsampling ANTs -> maschera ROI target -> campionamento LayNii sui 10 bin).
%
% Uso:
%   cfg = gppi_laminar_config();
%   gppi_laminar(cfg);

assert(exist('spm','file')==2, 'SPM non e'' nel path.');
spm('defaults','FMRI');
spm_jobman('initcfg');

if ischar(cfg.model)
    cfg.model = find(strcmpi(cfg.model, {'family','trend'}));
end
assert(ismember(cfg.model,[1 2]), 'cfg.model deve essere 1/''family'' o 2/''trend''.');

for s = 1:numel(cfg.subjects)
    subj = cfg.subjects{s};
    srcSPM = fullfile(cfg.firstLevelDir(subj), 'SPM.mat');
    if ~exist(srcSPM,'file')
        warning('%s: SPM.mat di primo livello non trovato, salto.', subj); continue
    end
    S = load(srcSPM); SPMsrc = S.SPM;
    nSess = numel(SPMsrc.Sess);
    fprintf('\n=== %s : %d sessioni ===\n', subj, nSess);

    % i dati devono essere raggiungibili: il SPM.mat contiene percorsi
    % assoluti, e se i dati sono stati spostati vanno riscritti
    check_spm_paths(SPMsrc, srcSPM);

    % indici delle condizioni, per sessione (verificati uno a uno)
    idx = find_conditions(SPMsrc, cfg);
    if s == 1, report_conditions(idx); end

    % contrasto F "effects of interest" per l'adjust delle VOI
    Feoi = ensure_eoi_contrast(srcSPM, SPMsrc);

    for f = 1:numel(cfg.roiFamilies)
        fam = cfg.roiFamilies{f};

        % =============================================================
        % VOI: una per ROI e per sessione
        % =============================================================
        roiImg = cell(1, numel(cfg.rois));
        for r = 1:numel(cfg.rois)
            roiImg{r} = resolve_roi(cfg, subj, cfg.rois{r}, fam, s==1);
        end
        check_roi_overlap(cfg, subj, fam, roiImg);

        voiFiles = cell(numel(cfg.rois), nSess);   % {roi, sessione}
        for r = 1:numel(cfg.rois)
            voiFiles(r,:) = extract_voi(srcSPM, Feoi, ...
                                sprintf('%s_%s', cfg.rois{r}, fam), ...
                                roiImg{r}, nSess);
        end

        % =============================================================
        % un modello per ogni seed
        % =============================================================
        for r = 1:numel(cfg.rois)
            seed   = cfg.rois{r};
            others = setdiff(1:numel(cfg.rois), r);

            outDir = cfg.outDir(subj, seed, fam, cfg.model);
            if exist(fullfile(outDir,'SPM.mat'),'file') && ~cfg.overwrite
                fprintf('%s | %s | seed %-8s : gia'' fatto, salto.\n', subj, fam, seed);
                continue
            end
            if ~exist(outDir,'dir'), mkdir(outDir); end

            % ---- regressori aggiuntivi, sessione per sessione -------
            regFiles = cell(1, nSess);
            for ss = 1:nSess
                [ppi, ppiNames, Yphys] = build_ppi_terms( ...
                        srcSPM, voiFiles{r,ss}, idx, cfg, seed, fam, ss);

                nuis = []; nuisNames = {};
                if cfg.controlOtherRois
                    for o = others(:)'
                        V = load(voiFiles{o,ss});
                        nuis(:,end+1)    = V.Y;                              %#ok<AGROW>
                        nuisNames{end+1} = sprintf('phys_%s', cfg.rois{o});  %#ok<AGROW>
                    end
                end

                % SPM legge dai multi_reg le variabili 'R' e 'names':
                % i nomi devono essere esattamente questi.
                R     = [Yphys, ppi, nuis];
                names = [{sprintf('phys_%s', seed)}, ppiNames, nuisNames];
                regFiles{ss} = fullfile(outDir, sprintf('ppi_regressors_run%d.mat', ss));
                save(regFiles{ss}, 'R', 'names');
            end

            specify_design(outDir, SPMsrc, cfg, subj, regFiles);
            report_design(outDir, cfg, subj, fam, seed);

            if cfg.reviewDesign
                fprintf(['\nSOLO REVISIONE: modello specificato ma NON stimato.\n' ...
                         'Controlla la matrice qui sopra e in spm_DesRep, poi\n' ...
                         'rilancia con cfg.reviewDesign = false.\n' ...
                         'Cartella: %s\n'], outDir);
                return
            end

            estimate_design(outDir);
            define_contrasts(outDir, cfg);

            fprintf('%s | %s | seed %-8s : %d termini PPI x %d sessioni.\n', ...
                    subj, fam, seed, numel(ppiNames), nSess);
        end
    end
end
end % gppi_laminar


% =========================================================================
% CONDIZIONI
% =========================================================================
function idx = find_conditions(SPM, cfg)
% Gli indici in SPM.Sess(s).U possono in linea di principio differire fra
% sessioni, quindi li cerco per nome in ognuna invece di assumerli uguali.
nSess = numel(SPM.Sess);
idx.baseline  = zeros(1, nSess);
idx.mirroring = zeros(numel(cfg.cond.mirroring), nSess);
idx.spacing   = zeros(numel(cfg.cond.spacing),   nSess);
idx.names     = cell(1, nSess);

for s = 1:nSess
    nm = condition_names(SPM.Sess(s).U);
    idx.names{s} = nm;
    get = @(pat) find(strcmpi(nm, pat), 1);

    j = get(cfg.cond.baseline);
    assert(~isempty(j), 'Sessione %d: condizione baseline "%s" non trovata.', ...
           s, cfg.cond.baseline);
    idx.baseline(s) = j;

    for k = 1:numel(cfg.cond.mirroring)
        j = get(cfg.cond.mirroring{k});
        assert(~isempty(j), 'Sessione %d: condizione "%s" non trovata.', ...
               s, cfg.cond.mirroring{k});
        idx.mirroring(k,s) = j;
    end
    for k = 1:numel(cfg.cond.spacing)
        j = get(cfg.cond.spacing{k});
        assert(~isempty(j), 'Sessione %d: condizione "%s" non trovata.', ...
               s, cfg.cond.spacing{k});
        idx.spacing(k,s) = j;
    end
end
end

function check_spm_paths(SPM, spmPath)
% Verifica che i volumi referenziati dal SPM.mat esistano ancora. Un SPM.mat
% contiene percorsi ASSOLUTI: se i dati sono stati spostati di disco, SPM
% fallisce all'estrazione delle VOI con un "Could not access file".
missing = {};
for s = 1:numel(SPM.Sess)
    rows = SPM.Sess(s).row;
    % basta il primo e l'ultimo volume di ogni sessione
    for r = rows([1 end])
        f = strtok(SPM.xY.P(r,:), ',');     % toglie il ",1" dell'indice volume
        f = strtrim(f);
        if exist(f,'file') ~= 2, missing{end+1} = f; end %#ok<AGROW>
    end
end

if ~isempty(missing)
    old = fileparts(fileparts(missing{1}));
    error(['I dati referenziati da %s non sono raggiungibili.\n' ...
           '  primo mancante: %s\n\n' ...
           'Il SPM.mat contiene percorsi assoluti. Riscrivili con:\n' ...
           '  copyfile(''%s'', ''%s.bak'');\n' ...
           '  spm_changepath(''%s'', ''<vecchia radice>'', ''<nuova radice>'');\n\n' ...
           '(radice attuale nei file: %s)'], ...
           spmPath, missing{1}, spmPath, spmPath, spmPath, old);
end

if exist(SPM.swd,'dir') ~= 7
    warning(['SPM.swd punta a una cartella inesistente (%s). ' ...
             'Le VOI potrebbero non trovare beta e ResMS: ' ...
             'passa anche quello con spm_changepath.'], SPM.swd);
end
end

function nm = condition_names(U)
% Nomi delle condizioni di una sessione, come cell array di stringhe.
% U e' un array di struct; U(i).name e' di norma una cella 1x1, ma qualche
% versione/pipeline ci mette direttamente un char: gestisco entrambi.
nm = cell(1, numel(U));
for i = 1:numel(U)
    n = U(i).name;
    if iscell(n), n = n{1}; end
    nm{i} = char(n);
end
end

function report_conditions(idx)
for s = 1:numel(idx.names)
    fprintf('Condizioni in SPM.Sess(%d).U:\n', s);
    for i = 1:numel(idx.names{s})
        tag = '';
        if i == idx.baseline(s),            tag = '  <- baseline'; end
        k = find(idx.mirroring(:,s)==i, 1);
        if ~isempty(k), tag = sprintf('  <- mirroring %d', k); end
        k = find(idx.spacing(:,s)==i, 1);
        if ~isempty(k), tag = sprintf('  <- spacing %d', k); end
        fprintf('  %2d  %-20s%s\n', i, idx.names{s}{i}, tag);
    end
end
fprintf('\n');
end


% =========================================================================
% VOI
% =========================================================================
function Feoi = ensure_eoi_contrast(spmPath, SPM)
% Contrasto F su TUTTI i regressori di condizione, di tutte le sessioni.
% Le colonne di condizione sono riconoscibili perche' SPM le nomina
% "Sn(s) nome*bf(1)"; i regressori extra non hanno "*bf(".
Feoi = [];
if isfield(SPM,'xCon') && ~isempty(SPM.xCon)
    for i = 1:numel(SPM.xCon)
        if SPM.xCon(i).STAT == 'F' && strcmpi(SPM.xCon(i).name, 'effects of interest')
            Feoi = i; return
        end
    end
end
isCond = contains(SPM.xX.name, '*bf(');
assert(any(isCond), 'Nessuna colonna di condizione riconosciuta in SPM.xX.name.');
c = zeros(nnz(isCond), size(SPM.xX.X,2));
c(:, isCond) = eye(nnz(isCond));

clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spmPath};
matlabbatch{1}.spm.stats.con.consess{1}.fcon.name    = 'effects of interest';
matlabbatch{1}.spm.stats.con.consess{1}.fcon.weights = c;
matlabbatch{1}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 0;
spm_jobman('run', matlabbatch);

S = load(spmPath);
Feoi = numel(S.SPM.xCon);
end

function img = resolve_roi(cfg, subj, roi, fam, verbose)
% Espande il pattern della maschera e, se serve, la scompatta: SPM non legge
% i .nii.gz. La copia scompattata viene riusata nelle chiamate successive.
spmt = cfg.roiSpmT.(fam);
pat  = cfg.roiPattern(subj, roi, spmt);

d = dir(pat);
d = d(~[d.isdir]);
assert(~isempty(d), ['Nessuna maschera per %s / %s (%s).\n  pattern: %s\n' ...
                     '  Controlla cfg.roiPattern e cfg.roiSpmT.'], ...
                     roi, fam, spmt, pat);
assert(numel(d)==1, ['%d maschere corrispondono a %s / %s: il pattern non e'' ' ...
                     'univoco.\n  pattern: %s\n  trovate: %s'], ...
                     numel(d), roi, fam, pat, strjoin({d.name}, ', '));
img = fullfile(d(1).folder, d(1).name);

if endsWith(img, '.gz')
    workDir = cfg.workDir(subj);
    if ~exist(workDir,'dir'), mkdir(workDir); end
    [~, base] = fileparts(d(1).name);          % toglie .gz, resta *.nii
    plain = fullfile(workDir, base);
    if ~exist(plain,'file')
        gunzip(img, workDir);
    end
    img = plain;
end

% Intersezione con la maschera di analisi del primo livello.
% SPM scrive NaN nei beta_*.nii FUORI da mask.nii; l'aggiustamento della VOI
% legge quei beta, quindi un solo voxel della ROI fuori maschera fa fallire
% spm_regions con "Data contain NaN or Inf". spm_imcalc ricampiona inoltre la
% ROI sulla griglia di mask.nii, che e' quella su cui il GLM e' stato stimato.
anaMask = fullfile(cfg.firstLevelDir(subj), 'mask.nii');
if exist(anaMask,'file') ~= 2
    anaMask = fullfile(cfg.firstLevelDir(subj), 'mask.img');
end
assert(exist(anaMask,'file')==2, ...
       'Maschera di analisi non trovata in %s (attesa mask.nii).', ...
       cfg.firstLevelDir(subj));

workDir = cfg.workDir(subj);
if ~exist(workDir,'dir'), mkdir(workDir); end
[~, rbase] = fileparts(img);
outImg = fullfile(workDir, sprintf('%s_inmask.nii', rbase));

if ~exist(outImg,'file')
    spm_imcalc({anaMask; img}, outImg, '(i1>0) .* (i2>0.5)', ...
               struct('interp', 0, 'dtype', spm_type('uint8')));
end

nAfter = nnz(spm_read_vols(spm_vol(outImg)) > 0);
assert(nAfter > 0, ...
      ['La ROI %s / %s non ha alcun voxel dentro la maschera di analisi.\n' ...
       '  ROI:      %s\n  maschera: %s\n' ...
       'Di solito vuol dire che le due sono in spazi diversi.'], ...
       roi, fam, img, anaMask);

if verbose
    nBefore = nnz(spm_read_vols(spm_vol(img)) > 0.5);
    fprintf('  ROI %-8s %-10s  %d voxel -> %d dentro mask.nii (%.0f%%)\n', ...
            roi, fam, nBefore, nAfter, 100*nAfter/max(nBefore,1));
    fprintf('       %s\n', outImg);
end
if nAfter < 20
    warning(['ROI %s / %s: solo %d voxel sopravvivono all''intersezione con ' ...
             'la maschera di analisi. Controlla allineamento e copertura ' ...
             'dello slab.'], roi, fam, nAfter);
end

img = outImg;
end

function voiFiles = extract_voi(spmPath, Feoi, voiName, roiImage, nSess)
% Una VOI per sessione. SPM nomina i file VOI_<name>_<sessione>.mat
outDir   = fileparts(spmPath);
voiFiles = cell(1, nSess);
assert(exist(roiImage,'file')==2, 'Maschera ROI non trovata: %s', roiImage);

for s = 1:nSess
    voiFiles{s} = fullfile(outDir, sprintf('VOI_%s_%d.mat', voiName, s));
    if exist(voiFiles{s},'file'), continue, end

    clear matlabbatch
    matlabbatch{1}.spm.util.voi.spmmat  = {spmPath};
    matlabbatch{1}.spm.util.voi.adjust  = Feoi;   % toglie i nuisance dalla serie
    matlabbatch{1}.spm.util.voi.session = s;
    matlabbatch{1}.spm.util.voi.name    = voiName;
    matlabbatch{1}.spm.util.voi.roi{1}.mask.image     = {roiImage};
    matlabbatch{1}.spm.util.voi.roi{1}.mask.threshold = 0.5;
    matlabbatch{1}.spm.util.voi.expression = 'i1';   % nessuna soglia statistica:
    spm_jobman('run', matlabbatch);                  % la ROI e' definita a priori

    % la convenzione di naming delle VOI e' cambiata fra versioni di SPM:
    % invece di fidarmi, controllo che il file che mi aspetto esista davvero
    if ~exist(voiFiles{s},'file')
        cand = dir(fullfile(outDir, sprintf('VOI_%s*.mat', voiName)));
        error(['VOI attesa non trovata: %s\n  File VOI_%s* presenti: %s\n' ...
               '  La tua versione di SPM usa un naming diverso: adegua ' ...
               'extract_voi.'], voiFiles{s}, voiName, ...
               strjoin({cand.name}, ', '));
    end
end
end

function check_roi_overlap(cfg, subj, fam, roiImg)
% Un voxel condiviso fra seed e target produce accoppiamento spurio.
n = numel(cfg.rois);
for a = 1:n-1
    Va = spm_read_vols(spm_vol(roiImg{a})) > 0.5;
    for b = a+1:n
        Vb = spm_read_vols(spm_vol(roiImg{b})) > 0.5;
        ov = nnz(Va & Vb);
        if ov > 0
            warning(['%s | %s : %s e %s condividono %d voxel. ' ...
                     'Erodi le maschere prima di fidarti del risultato.'], ...
                     subj, fam, cfg.rois{a}, cfg.rois{b}, ov);
        end
    end
end
end


% =========================================================================
% TERMINI PPI  (una sessione per volta)
% =========================================================================
function [ppi, names, Yphys] = build_ppi_terms(spmPath, voiFile, idx, cfg, seed, fam, sess)
% Uu e' una matrice n x 3: [indice condizione, colonna dentro la condizione, peso]
% Righe multiple con pesi -> spm_peb_ppi costruisce il regressore psicologico
% come combinazione pesata delle condizioni. Da qui:
%   pesi [1 1 1] su una famiglia -> effetto medio della famiglia
%   pesi [-1 0 +1]               -> trend lineare, gia' centrato

b  = idx.baseline(sess);
m  = idx.mirroring(:,sess);
sp = idx.spacing(:,sess);

nm_ = numel(m); ns_ = numel(sp);
spec = { ...
    'base', [b,  1,            1          ] ; ...
    'mir',  [m,  ones(nm_,1),  ones(nm_,1)] ; ...
    'spa',  [sp, ones(ns_,1),  ones(ns_,1)] };

if cfg.model == 2
    wm = cfg.mirroringTrendWeights(:);
    ws = cfg.spacingTrendWeights(:);
    assert(numel(wm)==nm_ && numel(ws)==ns_, ...
           'I pesi del trend devono essere tanti quanti i livelli (%d mir, %d spa).', ...
           nm_, ns_);
    assert(abs(sum(wm)) < 1e-12 && abs(sum(ws)) < 1e-12, ...
           'I pesi del trend devono sommare a zero.');
    spec = [spec ; ...
        {'mirTrend', [m,  ones(nm_,1), wm]} ; ...
        {'spaTrend', [sp, ones(ns_,1), ws]} ];
end

ppi = []; names = cell(1, size(spec,1)); Yphys = [];
for k = 1:size(spec,1)
    nm = sprintf('%s_%s_%s_run%d', seed, fam, spec{k,1}, sess);
    P  = spm_peb_ppi(spmPath, 'ppi', voiFile, spec{k,2}, nm, 0);
    ppi(:,k) = P.ppi;                        %#ok<AGROW>
    names{k} = sprintf('PPI_%s', spec{k,1});
    if isempty(Yphys), Yphys = P.Y; end      % identica per tutti i termini
end
end


% =========================================================================
% SPECIFICA E STIMA  (multi-sessione)
% =========================================================================
function specify_design(outDir, SPMsrc, cfg, subj, regFiles)
% Ricostruisce le condizioni di ogni sessione dal primo livello originale e
% vi aggiunge i regressori PPI di quella sessione. I regressori psicologici
% restano tutti nel modello, che e' il punto della gPPI.

nSess = numel(SPMsrc.Sess);

% Le basi devono essere le stesse dell'originale, altrimenti il modello gPPI
% non e' lo stesso modello con in piu' le interazioni.
switch SPMsrc.xBF.name
    case 'hrf',                                          derivs = [0 0];
    case 'hrf (with time derivative)',                   derivs = [1 0];
    case 'hrf (with time and dispersion derivatives)',   derivs = [1 1];
    otherwise
        error(['Basi non supportate: "%s". Questo script replica solo le ' ...
               'basi HRF canoniche.'], SPMsrc.xBF.name);
end

% xVi.form dopo la stima diventa tipo "AR(0.2)", ma il batch accetta solo
% 'AR(1)', 'FAST' o 'None': rimappo.
if strncmpi(SPMsrc.xVi.form, 'AR', 2)
    cvi = 'AR(1)';
elseif strcmpi(SPMsrc.xVi.form, 'FAST')
    cvi = 'FAST';
else
    cvi = 'None';
end

% Maschera esplicita: se il primo livello ne usa una, il modello gPPI deve
% usare LA STESSA, altrimenti i due coprono volumi diversi e i profili
% laminari finirebbero campionati su voxel che nelle attivazioni non esistono.
explicitMask = {''};
if isfield(SPMsrc,'xM') && isfield(SPMsrc.xM,'VM') && ~isempty(SPMsrc.xM.VM)
    if numel(SPMsrc.xM.VM) > 1
        warning(['Il primo livello usa %d maschere esplicite; il batch ne ' ...
                 'accetta una sola, uso la prima.'], numel(SPMsrc.xM.VM));
    end
    mf = SPMsrc.xM.VM(1).fname;
    assert(exist(mf,'file')==2, ...
        ['Il primo livello usa la maschera esplicita\n  %s\n' ...
         'che non esiste piu''. Sistemala (spm_changepath) prima di continuare.'], mf);
    explicitMask = {mf};
    fprintf('  maschera esplicita ereditata: %s\n', mf);
end

clear matlabbatch
b = struct();
b.dir            = {outDir};
b.timing.units   = SPMsrc.xBF.UNITS;
b.timing.RT      = SPMsrc.xY.RT;
b.timing.fmri_t  = SPMsrc.xBF.T;
b.timing.fmri_t0 = SPMsrc.xBF.T0;

for s = 1:nSess
    scans = session_scans(SPMsrc, cfg, subj, s);
    b.sess(s).scans = scans;

    U = SPMsrc.Sess(s).U;
    for i = 1:numel(U)
        % Una condizione con modulatori parametrici o temporali ha piu' di un
        % nome in U(i).name. Ricostruirla ignorando i pmod darebbe un modello
        % diverso dall'originale, quindi mi fermo invece di farlo in silenzio.
        assert(numel(U(i).name) == 1, ...
            ['Sessione %d, condizione "%s": ha modulatori parametrici o ' ...
             'temporali. Questo script non li ricostruisce.'], ...
             s, char(U(i).name{1}));
        b.sess(s).cond(i).name     = U(i).name{1};
        b.sess(s).cond(i).onset    = U(i).ons;
        b.sess(s).cond(i).duration = U(i).dur;
        b.sess(s).cond(i).tmod     = 0;
        b.sess(s).cond(i).pmod     = struct('name',{},'param',{},'poly',{});
        b.sess(s).cond(i).orth     = 1;
    end

    b.sess(s).multi   = {''};
    b.sess(s).regress = struct('name',{},'val',{});

    mr = regFiles(s)';
    nf = cfg.nuisanceFile(subj, s);
    if ~isempty(nf)
        assert(exist(nf,'file')==2, 'Sessione %d: nuisance non trovato: %s', s, nf);
        mr = [mr ; {nf}];   %#ok<AGROW>
    end
    b.sess(s).multi_reg = mr;
    b.sess(s).hpf       = SPMsrc.xX.K(s).HParam;
end

b.fact             = struct('name',{},'levels',{});
b.bases.hrf.derivs = derivs;
b.volt             = 1;
b.global           = SPMsrc.xGX.iGXcalc;
b.mthresh          = cfg.maskThreshold;
b.mask             = explicitMask;
b.cvi              = cvi;

matlabbatch{1}.spm.stats.fmri_spec = b;
spm_jobman('run', matlabbatch);
end

function scans = session_scans(SPMsrc, cfg, subj, s)
% I volumi di una sessione. Di default li prende da SPM.xY.P del primo
% livello: sono esattamente i file usati, nell'ordine giusto, gia' con il
% suffisso di frame ",n". Nessun pattern da azzeccare, nessun rischio di
% ordine o conteggio diversi.
rows = SPMsrc.Sess(s).row;

useCfg = isfield(cfg,'scans') && ~isempty(cfg.scans);
if ~useCfg
    P = SPMsrc.xY.P;
    if ischar(P), scans = cellstr(P(rows,:)); else, scans = P(rows); end
    scans = strtrim(scans);

    f = strtok(scans{1}, ',');
    assert(exist(f,'file')==2, ...
        ['Sessione %d: il volume elencato in SPM.xY.P non esiste:\n  %s\n' ...
         'Sistema i percorsi con spm_changepath.'], s, f);
    return
end

% override esplicito: allora il conteggio deve tornare
scans = cellstr(cfg.scans(subj, s));
assert(numel(scans) == numel(rows), ...
    ['Sessione %d: cfg.scans restituisce %d volumi ma il primo livello ne ' ...
     'usa %d.\n  Se i dati sono un NIfTI 4D, spm_select(''ExtFPList'',...) ' ...
     'non sempre espande i frame:\n  togli cfg.scans e lo script li prende ' ...
     'da SPM.xY.P.'], s, numel(scans), numel(rows));
end

function estimate_design(outDir)
clear matlabbatch
matlabbatch{1}.spm.stats.fmri_est.spmmat = {fullfile(outDir,'SPM.mat')};
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
spm_jobman('run', matlabbatch);
end


% =========================================================================
% REVISIONE DELLA MATRICE DI DISEGNO
% =========================================================================
function report_design(outDir, cfg, subj, fam, seed)
% Stampa le colonne raggruppate per sessione con il ruolo di ciascuna, e
% verifica che la composizione sia quella attesa. Poi, se richiesto, apre
% spm_DesRep.
S = load(fullfile(outDir,'SPM.mat')); SPM = S.SPM;
nm    = SPM.xX.name(:);
nSess = numel(SPM.Sess);

nPPI   = 3 + 2*(cfg.model == 2);
nOther = (numel(cfg.rois)-1) * double(logical(cfg.controlOtherRois));

fprintf('\n=== matrice di disegno: %s | %s | seed %s ===\n', subj, fam, seed);
allOK = true;

for s = 1:nSess
    tok = regexp(nm, '^Sn\((\d+)\)', 'tokens', 'once');
    isS = cellfun(@(t) ~isempty(t) && str2double(t{1})==s, tok);
    cols = find(isS);

    fprintf('\n-- sessione %d : %d volumi, %d colonne --\n', ...
            s, numel(SPM.Sess(s).row), numel(cols));

    cnt = struct('cond',0,'seed',0,'ppi',0,'other',0,'nuis',0,'const',0);
    for c = cols(:)'
        [role, key] = classify_column(nm{c}, seed, cfg);
        cnt.(key) = cnt.(key) + 1;
        fprintf('  %3d  %-34s %s\n', c, nm{c}, role);
    end

    ok = (cnt.seed == 1) && (cnt.ppi == nPPI) && (cnt.other == nOther);
    allOK = allOK && ok;
    fprintf(['  atteso: 1 seed, %d PPI, %d altre ROI   ->   ' ...
             'trovati: %d, %d, %d   %s\n'], ...
             nPPI, nOther, cnt.seed, cnt.ppi, cnt.other, ternary(ok,'OK','<<< NO'));
end

% le costanti di sessione non hanno il prefisso Sn(): le conto a parte
nConst = sum(contains(nm, 'constant'));
fprintf('\ncostanti di sessione: %d (attese %d)  %s\n', ...
        nConst, nSess, ternary(nConst==nSess,'OK','<<< NO'));

if ~allOK
    warning(['La composizione della matrice non e'' quella attesa. Non ' ...
             'stimare finche'' non torna: i contrasti userebbero colonne ' ...
             'sbagliate.']);
end

if cfg.showDesRep
    try
        spm_DesRep('DesMtx', SPM.xX, [], SPM.xsDes);
    catch ME
        warning(ME.identifier,'spm_DesRep non ha funzionato (%s). Il resoconto sopra basta.', ...
                ME.message);
    end
end
end

function [role, key] = classify_column(name, seed, cfg)
if contains(name, '*bf(')
    role = 'condizione';                      key = 'cond';
elseif contains(name, ['phys_' seed])
    role = '<-- SEED (fisiologico)';          key = 'seed';
elseif contains(name, 'PPI_')
    role = '<-- INTERAZIONE';                 key = 'ppi';
elseif any(cellfun(@(r) contains(name, ['phys_' r]), cfg.rois))
    role = '<-- altra ROI (nuisance)';        key = 'other';
elseif contains(name, 'constant')
    role = 'costante';                        key = 'const';
else
    role = 'nuisance';                        key = 'nuis';
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end


% =========================================================================
% CONTRASTI  (media fra sessioni)
% =========================================================================
function define_contrasts(outDir, cfg)
spmPath = fullfile(outDir,'SPM.mat');
S = load(spmPath); SPM = S.SPM;
nCol  = size(SPM.xX.X, 2);
nSess = numel(SPM.Sess);

cons = { ...
  'PPI mir > base',  {'PPI_mir', 1; 'PPI_base', -1} ; ...
  'PPI spa > base',  {'PPI_spa', 1; 'PPI_base', -1} ; ...
  'PPI base',        {'PPI_base', 1} };          % diagnostica venosa

if cfg.model == 2
    cons = [cons ; ...
      {'PPI mir trend', {'PPI_mirTrend', 1}} ; ...
      {'PPI spa trend', {'PPI_spaTrend', 1}} ];
end

clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spmPath};
for i = 1:size(cons,1)
    matlabbatch{1}.spm.stats.con.consess{i}.tcon.name    = cons{i,1};
    matlabbatch{1}.spm.stats.con.consess{i}.tcon.weights = ...
        build_contrast(nCol, nSess, cons{i,2}, SPM.xX.name);
    matlabbatch{1}.spm.stats.con.consess{i}.tcon.sessrep = 'none';
end
matlabbatch{1}.spm.stats.con.delete = 1;
spm_jobman('run', matlabbatch);
end

function c = build_contrast(nCol, nSess, pairs, colNames)
% Prende TUTTE le colonne omologhe delle sessioni e divide per il loro
% numero: il contrasto e' la media fra run, non la somma.
c = zeros(1, nCol);
for i = 1:size(pairs,1)
    pat = ['\<' regexptranslate('escape', pairs{i,1}) '(_\d+)?$'];
    j   = find(~cellfun(@isempty, regexp(colNames, pat, 'once')));
    assert(~isempty(j), 'Regressore "%s" non trovato nella matrice.', pairs{i,1});
    assert(numel(j) == nSess, ...
        'Regressore "%s": trovate %d colonne ma le sessioni sono %d.', ...
        pairs{i,1}, numel(j), nSess);
    c(j) = pairs{i,2} / nSess;
end
end