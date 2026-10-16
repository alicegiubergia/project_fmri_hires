function cfg = gppi_laminar_config()
% Configurazione per GPPI_LAMINAR.
%
% Percorsi adattati al progetto. Restano da verificare, quando si passera' al
% modello 2, solo i pesi del trend sul mirroring (vedi in fondo).

% -------------------------------------------------------------------------
% QUALE MODELLO
% -------------------------------------------------------------------------
% 1 / 'family' : 3 interazioni (baseline, mirroring, spacing)
%                -> "quale ROI guida quale"
% 2 / 'trend'  : 5 interazioni (le tre sopra + trend lineare per famiglia)
%                -> "il bilancio FF/FB cambia con la degradazione"
% Gira prima il modello 1 su tutti i soggetti. Se non c'e' segnale li',
% il modello 2 non lo trovera'.
cfg.model = 1;

% -------------------------------------------------------------------------
% SOGGETTI E PERCORSI
% -------------------------------------------------------------------------
% Per la prima revisione lascia un soggetto solo.
cfg.subjects = {'sub-001'};
% cfg.subjects = {'sub-001','sub-002','sub-003','sub-004', ...
%                 'sub-005','sub-006','sub-007','sub-008'};

% Il numero di run NON si configura: viene letto da numel(SPM.Sess) del
% primo livello originale, cosi' non puo' andare fuori sincrono.

root = '/mnt/store1/projects/progetto_layer/data/';

% Cartella del primo livello già stimato (attivazioni).
cfg.firstLevelDir = @(subj) fullfile(root, 'derivatives', 'hires_pipeline', subj, 'stats-masked');

% Volumi funzionali: NON serve specificarli. Di default lo script li prende
% da SPM.xY.P del primo livello, quindi sono per costruzione gli stessi file
% nello stesso ordine, con i suffissi di frame gia' a posto.
% Definisci cfg.scans solo se vuoi puntare a dati diversi; in quel caso il
% conteggio dei volumi deve coincidere con quello del primo livello, e con un
% NIfTI 4D spm_select('ExtFPList',...) non sempre espande i frame - piu'
% sicuro elencarli con spm_vol.
cfg.scans = [];

% File dei nuisance PER RUN (outlier ART + parametri di movimento).
% Il numero di colonne puo' differire fra run: va bene.
cfg.nuisanceFile = @(subj, run) fullfile(root,'derivatives','hires_pipeline',subj,'func', ...
                sprintf('art_regression_outliers_and_movement_%s_task-reading_run-%02d_space-anat_desc-preproc_bold.mat', subj, run));
% Maschere ROI. Nei tuoi file la famiglia non compare per nome ma come
% stringa spmT: il contrasto 111 del mirroring e' spmT_0008, quello dello
% spacing spmT_0009.
cfg.roiSpmT = struct('mirroring','spmT_0008', 'spacing','spmT_0009');

% Pattern con wildcard: lo script lo espande con dir() e PRETENDE UNA SOLA
% corrispondenza, cosi' un pattern ambiguo si ferma invece di pescare a caso.
% Al primo soggetto stampa il file che ha risolto per ogni ROI: controllalo.
% L'ordine dei due campi dentro il nome e' un SEGNAPOSTO - non so dove cada
% il nome della ROI nel tuo naming, quindi ci sono wildcard da entrambi i lati.
cfg.roiPattern = @(subj, roi, spmt) fullfile(root,'derivatives','rois',subj, ...
                sprintf('*%s*_label-%s_mask.nii*', spmt, roi));

% SPM non legge i .nii.gz: le maschere compresse vengono scompattate qui,
% e qui finisce anche la loro intersezione con mask.nii del primo livello.
% Una volta sola, poi riusate. Gli originali non vengono toccati.
cfg.workDir = @(subj) fullfile(root,'derivatives','rois',subj,'spm_ready');

% Dove finiscono i modelli gPPI.
cfg.outDir = @(subj, seed, fam, model) fullfile(root, 'derivatives', ...
                sprintf('gppi_model%d', model), subj, sprintf('seed-%s_loc-%s', seed, fam));

cfg.overwrite = true;

% -------------------------------------------------------------------------
% REVISIONE DELLA MATRICE
% -------------------------------------------------------------------------
% true  : specifica il PRIMO modello, stampa le colonne raggruppate per
%         sessione con il ruolo di ciascuna, verifica che la composizione sia
%         quella attesa, apre spm_DesRep, e SI FERMA senza stimare.
% false : specifica, stima e definisce i contrasti per tutti.
% Lancia la prima volta con true su un soggetto solo.
cfg.reviewDesign = false;
cfg.showDesRep   = true;

% -------------------------------------------------------------------------
% ROI
% -------------------------------------------------------------------------
cfg.rois        = {'visual','dorsal','ventral'};
cfg.roiFamilies = {'mirroring','spacing'};   % i due set di localizzazione

% Serie temporali delle altre due ROI come fisiologici di non interesse.
% Tolgono la fluttuazione globale e vascolare condivisa dal conto del seed.
% (Mia aggiunta, non e' nel paper di Sharoh.)
cfg.controlOtherRois = true;

% -------------------------------------------------------------------------
% CONDIZIONI
% -------------------------------------------------------------------------
% I nomi devono corrispondere ESATTAMENTE a SPM.Sess(1).U(i).name{1}.
% Al primo soggetto lo script stampa l'elenco che trova: controllalo.
cfg.cond.baseline  = 'baseline';
cfg.cond.mirroring = {'flip_LRUD','flip_LR','flip_UD'};   % ordine crescente
cfg.cond.spacing   = {'spacing_2','spacing_4','spacing_6'};

% Pesi del trend lineare (solo modello 2). Devono sommare a zero.
% Sullo spacing e' naturale: 2 < 4 < 6 e' una scala ordinale.
% Sul mirroring segue l'ordine di degradazione LRUD < LR < UD che avevi
% stabilito; e' una TUA assunzione da dichiarare nei metodi, perche' i tre
% livelli non sono tre gradi della stessa manipolazione.
cfg.spacingTrendWeights   = [-1 0 +1];
cfg.mirroringTrendWeights = [-1 0 +1];

% -------------------------------------------------------------------------
% ALTRO
% -------------------------------------------------------------------------
% Soglia di maschera implicita. Con copertura a slab conviene abbassarla
% per non perdere voxel ai bordi.
cfg.maskThreshold = 0.2;
end