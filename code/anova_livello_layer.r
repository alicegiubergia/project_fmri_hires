#!/usr/bin/env Rscript
# ============================================================
# ANOVA sui profili laminari:
#     psc ~ b0 + b_lv * livello + b_layer * layer + b_int * livello:layer
# in versione a misure ripetute (LMM): soggetto come effetto random e
# residui AR(1) lungo la profondita'.
#
# Tabella con F, gradi di liberta', p ed eta quadro parziale per:
#   livello        -> il grado di degradazione sposta il profilo in verticale?
#   layer          -> il profilo cambia lungo la profondita'?
#   livello:layer  -> la FORMA del profilo cambia con la degradazione?
#
# Uso:  Rscript anova_livello_layer.R <OUT_DIR>/profili_long_<DENOMINATOR_MODE>.csv
# Input: lo stesso CSV usato da modelli_laminari_ar1.R
# Pacchetti: solo nlme
#
# Versione .Rmd equivalente: anova_livello_layer.Rmd (stessi risultati, con
# tabelle formattate e commento esteso).
# ============================================================

suppressPackageStartupMessages(library(nlme))

# I test SEQUENZIALI (tipo I) non dipendono dalla codifica dei fattori: i sums
# of squares derivano dalla successione dei modelli annidati, non dalla
# parametrizzazione. La riga sotto conta solo con TEST_TYPE = "marginal", dove
# serve perche' gli effetti principali siano medie marginali e non valori al
# livello di riferimento.
options(contrasts = c("contr.sum", "contr.poly"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1)
  stop("Uso: Rscript anova_livello_layer.R <profili_long_<mode>.csv>")
CSV      <- args[1]
OUT_DIR  <- dirname(CSV)
TAG      <- sub("^profili_long_?", "", sub("\\.csv$", "", basename(CSV)))
SUFFIX   <- if (nzchar(TAG)) paste0("_", TAG) else ""
out_path <- function(stem, ext) file.path(OUT_DIR, paste0(stem, SUFFIX, ".", ext))

# ============================================================
# CONFIGURAZIONE
# ============================================================

NR_BIN <- 10

# Come entra il livello (grado di degradazione):
#   "numeric" -> covariata continua dalla colonna `x` del CSV (gia' centrata
#                per famiglia: 1,2,3 per mirroring; 2,4,6 per spacing).
#                Costa 1 df invece di 2, e l'interazione 9 invece di 18.
#                Assume effetto MONOTONO e LINEARE nel grado di degradazione.
#   "factor"  -> tre livelli non ordinati, nessuna assunzione sulla forma.
LIVELLO_MODE <- "numeric"

# Come entra la profondita':
#   "factor"       -> 10 livelli, nessuna ipotesi sulla forma.
#   "compartments" -> 3 compartimenti (deep/middle/superficial): l'interazione
#                     scende a 2 df (numeric) o 4 (factor), molto piu' potente,
#                     ma assume che i confini scelti siano quelli giusti.
LAYER_MODE <- "compartments"

# Solo per LAYER_MODE = "compartments": mediare i bin dentro ogni compartimento
# prima di stimare. Consigliato TRUE: altrimenti la forma del profilo DENTRO il
# compartimento resta nei residui, l'AR(1) la assorbe, phi sale vicino a 1 e i
# test perdono potenza.
AGGREGA_COMPARTIMENTI <- TRUE

# [PLACEHOLDER] Confini dei compartimenti: sono una mia scelta, non derivata
# dai tuoi dati. Con 10 bin equidistanti 1..10 (1 = WM, 10 = CSF) una
# divisione simmetrica e' 1-3 / 4-7 / 8-10. Verifica che corrisponda a come
# hai definito deep/middle/superficial nello schema originale.
COMPARTIMENTI <- c(rep("deep", 4), rep("middle", 2), rep("superficial", 4))

# "sequential" (tipo I): ogni termine al netto di quelli che lo PRECEDONO nella
#   formula -> livello, poi layer dato livello, poi l'interazione.
# "marginal" (tipo III): ogni termine al netto di TUTTI gli altri.
# L'interazione e' identica nei due casi; cambiano gli effetti principali.
TEST_TYPE <- "sequential"

CTRL <- lmeControl(opt = "optim", maxIter = 500, msMaxIter = 500)

NUMERICO <- (LIVELLO_MODE == "numeric")

# ============================================================
# DATI
# ============================================================

df <- read.csv(CSV, stringsAsFactors = FALSE)
df <- df[!is.na(df$psc), ]

# Solo i tre livelli di degradazione. con_0001 resta fuori: e' un contrasto
# ASSOLUTO (lettura parole normali > riposo) mentre gli altri sono
# differenziali, quindi nello stesso termine `livello` misurerebbe quella
# differenza di natura e non la degradazione.
df <- df[df$condition == "livello", ]

df$subject <- factor(df$subject)
df$bin     <- as.integer(df$bin)

# `livello_grp` e' SEMPRE un fattore: serve al raggruppamento degli effetti
# random e alla struttura AR(1), che richiedono una variabile di gruppo.
df$livello_grp <- factor(paste0("lv", df$level), levels = c("lv1", "lv2", "lv3"))

# `livello` e' il predittore della parte fissa: numerico oppure fattore.
df$livello <- if (NUMERICO) as.numeric(df$x) else df$livello_grp
if (NUMERICO) df <- df[is.finite(df$livello), ]

df$layer <- if (LAYER_MODE == "compartments") {
  factor(COMPARTIMENTI[df$bin], levels = c("deep", "middle", "superficial"))
} else {
  factor(df$bin, levels = seq_len(NR_BIN))
}

AGGREGATO <- (LAYER_MODE == "compartments" && AGGREGA_COMPARTIMENTI)
if (AGGREGATO) {
  df <- aggregate(psc ~ subject + region + family + livello + livello_grp + layer,
                  data = df, FUN = mean)
  df$bin <- as.integer(df$layer)   # serve solo a ordinare le osservazioni
}

cat("Modello:", if (NUMERICO) "covariata numerica (1 df)" else
    sprintf("fattore (%d df)", nlevels(df$livello_grp) - 1),
    "| Layer:", LAYER_MODE, "(", nlevels(df$layer), "livelli )",
    if (AGGREGATO) "- bin mediati" else "", "\n")
cat("Test:", TEST_TYPE, "\n")

# ============================================================
# STIMA
# ============================================================

# Strutture random dalla piu' ricca alla piu' semplice: si usa la prima che
# converge. "subj/livello" da' a ogni profilo soggetto x livello la propria
# intercetta. Il raggruppamento usa sempre `livello_grp`, anche quando la
# parte fissa e' numerica.
RANDOM <- list(
  "subj/livello" = ~ 1 | subject/livello_grp,
  "subj"         = ~ 1 | subject
)

fit_uno <- function(formula, data, random, correlation, metodo) {
  tryCatch(
    lme(fixed = formula, data = data, random = random,
        correlation = correlation, method = metodo, control = CTRL),
    error = function(e) NULL
  )
}

fitta <- function(formula, data, metodo) {
  for (nome in names(RANDOM)) {
    # con i bin gia' mediati restano 3 punti per profilo: un AR(1) su 3 punti
    # non ha senso, quindi si salta
    corr <- if (AGGREGATO)                   NULL
            else if (nome == "subj/livello") corAR1(form = ~ bin | subject/livello_grp)
            else                             corAR1(form = ~ bin | subject)
    f <- fit_uno(formula, data, RANDOM[[nome]], corr, metodo)
    if (!is.null(f)) return(list(fit = f, random = nome))
    # senza AR1, come ultima risorsa
    f <- fit_uno(formula, data, RANDOM[[nome]], NULL, metodo)
    if (!is.null(f)) return(list(fit = f, random = paste0(nome, " (senza AR1)")))
  }
  NULL
}

phi_di <- function(fit) {
  cs <- fit$modelStruct$corStruct
  if (is.null(cs)) NA_real_ else unname(coef(cs, unconstrained = FALSE))
}

# eta quadro parziale, dalla F e dai suoi gradi di liberta'
eta2_p <- function(F, df1, df2) (F * df1) / (F * df1 + df2)

TERMINI <- c("livello", "layer", "livello:layer")

righe <- list(); medie <- list()

for (fam in sort(unique(df$family))) for (reg in sort(unique(df$region))) {
  g <- df[df$family == fam & df$region == reg, ]
  if (!nrow(g)) next

  # solo soggetti con tutti i livelli: il disegno resta bilanciato
  n_lv <- tapply(g$livello_grp, g$subject, function(v) length(unique(v)))
  g <- g[g$subject %in% names(n_lv)[n_lv == nlevels(g$livello_grp)], ]
  g$subject <- droplevels(g$subject); g$livello_grp <- droplevels(g$livello_grp)
  if (nlevels(g$subject) < 3) {
    cat("  [skip]", fam, reg, ": meno di 3 soggetti completi\n")
    next
  }
  g <- g[order(g$subject, g$livello_grp, g$bin), ]

  ff <- fitta(psc ~ livello * layer, g, metodo = "REML")
  if (is.null(ff)) {
    cat("  [skip]", fam, reg, ": nessuna convergenza\n")
    next
  }

  a <- anova(ff$fit, type = TEST_TYPE)

  riga <- list(family = fam, region = reg, n_soggetti = nlevels(g$subject),
               n = nrow(g), random = ff$random, phi = phi_di(ff$fit))
  for (t in TERMINI) {
    if (!t %in% rownames(a)) next
    F <- a[t, "F-value"]; d1 <- a[t, "numDF"]; d2 <- a[t, "denDF"]
    chiave <- gsub(":", "_x_", t)
    riga[[paste0("F_", chiave)]]     <- F
    riga[[paste0("numDF_", chiave)]] <- d1
    riga[[paste0("denDF_", chiave)]] <- d2
    riga[[paste0("p_", chiave)]]     <- a[t, "p-value"]
    riga[[paste0("eta2p_", chiave)]] <- eta2_p(F, d1, d2)
  }

  # controllo incrociato sull'interazione con un LRT (richiede ML)
  m_pieno  <- fitta(psc ~ livello * layer, g, metodo = "ML")
  m_ridott <- fitta(psc ~ livello + layer, g, metodo = "ML")
  if (!is.null(m_pieno) && !is.null(m_ridott) &&
      identical(m_pieno$random, m_ridott$random)) {
    riga$lrt_interazione_p <- anova(m_ridott$fit, m_pieno$fit)[["p-value"]][2]
  }

  righe[[length(righe) + 1]] <- as.data.frame(riga, stringsAsFactors = FALSE)

  mm <- aggregate(psc ~ livello_grp + layer, data = g, FUN = mean)
  se <- aggregate(psc ~ livello_grp + layer, data = g,
                  FUN = function(v) sd(v) / sqrt(length(v)))
  mm$se <- se$psc; mm$family <- fam; mm$region <- reg
  medie[[length(medie) + 1]] <- mm
}

ris <- do.call(rbind, righe)
md  <- do.call(rbind, medie)
if (is.null(ris)) stop("Nessun modello stimato")

# ============================================================
# TABELLE
# ============================================================

fmt_p <- function(p) ifelse(is.na(p), "-",
                    ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
stelle <- function(p) ifelse(is.na(p), "",
                     ifelse(p < 0.001, "***",
                     ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*",
                     ifelse(p < 0.10, ".", "")))))
con_stelle <- function(p) paste0(fmt_p(p), stelle(p))

cat("\n", strrep("=", 78),
    "\n ANOVA  psc ~ livello * layer  (LMM",
    if (AGGREGATO) ", senza AR1: bin mediati, 3 punti per profilo"
    else           ", AR1 sui bin",
    ")\n", strrep("=", 78), "\n", sep = "")

sintesi <- data.frame(
  famiglia        = ris$family,
  ROI             = ris$region,
  N               = ris$n_soggetti,
  phi             = ifelse(is.na(ris$phi), "-", sprintf("%.2f", ris$phi)),
  livello         = con_stelle(ris$p_livello),
  layer           = con_stelle(ris$p_layer),
  `livello:layer` = con_stelle(ris$p_livello_x_layer),
  LRT             = con_stelle(ris$lrt_interazione_p),
  check.names = FALSE
)
print(sintesi, row.names = FALSE)
cat("\n*** p<0.001  ** p<0.01  * p<0.05  . p<0.10\n")
cat("df di `livello`:", ris$numDF_livello[1],
    "| df di `livello:layer`:", ris$numDF_livello_x_layer[1], "\n")

write.csv(ris, out_path("anova_livello_layer_R", "csv"), row.names = FALSE)
write.csv(md,  out_path("anova_medie_celle_R", "csv"), row.names = FALSE)

# ============================================================
# GRAFICO DI INTERAZIONE
# ============================================================

famiglie <- sort(unique(md$family)); regioni <- sort(unique(md$region))
colori <- c("#4285f4", "#d1495b", "#0f9d58", "#8e44ad")

png(out_path("anova_interazione_R", "png"),
    width = 4.6 * length(regioni), height = 3.8 * length(famiglie),
    units = "in", res = 200)
par(mfrow = c(length(famiglie), length(regioni)), mar = c(4, 4, 2.5, 1),
    oma = c(0, 0, 2.5, 0))
for (fam in famiglie) for (reg in regioni) {
  g <- md[md$family == fam & md$region == reg, ]
  if (!nrow(g)) { plot.new(); next }
  livelli   <- levels(droplevels(g$livello_grp))
  etichette <- levels(droplevels(g$layer))
  xs <- seq_along(etichette)
  plot(NA, xlim = range(xs), ylim = range(g$psc - g$se, g$psc + g$se),
       xaxt = "n", ylab = "PSC (%)",
       xlab = if (LAYER_MODE == "factor") "bin (1=WM, 10=CSF)" else "compartimento",
       main = paste(fam, reg, sep = " / "))
  axis(1, at = xs, labels = etichette)
  abline(h = 0, col = "grey60")
  for (i in seq_along(livelli)) {
    h <- g[g$livello_grp == livelli[i], ]; h <- h[order(h$layer), ]
    arrows(xs, h$psc - h$se, xs, h$psc + h$se, angle = 90, code = 3,
           length = 0.02, col = colori[i])
    lines(xs, h$psc, type = "o", pch = 16, lwd = 2, col = colori[i])
  }
  legend("topleft", legend = livelli, col = colori[seq_along(livelli)],
         lwd = 2, pch = 16, bty = "n", cex = 0.85)
}
mtext("Medie di cella +/- SE: linee non parallele = interazione livello x layer",
      outer = TRUE, font = 2)
invisible(dev.off())

cat("\nCompletato. Output in:", OUT_DIR, "\n")
