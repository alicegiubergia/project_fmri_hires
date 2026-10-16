#!/usr/bin/env Rscript
# ============================================================
# Modelli laminari di gruppo con residui AR(1) lungo la profondità
#
# Uso:  Rscript modelli_laminari_ar1.R <OUT_DIR>/profili_long_<DENOMINATOR_MODE>.csv
# Input: CSV esportato da esporta_profili.py
#        colonne: subject, region, family, roi_key, condition, contrast, level, bin, psc, n_voxel, x
#        condition = "livello" (level 1..3) oppure "baseline" (con_0001, level 0)
# Pacchetti: solo nlme (incluso nella distribuzione standard di R)
# ============================================================

suppressPackageStartupMessages(library(nlme))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Uso: Rscript modelli_laminari_ar1.R <profili_long_<mode>.csv>")
CSV     <- args[1]
OUT_DIR <- dirname(CSV)
# suffisso degli output = modalita' del denominatore, cosi' voxelwise e roi_scalar non si sovrascrivono
TAG     <- sub("^profili_long_?", "", sub("\\.csv$", "", basename(CSV)))
SUFFIX  <- if (nzchar(TAG)) paste0("_", TAG) else ""
out_path <- function(stem, ext) file.path(OUT_DIR, paste0(stem, SUFFIX, ".", ext))
NR_BIN  <- 10

df <- read.csv(CSV, stringsAsFactors = FALSE)
df <- df[!is.na(df$psc), ]
df$subject <- factor(df$subject)
df$bin     <- as.integer(df$bin)

# Polinomi ortogonali calcolati una volta sui bin 1..NR_BIN:
# stessi valori per ogni sottoinsieme, coefficienti non collineari e interpretabili
P <- poly(seq_len(NR_BIN), 4)
for (k in 1:4) df[[paste0("P", k)]] <- P[df$bin, k]

CTRL <- lmeControl(opt = "optim", maxIter = 500, msMaxIter = 500)

fit_lme <- function(fixed, data, random, correlation) {
  tryCatch(
    lme(fixed = fixed, data = data, random = random, correlation = correlation,
        method = "ML", control = CTRL),
    error = function(e) NULL
  )
}

phi_di <- function(fit) {
  cs <- fit$modelStruct$corStruct
  if (is.null(cs)) NA_real_ else unname(coef(cs, unconstrained = FALSE))
}

# Fitta una famiglia di modelli annidati con la STESSA struttura random e di correlazione.
# Prova le strutture random dalla più ricca alla più semplice e usa la prima
# per cui convergono tutti i modelli (così gli AIC sono confrontabili).
fitta_famiglia <- function(formule, data, strutture_random, correlation) {
  for (nome_rand in names(strutture_random)) {
    fits <- lapply(formule, fit_lme, data = data,
                   random = strutture_random[[nome_rand]], correlation = correlation)
    if (!any(vapply(fits, is.null, logical(1)))) {
      return(list(fits = fits, random = nome_rand))
    }
  }
  NULL
}

riassumi <- function(ff, etichette, prefisso) {
  aic  <- vapply(ff$fits, AIC, numeric(1))
  vinc <- names(which.min(aic))
  riga <- etichette
  riga[[paste0(prefisso, "_vincitore")]] <- vinc
  riga[[paste0(prefisso, "_random")]]    <- ff$random
  riga[[paste0(prefisso, "_phi")]]       <- phi_di(ff$fits[[vinc]])
  for (m in names(aic)) {
    riga[[paste0(prefisso, "_aic_", m)]]   <- aic[[m]]
    riga[[paste0(prefisso, "_delta_", m)]] <- aic[[m]] - aic[[vinc]]
  }
  riga
}

# ============================================================
# DOMANDA 1 - Forma del profilo laminare (gruppo, per cella)
# celle: family x region x (livello 1..3 | baseline)
# ============================================================

FORMULE_D1 <- list(
  costante     = psc ~ 1,
  lineare      = psc ~ P1,
  quadratico   = psc ~ P1 + P2,
  cubico       = psc ~ P1 + P2 + P3,
  quarto_grado = psc ~ P1 + P2 + P3 + P4
)

RANDOM_D1 <- list(
  "int+P1+P2" = list(subject = pdDiag(~ P1 + P2)),
  "int+P1"    = list(subject = pdDiag(~ P1)),
  "int"       = list(subject = ~ 1)
)

domanda1 <- function(df) {
  cat("\n", strrep("=", 70), "\n DOMANDA 1 - Forma del profilo laminare (LMM, AR1 lungo i bin)\n",
      strrep("=", 70), "\n", sep = "")
  celle <- unique(df[, c("family", "region", "condition", "level")])
  righe <- list(); residui <- list()

  for (i in seq_len(nrow(celle))) {
    cella <- celle[i, ]
    g <- df[df$family == cella$family & df$region == cella$region &
            df$condition == cella$condition & df$level == cella$level, ]
    g <- g[order(g$subject, g$bin), ]
    if (nlevels(droplevels(g$subject)) < 3) next
    g$subject <- droplevels(g$subject)

    etichette <- as.list(cella)
    etichette$n_soggetti <- nlevels(g$subject)
    etichette$n <- nrow(g)

    ff_ind <- fitta_famiglia(FORMULE_D1, g, RANDOM_D1, correlation = NULL)
    ff_ar1 <- fitta_famiglia(FORMULE_D1, g, RANDOM_D1,
                             correlation = corAR1(form = ~ bin | subject))
    if (is.null(ff_ar1)) {
      cat("  [skip] nessuna convergenza AR1:", paste(unlist(cella), collapse = " / "), "\n")
      next
    }

    riga <- riassumi(ff_ar1, etichette, "ar1")
    if (!is.null(ff_ind)) {
      riga <- riassumi(ff_ind, riga, "ind")
      # AR1 serve? stesso modello pieno, stessa struttura random se possibile
      if (ff_ind$random == ff_ar1$random) {
        riga$delta_aic_ar1_vs_ind <- AIC(ff_ar1$fits$quarto_grado) - AIC(ff_ind$fits$quarto_grado)
      }
    }

    # LRT cubico vs quadratico sotto AR1
    lrt <- anova(ff_ar1$fits$quadratico, ff_ar1$fits$cubico)
    riga$lrt_cubico_vs_quadr_p <- lrt[["p-value"]][2]

    # Coefficienti del modello pieno (polinomi ortogonali: stime quasi indipendenti)
    tt <- summary(ff_ar1$fits$quarto_grado)$tTable
    for (term in c("P1", "P2", "P3", "P4")) {
      riga[[paste0("coef_", term)]] <- tt[term, "Value"]
      riga[[paste0("p_", term)]]    <- tt[term, "p-value"]
    }
    righe[[length(righe) + 1]] <- as.data.frame(riga, stringsAsFactors = FALSE)

    vinc <- ff_ar1$fits[[riga$ar1_vincitore]]
    residui[[length(residui) + 1]] <- data.frame(
      family = cella$family, region = cella$region, condition = cella$condition,
      level = cella$level, subject = g$subject, bin = g$bin, osservato = g$psc,
      predetto   = fitted(vinc, level = 1),
      residuo    = resid(vinc, type = "response", level = 1),
      residuo_nz = resid(vinc, type = "normalized", level = 1)   # decorrelato da AR1
    )
  }

  ris <- do.call(rbind, righe)
  cat("\nModello vincente per AIC (AR1):\n")
  print(table(paste(ris$family, ris$condition), ris$ar1_vincitore))
  if ("ind_vincitore" %in% names(ris)) {
    cat("\nModello vincente per AIC (residui indipendenti, per confronto):\n")
    print(table(paste(ris$family, ris$condition), ris$ind_vincitore))
  }
  cat("\nphi AR1 (mediana):", round(median(ris$ar1_phi, na.rm = TRUE), 3), "\n")
  list(ris = ris, residui = do.call(rbind, residui))
}

# ============================================================
# DOMANDA 2 - Effetto della difficoltà lungo la profondità
# Un modello unico per family x region su tutti i bin (i fit per singolo bin
# non possono tener conto della correlazione tra bin).
# Solo livelli 1..3 (con_0001 non ha livelli).
# ============================================================

FORMULE_D2 <- list(
  costante          = psc ~ fbin,                         # nessun effetto del livello
  lineare_comune    = psc ~ fbin + x,                     # stessa pendenza a tutte le profondità
  lineare_per_bin   = psc ~ fbin + fbin:x,                # pendenza diversa per bin
  quadratico_per_bin = psc ~ fbin + fbin:x + fbin:x2      # con 3 livelli = modello saturato
)

RANDOM_D2 <- list(
  "subj+subj:level" = ~ 1 | subject/level_f,
  "subj"            = ~ 1 | subject
)

domanda2 <- function(df) {
  cat("\n", strrep("=", 70), "\n DOMANDA 2 - Risposta all'aumento di difficoltà (LMM, AR1)\n",
      strrep("=", 70), "\n", sep = "")
  lv <- df[df$condition == "livello", ]
  righe <- list(); pendenze <- list()

  for (fam in unique(lv$family)) for (reg in unique(lv$region)) {
    g <- lv[lv$family == fam & lv$region == reg, ]
    completi <- tapply(g$level, g$subject, function(v) length(unique(v)))
    g <- g[g$subject %in% names(completi)[completi == 3], ]
    g$subject <- droplevels(g$subject)
    if (nlevels(g$subject) < 3) next

    g$fbin    <- factor(g$bin)
    g$level_f <- factor(g$level)
    g$x2      <- g$x^2
    g <- g[order(g$subject, g$level, g$bin), ]

    ff <- fitta_famiglia(FORMULE_D2, g, RANDOM_D2,
                         correlation = corAR1(form = ~ bin | subject/level_f))
    if (is.null(ff)) {
      cat("  [skip] nessuna convergenza:", fam, reg, "\n")
      next
    }
    riga <- riassumi(ff, list(family = fam, region = reg, n_soggetti = nlevels(g$subject)), "ar1")
    lrt <- anova(ff$fits$lineare_comune, ff$fits$lineare_per_bin)
    riga$lrt_pendenza_varia_con_bin_p <- lrt[["p-value"]][2]
    righe[[length(righe) + 1]] <- as.data.frame(riga, stringsAsFactors = FALSE)

    # Pendenza del livello per bin (dal modello lineare_per_bin), con IC 95%
    tt <- summary(ff$fits$lineare_per_bin)$tTable
    idx <- grep(":x$", rownames(tt))
    pendenze[[length(pendenze) + 1]] <- data.frame(
      family = fam, region = reg,
      bin = as.integer(sub("^fbin([0-9]+):x$", "\\1", rownames(tt)[idx])),
      pendenza = tt[idx, "Value"], se = tt[idx, "Std.Error"], p = tt[idx, "p-value"],
      ci_low  = tt[idx, "Value"] - 1.96 * tt[idx, "Std.Error"],
      ci_high = tt[idx, "Value"] + 1.96 * tt[idx, "Std.Error"]
    )
  }

  ris <- do.call(rbind, righe)
  cat("\nModello vincente per AIC:\n")
  print(ris[, c("family", "region", "ar1_vincitore", "ar1_random", "ar1_phi",
                "lrt_pendenza_varia_con_bin_p")], row.names = FALSE)
  list(ris = ris, pendenze = do.call(rbind, pendenze))
}

# ============================================================
# FIGURE (base R)
# ============================================================

figura_residui <- function(res, colonna, titolo, nomefile) {
  regioni <- sort(unique(res$region))
  pannelli <- unique(res[, c("family", "condition")])
  pannelli <- pannelli[order(pannelli$family, pannelli$condition), ]
  png(out_path(nomefile, "png"), width = 4.5 * length(regioni), height = 3.6 * nrow(pannelli),
      units = "in", res = 200)
  par(mfrow = c(nrow(pannelli), length(regioni)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(pannelli))) for (reg in regioni) {
    g <- res[res$family == pannelli$family[i] & res$condition == pannelli$condition[i] &
             res$region == reg, ]
    if (!nrow(g)) { plot.new(); next }
    plot(g$bin, g[[colonna]], pch = 16, cex = 0.6, col = adjustcolor("#4285f4", 0.35),
         xlab = "bin (1=WM, 10=CSF)", ylab = colonna,
         main = paste(pannelli$family[i], pannelli$condition[i], reg, sep = " / "))
    medie <- tapply(g[[colonna]], g$bin, mean)
    lines(as.integer(names(medie)), medie, col = "#d1495b", lwd = 2, type = "o", pch = 16)
    abline(h = 0)
  }
  mtext(titolo, outer = TRUE, font = 2)
  invisible(dev.off())
}

figura_pendenze <- function(pend, nomefile) {
  famiglie <- sort(unique(pend$family)); regioni <- sort(unique(pend$region))
  png(out_path(nomefile, "png"), width = 4.5 * length(regioni), height = 3.6 * length(famiglie),
      units = "in", res = 200)
  par(mfrow = c(length(famiglie), length(regioni)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
  for (fam in famiglie) for (reg in regioni) {
    g <- pend[pend$family == fam & pend$region == reg, ]
    if (!nrow(g)) { plot.new(); next }
    g <- g[order(g$bin), ]
    plot(g$bin, g$pendenza, type = "o", pch = 16, ylim = range(g$ci_low, g$ci_high, 0),
         xlab = "bin (1=WM, 10=CSF)", ylab = "pendenza livello (%/unità x)",
         main = paste(fam, reg, sep = " / "))
    arrows(g$bin, g$ci_low, g$bin, g$ci_high, angle = 90, code = 3, length = 0.03)
    abline(h = 0, lty = 2)
  }
  mtext("Domanda 2: effetto del livello per bin (LMM, AR1)", outer = TRUE, font = 2)
  invisible(dev.off())
}

# ============================================================
# MAIN
# ============================================================

d1 <- domanda1(df)
write.csv(d1$ris,     out_path("domanda1_R", "csv"), row.names = FALSE)
write.csv(d1$residui, out_path("domanda1_residui_R", "csv"), row.names = FALSE)
figura_residui(d1$residui, "residuo",    "Residui del modello vincente (LMM + AR1)",
               "validazione_residui_R")
figura_residui(d1$residui, "residuo_nz", "Residui normalizzati (decorrelati AR1)",
               "validazione_residui_normalizzati_R")

d2 <- domanda2(df)
write.csv(d2$ris,      out_path("domanda2_R", "csv"), row.names = FALSE)
write.csv(d2$pendenze, out_path("domanda2_pendenze_R", "csv"), row.names = FALSE)
figura_pendenze(d2$pendenze, "domanda2_pendenze_R")

cat("\nCompletato. Output in:", OUT_DIR, "\n")