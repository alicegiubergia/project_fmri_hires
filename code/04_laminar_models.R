#!/usr/bin/env Rscript
# ============================================================
# 04 - Group laminar models with AR(1) residuals along depth
#
# Usage:  Rscript 04_laminar_models.R <ROOT>/laminar_psc/<METHOD>/01_profiles/profiles_<METHOD>.csv
# Input:  single CSV exported by 01_export_profiles.py
#         columns: subject, region, family, roi_key, condition, contrast, level, bin, psc, n_voxel, x,
#                  method
#         condition = "level" (level 1..3) or "baseline" (con_0001, level 0)
# Output: <ROOT>/laminar_psc/<METHOD>/04_laminar_models/, with suffix _<METHOD>
# Packages: nlme only (part of the standard R distribution)
# ============================================================

suppressPackageStartupMessages(library(nlme))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript 04_laminar_models.R <.../01_profiles/profiles_<method>.csv>")
CSV     <- args[1]
# sibling folder of 01_profiles/, inside the metric folder
OUT_DIR <- file.path(dirname(dirname(normalizePath(CSV))), "04_laminar_models")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
# output suffix = <METHOD>, taken from the CSV name:
# equidist and equivol never overwrite each other
TAG     <- sub("^profiles_?", "", sub("\\.csv$", "", basename(CSV)))
SUFFIX  <- if (nzchar(TAG)) paste0("_", TAG) else ""
out_path <- function(stem, ext) file.path(OUT_DIR, paste0(stem, SUFFIX, ".", ext))
NR_BIN  <- 10

df <- read.csv(CSV, stringsAsFactors = FALSE)
# The CSV must contain a single configuration, consistent with the file name
for (col in c("method")) {
  if (!col %in% names(df)) stop("Column '", col, "' missing: CSV from an older version? Regenerate it with 01_export_profiles.py")
  if (length(unique(df[[col]])) != 1) stop("More than one value of '", col, "' in the same CSV: ", paste(unique(df[[col]]), collapse = ", "))
}
CONFIG <- df$method[1]
if (nzchar(TAG) && TAG != CONFIG) stop("The file name (", TAG, ") does not match its content (", CONFIG, ")")
cat("Configuration:", CONFIG, "\n")
df <- df[!is.na(df$psc), ]
df$subject <- factor(df$subject)
df$bin     <- as.integer(df$bin)

# Orthogonal polynomials computed once on bins 1..NR_BIN:
# same values for every subset, non-collinear and interpretable coefficients
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

phi_of <- function(fit) {
  cs <- fit$modelStruct$corStruct
  if (is.null(cs)) NA_real_ else unname(coef(cs, unconstrained = FALSE))
}

# Fits a family of nested models with the SAME random and correlation structure.
# Tries the random structures from the richest to the simplest and uses the
# first one for which all models converge (so that AICs are comparable).
fit_family <- function(formulas, data, random_structures, correlation) {
  for (rand_name in names(random_structures)) {
    fits <- lapply(formulas, fit_lme, data = data,
                   random = random_structures[[rand_name]], correlation = correlation)
    if (!any(vapply(fits, is.null, logical(1)))) {
      return(list(fits = fits, random = rand_name))
    }
  }
  NULL
}

summarise_family <- function(ff, labels, prefix) {
  aic    <- vapply(ff$fits, AIC, numeric(1))
  winner <- names(which.min(aic))
  row    <- labels
  row[[paste0(prefix, "_winner")]] <- winner
  row[[paste0(prefix, "_random")]] <- ff$random
  row[[paste0(prefix, "_phi")]]    <- phi_of(ff$fits[[winner]])
  for (m in names(aic)) {
    row[[paste0(prefix, "_aic_", m)]]   <- aic[[m]]
    row[[paste0(prefix, "_delta_", m)]] <- aic[[m]] - aic[[winner]]
  }
  row
}

# ============================================================
# QUESTION 1 - Shape of the laminar profile (group, per cell)
# cells: family x region x (level 1..3 | baseline)
# ============================================================

FORMULAS_Q1 <- list(
  constant  = psc ~ 1,
  linear    = psc ~ P1,
  quadratic = psc ~ P1 + P2,
  cubic     = psc ~ P1 + P2 + P3,
  quartic   = psc ~ P1 + P2 + P3 + P4
)

RANDOM_Q1 <- list(
  "int+P1+P2" = list(subject = pdDiag(~ P1 + P2)),
  "int+P1"    = list(subject = pdDiag(~ P1)),
  "int"       = list(subject = ~ 1)
)

question1 <- function(df) {
  cat("\n", strrep("=", 70), "\n QUESTION 1 - Shape of the laminar profile (LMM, AR1 along bins)\n",
      strrep("=", 70), "\n", sep = "")
  cells <- unique(df[, c("family", "region", "condition", "level")])
  rows <- list(); residuals_list <- list()

  for (i in seq_len(nrow(cells))) {
    cell <- cells[i, ]
    g <- df[df$family == cell$family & df$region == cell$region &
            df$condition == cell$condition & df$level == cell$level, ]
    g <- g[order(g$subject, g$bin), ]
    if (nlevels(droplevels(g$subject)) < 3) next
    g$subject <- droplevels(g$subject)

    labels <- as.list(cell)
    labels$n_subjects <- nlevels(g$subject)
    labels$n <- nrow(g)

    ff_ind <- fit_family(FORMULAS_Q1, g, RANDOM_Q1, correlation = NULL)
    ff_ar1 <- fit_family(FORMULAS_Q1, g, RANDOM_Q1,
                         correlation = corAR1(form = ~ bin | subject))
    if (is.null(ff_ar1)) {
      cat("  [skip] no AR1 convergence:", paste(unlist(cell), collapse = " / "), "\n")
      next
    }

    row <- summarise_family(ff_ar1, labels, "ar1")
    if (!is.null(ff_ind)) {
      row <- summarise_family(ff_ind, row, "ind")
      # Is AR1 needed? same full model, same random structure if possible
      if (ff_ind$random == ff_ar1$random) {
        row$delta_aic_ar1_vs_ind <- AIC(ff_ar1$fits$quartic) - AIC(ff_ind$fits$quartic)
      }
    }

    # LRT cubic vs quadratic under AR1
    lrt <- anova(ff_ar1$fits$quadratic, ff_ar1$fits$cubic)
    row$lrt_cubic_vs_quadratic_p <- lrt[["p-value"]][2]

    # Coefficients of the full model (orthogonal polynomials: nearly independent estimates)
    tt <- summary(ff_ar1$fits$quartic)$tTable
    for (term in c("P1", "P2", "P3", "P4")) {
      row[[paste0("coef_", term)]] <- tt[term, "Value"]
      row[[paste0("p_", term)]]    <- tt[term, "p-value"]
    }
    rows[[length(rows) + 1]] <- as.data.frame(row, stringsAsFactors = FALSE)

    best <- ff_ar1$fits[[row$ar1_winner]]
    residuals_list[[length(residuals_list) + 1]] <- data.frame(
      family = cell$family, region = cell$region, condition = cell$condition,
      level = cell$level, subject = g$subject, bin = g$bin, observed = g$psc,
      predicted   = fitted(best, level = 1),
      residual    = resid(best, type = "response", level = 1),
      residual_nz = resid(best, type = "normalized", level = 1)   # AR1-decorrelated
    )
  }

  res <- do.call(rbind, rows)
  cat("\nWinning model by AIC (AR1):\n")
  print(table(paste(res$family, res$condition), res$ar1_winner))
  if ("ind_winner" %in% names(res)) {
    cat("\nWinning model by AIC (independent residuals, for comparison):\n")
    print(table(paste(res$family, res$condition), res$ind_winner))
  }
  cat("\nAR1 phi (median):", round(median(res$ar1_phi, na.rm = TRUE), 3), "\n")
  list(res = res, residuals = do.call(rbind, residuals_list))
}

# ============================================================
# QUESTION 2 - Effect of difficulty across depth
# One model per family x region over all bins (per-bin fits cannot account
# for the correlation between bins).
# Levels 1..3 only (con_0001 has no levels).
# ============================================================

FORMULAS_Q2 <- list(
  constant          = psc ~ fbin,                         # no level effect
  common_linear     = psc ~ fbin + x,                     # same slope at all depths
  linear_per_bin    = psc ~ fbin + fbin:x,                # different slope per bin
  quadratic_per_bin = psc ~ fbin + fbin:x + fbin:x2       # with 3 levels = saturated model
)

RANDOM_Q2 <- list(
  "subj+subj:level" = ~ 1 | subject/level_f,
  "subj"            = ~ 1 | subject
)

question2 <- function(df) {
  cat("\n", strrep("=", 70), "\n QUESTION 2 - Response to increasing difficulty (LMM, AR1)\n",
      strrep("=", 70), "\n", sep = "")
  lv <- df[df$condition == "level", ]
  rows <- list(); slopes <- list()

  for (fam in unique(lv$family)) for (reg in unique(lv$region)) {
    g <- lv[lv$family == fam & lv$region == reg, ]
    complete <- tapply(g$level, g$subject, function(v) length(unique(v)))
    g <- g[g$subject %in% names(complete)[complete == 3], ]
    g$subject <- droplevels(g$subject)
    if (nlevels(g$subject) < 3) next

    g$fbin    <- factor(g$bin)
    g$level_f <- factor(g$level)
    g$x2      <- g$x^2
    g <- g[order(g$subject, g$level, g$bin), ]

    ff <- fit_family(FORMULAS_Q2, g, RANDOM_Q2,
                     correlation = corAR1(form = ~ bin | subject/level_f))
    if (is.null(ff)) {
      cat("  [skip] no convergence:", fam, reg, "\n")
      next
    }
    row <- summarise_family(ff, list(family = fam, region = reg, n_subjects = nlevels(g$subject)), "ar1")
    lrt <- anova(ff$fits$common_linear, ff$fits$linear_per_bin)
    row$lrt_slope_varies_with_bin_p <- lrt[["p-value"]][2]
    rows[[length(rows) + 1]] <- as.data.frame(row, stringsAsFactors = FALSE)

    # Level slope per bin (from the linear_per_bin model), with 95% CI
    tt <- summary(ff$fits$linear_per_bin)$tTable
    idx <- grep(":x$", rownames(tt))
    slopes[[length(slopes) + 1]] <- data.frame(
      family = fam, region = reg,
      bin = as.integer(sub("^fbin([0-9]+):x$", "\\1", rownames(tt)[idx])),
      slope = tt[idx, "Value"], se = tt[idx, "Std.Error"], p = tt[idx, "p-value"],
      ci_low  = tt[idx, "Value"] - 1.96 * tt[idx, "Std.Error"],
      ci_high = tt[idx, "Value"] + 1.96 * tt[idx, "Std.Error"]
    )
  }

  res <- do.call(rbind, rows)
  cat("\nWinning model by AIC:\n")
  print(res[, c("family", "region", "ar1_winner", "ar1_random", "ar1_phi",
                "lrt_slope_varies_with_bin_p")], row.names = FALSE)
  list(res = res, slopes = do.call(rbind, slopes))
}

# ============================================================
# FIGURES (base R)
# ============================================================

plot_residuals <- function(res, column, title, file_stem) {
  regions <- sort(unique(res$region))
  panels  <- unique(res[, c("family", "condition")])
  panels  <- panels[order(panels$family, panels$condition), ]
  png(out_path(file_stem, "png"), width = 4.5 * length(regions), height = 3.6 * nrow(panels),
      units = "in", res = 200)
  par(mfrow = c(nrow(panels), length(regions)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(panels))) for (reg in regions) {
    g <- res[res$family == panels$family[i] & res$condition == panels$condition[i] &
             res$region == reg, ]
    if (!nrow(g)) { plot.new(); next }
    plot(g$bin, g[[column]], pch = 16, cex = 0.6, col = adjustcolor("#4285f4", 0.35),
         xlab = "bin (1=WM, 10=CSF)", ylab = column,
         main = paste(panels$family[i], panels$condition[i], reg, sep = " / "))
    means <- tapply(g[[column]], g$bin, mean)
    lines(as.integer(names(means)), means, col = "#d1495b", lwd = 2, type = "o", pch = 16)
    abline(h = 0)
  }
  mtext(paste0(title, "  [", CONFIG, "]"), outer = TRUE, font = 2)
  invisible(dev.off())
}

plot_slopes <- function(slopes, file_stem) {
  families <- sort(unique(slopes$family)); regions <- sort(unique(slopes$region))
  png(out_path(file_stem, "png"), width = 4.5 * length(regions), height = 3.6 * length(families),
      units = "in", res = 200)
  par(mfrow = c(length(families), length(regions)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
  for (fam in families) for (reg in regions) {
    g <- slopes[slopes$family == fam & slopes$region == reg, ]
    if (!nrow(g)) { plot.new(); next }
    g <- g[order(g$bin), ]
    plot(g$bin, g$slope, type = "o", pch = 16, ylim = range(g$ci_low, g$ci_high, 0),
         xlab = "bin (1=WM, 10=CSF)", ylab = "level slope (% per unit of x)",
         main = paste(fam, reg, sep = " / "))
    arrows(g$bin, g$ci_low, g$bin, g$ci_high, angle = 90, code = 3, length = 0.03)
    abline(h = 0, lty = 2)
  }
  mtext(paste0("Question 2: level effect per bin (LMM, AR1)  [", CONFIG, "]"), outer = TRUE, font = 2)
  invisible(dev.off())
}

# ============================================================
# MAIN
# ============================================================

q1 <- question1(df)
write.csv(q1$res,       out_path("q1_R", "csv"), row.names = FALSE)
write.csv(q1$residuals, out_path("q1_residuals_R", "csv"), row.names = FALSE)
plot_residuals(q1$residuals, "residual",    "Residuals of the winning model (LMM + AR1)",
               "residual_check_R")
plot_residuals(q1$residuals, "residual_nz", "Normalised residuals (AR1-decorrelated)",
               "residual_check_normalised_R")

q2 <- question2(df)
write.csv(q2$res,    out_path("q2_R", "csv"), row.names = FALSE)
write.csv(q2$slopes, out_path("q2_slopes_R", "csv"), row.names = FALSE)
plot_slopes(q2$slopes, "q2_slopes_R")

cat("\nDone. Output in:", OUT_DIR, "\n")