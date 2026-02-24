# ============================================================
# BIOS226 - Week 5
# Teaching script: ROC curves for common modeling pitfalls
# ============================================================
#
# This script generates synthetic tumor data and saves ROC + truth-table plots,
# illustrating the following scenarios:
# 0) "Perfect dataset" (high n, high p, strong signal)
# 1) Overfitting
# 2) Underfitting
# 3) Wrong-label training ("model messed up")
# 4) Data leakage from feature selection before splitting
# 5) Ignoring class imbalance (90:10) with random (non-stratified) split


# ============================================================
# STEP 0) Paths to support scripts
# ============================================================

generator_script <- if (file.exists("week_5/scripts/generate_tumor_dataset.R")) {
  "week_5/scripts/generate_tumor_dataset.R"
} else {
  "generate_tumor_dataset.R"
}

helper_script <- if (file.exists("week_5/scripts/model_workflow_helpers.R")) {
  "week_5/scripts/model_workflow_helpers.R"
} else {
  "model_workflow_helpers.R"
}


# ============================================================
# STEP 1) Load external function definitions
# ============================================================

source(generator_script)
source(helper_script)


# ============================================================
# STEP 2) Configuration
# ============================================================

dataset_seed_main <- 123
dataset_seed_imbalance <- 124
dataset_seed_perfect <- 125

dataset_n_samples <- 120
dataset_n_genes <- 10000
dataset_n_informative <- 50
dataset_noise_sd <- 1.5

dataset_class_proportion_main <- 0.6
dataset_class_proportion_imbalance <- 0.9
dataset_class_proportion_perfect <- 0.5

perfect_n_samples <- 240
perfect_n_genes <- 8000
perfect_n_informative <- 25

positive_class <- "Basal_like"
train_fraction <- 0.8

scenario_seeds <- list(
  perfect = 1006,
  overfitting = 1201,
  underfitting = 1002,
  wrong_labels = 1003,
  leakage = 1000,
  imbalance = 1005
)

top_k_perfect <- 25
top_k_overfitting <- 50
top_k_underfitting <- 1
top_k_leakage <- 25
top_k_imbalance <- 25

k_folds <- 5
threshold_for_truth_table <- 0.5
wrong_label_auc_warning_cutoff <- 0.5

# PNG export size (16:9 slide-like aspect ratio).
export_png_width_px <- 1280
export_png_height_px <- 720

# Live plotting behavior (RStudio Plot pane / current graphics device).
show_plots_live <- TRUE
open_new_device_for_live_plots <- FALSE


# ============================================================
# STEP 3) Local helper functions
# ============================================================

table_to_string <- function(x) {
  if (length(x) == 0) {
    return("none")
  }
  paste(paste(names(x), as.integer(x), sep = ":"), collapse = ", ")
}

make_random_split <- function(labels, train_fraction, seed, max_tries = 200) {
  if (!(is.numeric(train_fraction) &&
        length(train_fraction) == 1 &&
        is.finite(train_fraction) &&
        train_fraction > 0 &&
        train_fraction < 1)) {
    stop("`train_fraction` must be a numeric scalar between 0 and 1.")
  }

  n <- length(labels)
  if (n < 2) {
    stop("Need at least 2 samples for a random split.")
  }

  n_train <- floor(n * train_fraction)
  n_train <- max(1, min(n_train, n - 1))

  set.seed(seed)
  for (attempt in seq_len(max_tries)) {
    idx <- sample.int(n, size = n, replace = FALSE)
    train_idx <- sort(idx[seq_len(n_train)])
    test_idx <- sort(idx[(n_train + 1):n])

    train_tab <- table(labels[train_idx])
    test_tab <- table(labels[test_idx])

    if (length(train_tab) == 2 && all(train_tab > 0) &&
        length(test_tab) == 2 && all(test_tab > 0)) {
      return(list(train = train_idx, test = test_idx, attempts_used = attempt))
    }
  }

  stop("Could not find random split with both classes in train and test within max_tries.")
}

fit_and_predict_logistic <- function(X_train, y_train_fit, X_score, selected_genes) {
  if (length(selected_genes) == 0) {
    stop("No selected genes provided.")
  }

  X_train_sel <- X_train[, selected_genes, drop = FALSE]
  X_score_sel <- X_score[, selected_genes, drop = FALSE]

  scaler <- fit_scaler(X_train_sel)
  X_train_scaled <- apply_scaler(X_train_sel, scaler$means, scaler$sds)
  X_score_scaled <- apply_scaler(X_score_sel, scaler$means, scaler$sds)

  train_df <- as.data.frame(X_train_scaled)
  train_df$Outcome <- y_train_fit

  model <- suppressWarnings(
    glm(Outcome ~ ., data = train_df, family = binomial())
  )

  score_probs <- suppressWarnings(
    as.numeric(
      predict(model, newdata = as.data.frame(X_score_scaled), type = "response")
    )
  )

  list(
    probs = score_probs,
    model = model,
    scaler = scaler
  )
}

format_pn_text <- function(model_dims) {
  paste0(
    "p_total=", model_dims$p_total,
    ", p_model=", model_dims$p_model,
    " | n_total=", model_dims$n_total,
    ", n_train=", model_dims$n_train,
    ", n_test=", model_dims$n_test
  )
}

plot_truth_table_panel <- function(confusion_matrix, threshold_value = NA_real_) {
  mat <- as.matrix(confusion_matrix)
  if (!all(dim(mat) == c(2, 2))) {
    stop("Truth-table panel expects a 2x2 confusion matrix.")
  }

  max_count <- max(mat)
  color_scale <- colorRampPalette(c("#eef4fb", "#1f5a99"))(100)
  scale_index <- function(value) {
    if (max_count == 0) {
      return(1)
    }
    min(100, max(1, floor((value / max_count) * 99) + 1))
  }

  truth_title <- if (is.numeric(threshold_value) &&
                     length(threshold_value) == 1 &&
                     is.finite(threshold_value)) {
    paste0("Truth Table (threshold=", format_metric(threshold_value, digits = 2), ")")
  } else {
    "Truth Table"
  }

  plot(c(0, 2), c(0, 2), type = "n", xaxt = "n", yaxt = "n",
       xlab = "Actual", ylab = "Predicted", main = truth_title)

  cell_labels <- matrix(c("TN", "FN", "FP", "TP"), nrow = 2, byrow = TRUE)
  for (i in seq_len(2)) {
    for (j in seq_len(2)) {
      x_left <- j - 1
      x_right <- j
      y_bottom <- 2 - i
      y_top <- 3 - i
      value <- mat[i, j]

      rect(
        xleft = x_left, ybottom = y_bottom, xright = x_right, ytop = y_top,
        col = color_scale[scale_index(value)], border = "white", lwd = 2
      )

      text_col <- if (max_count > 0 && (value / max_count) > 0.6) "white" else "black"
      text(x = x_left + 0.5, y = y_bottom + 0.64, labels = cell_labels[i, j], cex = 0.9, font = 2, col = text_col)
      text(x = x_left + 0.5, y = y_bottom + 0.38, labels = value, cex = 1.15, font = 2, col = text_col)
    }
  }

  axis(1, at = c(0.5, 1.5), labels = colnames(mat), cex.axis = 0.85)
  axis(2, at = c(1.5, 0.5), labels = rownames(mat), las = 1, cex.axis = 0.85)
  box()
}

run_cv_for_scenario <- function(
  X_train,
  y_train_true,
  labels_train,
  k_folds,
  top_k_genes,
  seed,
  corrupt_training_labels = FALSE,
  selected_genes_global = NULL
) {
  fold_id <- make_stratified_folds(labels_train, k_folds, seed + 1)
  cv_auc <- rep(NA_real_, k_folds)
  cv_roc_curves <- vector(mode = "list", length = k_folds)

  for (fold in seq_len(k_folds)) {
    val_pos <- which(fold_id == fold)
    tr_pos <- which(fold_id != fold)

    X_fold_train <- X_train[tr_pos, , drop = FALSE]
    y_fold_train_true <- y_train_true[tr_pos]
    X_fold_val <- X_train[val_pos, , drop = FALSE]
    y_fold_val_true <- y_train_true[val_pos]

    selected_genes_fold <- if (is.null(selected_genes_global)) {
      select_top_genes(X_fold_train, y_fold_train_true, top_k_genes)
    } else {
      selected_genes_global
    }

    y_fold_fit <- if (isTRUE(corrupt_training_labels)) {
      1L - y_fold_train_true
    } else {
      y_fold_train_true
    }

    fold_fit <- fit_and_predict_logistic(
      X_train = X_fold_train,
      y_train_fit = y_fold_fit,
      X_score = X_fold_val,
      selected_genes = selected_genes_fold
    )
    fold_roc <- compute_roc_auc(y_fold_val_true, fold_fit$probs)

    cv_auc[fold] <- fold_roc$auc
    cv_roc_curves[[fold]] <- fold_roc
  }

  list(
    auc_values = cv_auc,
    roc_curves = cv_roc_curves,
    auc_mean = if (all(is.na(cv_auc))) NA_real_ else mean(cv_auc, na.rm = TRUE),
    auc_sd = if (sum(!is.na(cv_auc)) > 1) sd(cv_auc, na.rm = TRUE) else NA_real_
  )
}

plot_single_scenario_roc_on_current_device <- function(
  fpr,
  tpr,
  auc_value,
  cv_roc_curves,
  confusion_matrix,
  threshold_value = NA_real_,
  scenario_title,
  subtitle = NULL,
  pn_text = NULL
) {
  cv_summary <- NULL
  if (!is.null(cv_roc_curves) && length(cv_roc_curves) > 0) {
    cv_summary <- summarize_cv_roc_curves(cv_roc_curves)
  }

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  on.exit(layout(matrix(1, 1, 1)), add = TRUE)

  layout(matrix(c(1, 2), nrow = 1), widths = c(2.15, 1.2))
  par(mar = c(5.1, 4.6, 8.6, 2.1))

  main_text <- paste0(
    scenario_title,
    " (CV AUC = ", format_metric(if (is.null(cv_summary)) NA_real_ else cv_summary$mean_auc),
    "; Test AUC = ", format_metric(auc_value), ")"
  )

  plot(
    c(0, 1), c(0, 1), type = "n",
    xlim = c(0, 1), ylim = c(0, 1),
    xaxs = "i", yaxs = "i",
    xlab = "False Positive Rate",
    ylab = "True Positive Rate",
    main = ""
  )
  title(main = main_text, line = 4.4, cex.main = 1)
  abline(a = 0, b = 1, lty = 2, col = "gray50")

  if (!is.null(cv_summary)) {
    polygon(
      x = c(cv_summary$fpr_grid, rev(cv_summary$fpr_grid)),
      y = c(cv_summary$upper_tpr, rev(cv_summary$lower_tpr)),
      border = NA,
      col = rgb(0.45, 0.45, 0.45, alpha = 0.25)
    )
    lines(cv_summary$fpr_grid, cv_summary$mean_tpr, lwd = 2.5, col = "gray35")
  }

  lines(fpr, tpr, lwd = 3, col = "#1f5a99")
  grid()

  legend_labels <- c("Reference (random)")
  legend_colors <- c("gray50")
  legend_lty <- c(2)
  legend_lwd <- c(1)

  if (!is.null(cv_summary)) {
    legend_labels <- c(
      legend_labels,
      paste0("CV mean (", cv_summary$n_curves, " folds, AUC=", format_metric(cv_summary$mean_auc), ")")
    )
    legend_colors <- c(legend_colors, "gray35")
    legend_lty <- c(legend_lty, 1)
    legend_lwd <- c(legend_lwd, 2.5)
  }

  legend_labels <- c(legend_labels, paste0("Test ROC (AUC=", format_metric(auc_value), ")"))
  legend_colors <- c(legend_colors, "#1f5a99")
  legend_lty <- c(legend_lty, 1)
  legend_lwd <- c(legend_lwd, 3)

  legend(
    "bottomright",
    legend = legend_labels,
    col = legend_colors,
    lty = legend_lty,
    lwd = legend_lwd,
    bty = "n",
    cex = 0.85
  )

  if (!is.null(cv_summary)) {
    legend(
      "bottomright",
      inset = c(0, 0.15),
      legend = "CV envelope (mean +/- 1 SD)",
      fill = rgb(0.45, 0.45, 0.45, alpha = 0.25),
      border = NA,
      bty = "n",
      cex = 0.8
    )
  }

  if (!is.null(pn_text) && nzchar(pn_text)) {
    mtext(pn_text, side = 3, line = 2.5, cex = 0.82)
  }
  if (!is.null(subtitle) && nzchar(subtitle)) {
    mtext(subtitle, side = 3, line = 1.2, cex = 0.85)
  }

  par(mar = c(5.1, 2.4, 4.0, 2.1))
  plot_truth_table_panel(confusion_matrix = confusion_matrix, threshold_value = threshold_value)
}

save_single_scenario_roc <- function(
  fpr,
  tpr,
  auc_value,
  cv_roc_curves,
  confusion_matrix,
  threshold_value = NA_real_,
  scenario_title,
  file_path,
  subtitle = NULL,
  pn_text = NULL,
  png_width_px = export_png_width_px,
  png_height_px = export_png_height_px,
  show_live = FALSE,
  open_new_device_for_live_plots = FALSE
) {
  png(filename = file_path, width = png_width_px, height = png_height_px)
  plot_single_scenario_roc_on_current_device(
    fpr = fpr,
    tpr = tpr,
    auc_value = auc_value,
    cv_roc_curves = cv_roc_curves,
    confusion_matrix = confusion_matrix,
    threshold_value = threshold_value,
    scenario_title = scenario_title,
    subtitle = subtitle,
    pn_text = pn_text
  )
  dev.off()

  if (isTRUE(show_live) && interactive()) {
    if (isTRUE(open_new_device_for_live_plots)) {
      dev.new(width = 8, height = 6)
    }
    plot_single_scenario_roc_on_current_device(
      fpr = fpr,
      tpr = tpr,
      auc_value = auc_value,
      cv_roc_curves = cv_roc_curves,
      confusion_matrix = confusion_matrix,
      threshold_value = threshold_value,
      scenario_title = scenario_title,
      subtitle = subtitle,
      pn_text = pn_text
    )
  }
}

run_scenario_core <- function(
  df,
  seed,
  train_fraction,
  positive_class,
  top_k_genes,
  k_folds = 5,
  threshold_value = 0.5,
  split_mode = c("stratified", "random"),
  feature_selection_mode = c("train_only", "full_data_leakage"),
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
) {
  split_mode <- match.arg(split_mode)
  feature_selection_mode <- match.arg(feature_selection_mode)

  data_parts <- prepare_model_data(df, positive_class)

  split_idx <- if (split_mode == "stratified") {
    make_stratified_split(data_parts$labels, train_fraction, split_seed = seed)
  } else {
    make_random_split(data_parts$labels, train_fraction = train_fraction, seed = seed)
  }

  train_idx <- split_idx$train
  test_idx <- split_idx$test

  X_train <- data_parts$X_all[train_idx, , drop = FALSE]
  X_test <- data_parts$X_all[test_idx, , drop = FALSE]
  y_train_true <- data_parts$y_all_binary[train_idx]
  y_test_true <- data_parts$y_all_binary[test_idx]
  labels_train <- data_parts$labels[train_idx]
  labels_test <- data_parts$labels[test_idx]

  selected_genes_global <- if (feature_selection_mode == "full_data_leakage") {
    select_top_genes(data_parts$X_all, data_parts$y_all_binary, top_k_genes)
  } else {
    NULL
  }

  selected_genes <- if (is.null(selected_genes_global)) {
    select_top_genes(X_train, y_train_true, top_k_genes)
  } else {
    selected_genes_global
  }

  y_train_fit <- y_train_true
  if (isTRUE(corrupt_training_labels)) {
    y_train_fit <- 1L - y_train_true
  }

  cv_results <- run_cv_for_scenario(
    X_train = X_train,
    y_train_true = y_train_true,
    labels_train = labels_train,
    k_folds = k_folds,
    top_k_genes = top_k_genes,
    seed = seed,
    corrupt_training_labels = corrupt_training_labels,
    selected_genes_global = selected_genes_global
  )

  fit_obj <- fit_and_predict_logistic(
    X_train = X_train,
    y_train_fit = y_train_fit,
    X_score = X_test,
    selected_genes = selected_genes
  )
  test_probs <- fit_obj$probs
  test_roc <- compute_roc_auc(y_test_true, test_probs)
  test_conf <- compute_confusion_stats(y_test_true, test_probs, threshold_value)

  pred_labels <- ifelse(test_conf$pred == 1L, data_parts$positive_class, data_parts$negative_class)
  actual_labels <- ifelse(y_test_true == 1L, data_parts$positive_class, data_parts$negative_class)
  confusion_matrix <- table(
    Predicted = factor(pred_labels, levels = c(data_parts$negative_class, data_parts$positive_class)),
    Actual = factor(actual_labels, levels = c(data_parts$negative_class, data_parts$positive_class))
  )

  train_auc <- NA_real_
  if (isTRUE(compute_train_auc)) {
    X_train_sel <- X_train[, selected_genes, drop = FALSE]
    X_train_scaled <- apply_scaler(X_train_sel, fit_obj$scaler$means, fit_obj$scaler$sds)
    train_probs <- suppressWarnings(
      as.numeric(
        predict(fit_obj$model, newdata = as.data.frame(X_train_scaled), type = "response")
      )
    )
    train_roc <- compute_roc_auc(y_train_true, train_probs)
    train_auc <- train_roc$auc
  }

  list(
    auc_cv_mean = cv_results$auc_mean,
    auc_cv_sd = cv_results$auc_sd,
    cv = cv_results,
    threshold_value = threshold_value,
    auc_test = test_roc$auc,
    roc_test = test_roc,
    confusion_matrix = confusion_matrix,
    auc_train = train_auc,
    selected_genes = selected_genes,
    model_dims = list(
      p_total = ncol(data_parts$X_all),
      p_model = length(selected_genes),
      n_total = nrow(data_parts$X_all),
      n_train = length(train_idx),
      n_test = length(test_idx)
    ),
    split_summary = list(
      train_counts = table(labels_train),
      test_counts = table(labels_test)
    )
  )
}


# ============================================================
# STEP 4) Generate datasets
# ============================================================

base_df <- generate_tumor_dataset(
  seed = dataset_seed_main,
  n_samples = dataset_n_samples,
  n_genes = dataset_n_genes,
  n_informative = dataset_n_informative,
  class_proportion = dataset_class_proportion_main,
  noise_sd = dataset_noise_sd
)

imbalance_df <- generate_tumor_dataset(
  seed = dataset_seed_imbalance,
  n_samples = dataset_n_samples,
  n_genes = dataset_n_genes,
  n_informative = dataset_n_informative,
  class_proportion = dataset_class_proportion_imbalance,
  noise_sd = dataset_noise_sd
)

perfect_df <- generate_tumor_dataset(
  seed = dataset_seed_perfect,
  n_samples = perfect_n_samples,
  n_genes = perfect_n_genes,
  n_informative = perfect_n_informative,
  class_proportion = dataset_class_proportion_perfect,
  noise_sd = dataset_noise_sd
)


# ============================================================
# STEP 5) Run scenario analyses and save ROC plots
# ============================================================

scenario_rows <- list()

# Scenario 0: High n, high p "perfect" dataset
scenario_0 <- run_scenario_core(
  df = perfect_df,
  seed = scenario_seeds$perfect,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_perfect,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "train_only",
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
)

scenario_0_file <- paste0(
  "roc_scenario_0_perfect_high_n_high_p_seed_",
  scenario_seeds$perfect,
  ".png"
)
save_single_scenario_roc(
  fpr = scenario_0$roc_test$fpr,
  tpr = scenario_0$roc_test$tpr,
  auc_value = scenario_0$auc_test,
  cv_roc_curves = scenario_0$cv$roc_curves,
  confusion_matrix = scenario_0$confusion_matrix,
  threshold_value = scenario_0$threshold_value,
  scenario_title = "Scenario 0: High n, High p, Perfect Dataset",
  file_path = scenario_0_file,
  pn_text = format_pn_text(scenario_0$model_dims),
  subtitle = paste0(
    "n_samples=", perfect_n_samples,
    ", n_genes=", perfect_n_genes,
    ", n_informative=", perfect_n_informative,
    ", noise_sd=", dataset_noise_sd,
    " | CV mean AUC=", format_metric(scenario_0$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[1]] <- data.frame(
  scenario = "0. Perfect (high n, high p)",
  auc_cv_mean = scenario_0$auc_cv_mean,
  auc_test = scenario_0$auc_test,
  auc_train = NA_real_,
  notes = paste0(
    "n=", perfect_n_samples,
    "; p=", perfect_n_genes,
    "; informative=", perfect_n_informative,
    "; noise_sd=", dataset_noise_sd,
    "; cv_auc=", format_metric(scenario_0$auc_cv_mean)
  ),
  output_file = scenario_0_file,
  stringsAsFactors = FALSE
)

# Scenario 1: Overfitting
scenario_1 <- run_scenario_core(
  df = base_df,
  seed = scenario_seeds$overfitting,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_overfitting,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "train_only",
  corrupt_training_labels = FALSE,
  compute_train_auc = TRUE
)

scenario_1_file <- paste0(
  "roc_scenario_1_overfitting_seed_",
  scenario_seeds$overfitting,
  ".png"
)
scenario_1_gap <- scenario_1$auc_train - scenario_1$auc_test
save_single_scenario_roc(
  fpr = scenario_1$roc_test$fpr,
  tpr = scenario_1$roc_test$tpr,
  auc_value = scenario_1$auc_test,
  cv_roc_curves = scenario_1$cv$roc_curves,
  confusion_matrix = scenario_1$confusion_matrix,
  threshold_value = scenario_1$threshold_value,
  scenario_title = "Scenario 1: Overfitting",
  file_path = scenario_1_file,
  pn_text = format_pn_text(scenario_1$model_dims),
  subtitle = paste0(
    "top_k=", top_k_overfitting,
    " | train-test AUC gap=", format_metric(scenario_1_gap),
    " | CV mean AUC=", format_metric(scenario_1$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[2]] <- data.frame(
  scenario = "1. Overfitting",
  auc_cv_mean = scenario_1$auc_cv_mean,
  auc_test = scenario_1$auc_test,
  auc_train = scenario_1$auc_train,
  notes = paste0(
    "top_k=", top_k_overfitting,
    "; cv_auc=", format_metric(scenario_1$auc_cv_mean),
    "; train_vs_test_gap=", format_metric(scenario_1_gap)
  ),
  output_file = scenario_1_file,
  stringsAsFactors = FALSE
)


# Scenario 2: Underfitting
scenario_2 <- run_scenario_core(
  df = base_df,
  seed = scenario_seeds$underfitting,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_underfitting,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "train_only",
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
)

scenario_2_file <- paste0(
  "roc_scenario_2_underfitting_seed_",
  scenario_seeds$underfitting,
  ".png"
)
save_single_scenario_roc(
  fpr = scenario_2$roc_test$fpr,
  tpr = scenario_2$roc_test$tpr,
  auc_value = scenario_2$auc_test,
  cv_roc_curves = scenario_2$cv$roc_curves,
  confusion_matrix = scenario_2$confusion_matrix,
  threshold_value = scenario_2$threshold_value,
  scenario_title = "Scenario 2: Underfitting",
  file_path = scenario_2_file,
  pn_text = format_pn_text(scenario_2$model_dims),
  subtitle = paste0(
    "top_k=", top_k_underfitting,
    " (very low model capacity)",
    " | CV mean AUC=", format_metric(scenario_2$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[3]] <- data.frame(
  scenario = "2. Underfitting",
  auc_cv_mean = scenario_2$auc_cv_mean,
  auc_test = scenario_2$auc_test,
  auc_train = NA_real_,
  notes = paste0(
    "top_k=", top_k_underfitting,
    "; cv_auc=", format_metric(scenario_2$auc_cv_mean),
    "; intentionally low capacity"
  ),
  output_file = scenario_2_file,
  stringsAsFactors = FALSE
)


# Scenario 3: Wrong-label training (messed up model)
scenario_3 <- run_scenario_core(
  df = base_df,
  seed = scenario_seeds$wrong_labels,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_leakage,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "train_only",
  corrupt_training_labels = TRUE,
  compute_train_auc = FALSE
)

if (!is.na(scenario_3$auc_test) && scenario_3$auc_test >= wrong_label_auc_warning_cutoff) {
  warning(
    paste0(
      "Scenario 3 warning: test AUC is ",
      format_metric(scenario_3$auc_test),
      " (not below ",
      wrong_label_auc_warning_cutoff,
      "). The curve may not clearly fall below random."
    ),
    call. = FALSE
  )
}

scenario_3_file <- paste0(
  "roc_scenario_3_wrong_labels_seed_",
  scenario_seeds$wrong_labels,
  ".png"
)
save_single_scenario_roc(
  fpr = scenario_3$roc_test$fpr,
  tpr = scenario_3$roc_test$tpr,
  auc_value = scenario_3$auc_test,
  cv_roc_curves = scenario_3$cv$roc_curves,
  confusion_matrix = scenario_3$confusion_matrix,
  threshold_value = scenario_3$threshold_value,
  scenario_title = "Scenario 3: Wrong Labels in Training",
  file_path = scenario_3_file,
  pn_text = format_pn_text(scenario_3$model_dims),
  subtitle = paste0(
    "Training labels intentionally flipped before fitting",
    " | CV mean AUC=", format_metric(scenario_3$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[4]] <- data.frame(
  scenario = "3. Wrong labels in training",
  auc_cv_mean = scenario_3$auc_cv_mean,
  auc_test = scenario_3$auc_test,
  auc_train = NA_real_,
  notes = if (is.na(scenario_3$auc_test)) {
    "AUC unavailable"
  } else if (scenario_3$auc_test < wrong_label_auc_warning_cutoff) {
    paste0("cv_auc=", format_metric(scenario_3$auc_cv_mean), "; AUC below 0.5 as expected")
  } else {
    paste0("cv_auc=", format_metric(scenario_3$auc_cv_mean), "; Warning: AUC not below 0.5")
  },
  output_file = scenario_3_file,
  stringsAsFactors = FALSE
)


# Scenario 4: Leakage from feature selection before split
scenario_4_leaked <- run_scenario_core(
  df = base_df,
  seed = scenario_seeds$leakage,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_leakage,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "full_data_leakage",
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
)

scenario_4_safe <- run_scenario_core(
  df = base_df,
  seed = scenario_seeds$leakage,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_leakage,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "stratified",
  feature_selection_mode = "train_only",
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
)

scenario_4_delta <- scenario_4_leaked$auc_test - scenario_4_safe$auc_test
scenario_4_file <- paste0(
  "roc_scenario_4_feature_leakage_seed_",
  scenario_seeds$leakage,
  ".png"
)
save_single_scenario_roc(
  fpr = scenario_4_leaked$roc_test$fpr,
  tpr = scenario_4_leaked$roc_test$tpr,
  auc_value = scenario_4_leaked$auc_test,
  cv_roc_curves = scenario_4_leaked$cv$roc_curves,
  confusion_matrix = scenario_4_leaked$confusion_matrix,
  threshold_value = scenario_4_leaked$threshold_value,
  scenario_title = "Scenario 4: Feature-Selection Leakage",
  file_path = scenario_4_file,
  pn_text = format_pn_text(scenario_4_leaked$model_dims),
  subtitle = paste0(
    "leaked AUC=", format_metric(scenario_4_leaked$auc_test),
    " | safe AUC=", format_metric(scenario_4_safe$auc_test),
    " | delta=", format_metric(scenario_4_delta),
    " | CV mean AUC=", format_metric(scenario_4_leaked$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[5]] <- data.frame(
  scenario = "4. Leakage before split",
  auc_cv_mean = scenario_4_leaked$auc_cv_mean,
  auc_test = scenario_4_leaked$auc_test,
  auc_train = NA_real_,
  notes = paste0(
    "leaked_auc=", format_metric(scenario_4_leaked$auc_test),
    "; safe_auc=", format_metric(scenario_4_safe$auc_test),
    "; delta=", format_metric(scenario_4_delta),
    "; cv_auc=", format_metric(scenario_4_leaked$auc_cv_mean)
  ),
  output_file = scenario_4_file,
  stringsAsFactors = FALSE
)


# Scenario 5: Ignore 90:10 imbalance (non-stratified split)
scenario_5 <- run_scenario_core(
  df = imbalance_df,
  seed = scenario_seeds$imbalance,
  train_fraction = train_fraction,
  positive_class = positive_class,
  top_k_genes = top_k_imbalance,
  k_folds = k_folds,
  threshold_value = threshold_for_truth_table,
  split_mode = "random",
  feature_selection_mode = "train_only",
  corrupt_training_labels = FALSE,
  compute_train_auc = FALSE
)

scenario_5_file <- paste0(
  "roc_scenario_5_ignore_imbalance_90_10_seed_",
  scenario_seeds$imbalance,
  ".png"
)
save_single_scenario_roc(
  fpr = scenario_5$roc_test$fpr,
  tpr = scenario_5$roc_test$tpr,
  auc_value = scenario_5$auc_test,
  cv_roc_curves = scenario_5$cv$roc_curves,
  confusion_matrix = scenario_5$confusion_matrix,
  threshold_value = scenario_5$threshold_value,
  scenario_title = "Scenario 5: Ignore Class Imbalance (90:10)",
  file_path = scenario_5_file,
  pn_text = format_pn_text(scenario_5$model_dims),
  subtitle = paste0(
    "train counts {", table_to_string(scenario_5$split_summary$train_counts), "}",
    " | test counts {", table_to_string(scenario_5$split_summary$test_counts), "}",
    " | CV mean AUC=", format_metric(scenario_5$auc_cv_mean)
  ),
  show_live = show_plots_live,
  open_new_device_for_live_plots = open_new_device_for_live_plots
)

scenario_rows[[6]] <- data.frame(
  scenario = "5. Ignore 90:10 imbalance",
  auc_cv_mean = scenario_5$auc_cv_mean,
  auc_test = scenario_5$auc_test,
  auc_train = NA_real_,
  notes = paste0(
    "train_counts={", table_to_string(scenario_5$split_summary$train_counts), "}",
    "; test_counts={", table_to_string(scenario_5$split_summary$test_counts), "}",
    "; cv_auc=", format_metric(scenario_5$auc_cv_mean)
  ),
  output_file = scenario_5_file,
  stringsAsFactors = FALSE
)


# ============================================================
# STEP 6) Print summary and file list
# ============================================================

scenario_summary <- do.call(rbind, scenario_rows)

cat("\n=== ROC Scenario Summary ===\n")
print(scenario_summary, row.names = FALSE)

cat("\nGenerated ROC files:\n")
for (f in scenario_summary$output_file) {
  cat(" - ", f, "\n", sep = "")
}
cat("Done.\n")
