# ============================================================
# BIOS226 - Week 5
# Helper functions for leakage-safe tumor subtype modeling
# ============================================================

is_positive_integer <- function(x) {
  is.numeric(x) && length(x) == 1 && is.finite(x) && x > 0 && x == as.integer(x)
}

format_metric <- function(x, digits = 3) {
  if (is.na(x)) {
    return("NA")
  }
  format(round(x, digits), nsmall = digits, trim = TRUE)
}

validate_model_config <- function(train_fraction, k_folds, top_k_genes, positive_class, threshold) {
  if (!(is.numeric(train_fraction) &&
        length(train_fraction) == 1 &&
        is.finite(train_fraction) &&
        train_fraction > 0 &&
        train_fraction < 1)) {
    stop("`train_fraction` must be a numeric scalar between 0 and 1.")
  }
  if (!is_positive_integer(k_folds) || k_folds < 2) {
    stop("`k_folds` must be an integer >= 2.")
  }
  if (!is_positive_integer(top_k_genes)) {
    stop("`top_k_genes` must be a positive integer.")
  }
  if (!(is.character(positive_class) && length(positive_class) == 1)) {
    stop("`positive_class` must be a single character label.")
  }
  if (!(is.numeric(threshold) &&
        length(threshold) == 1 &&
        is.finite(threshold) &&
        threshold > 0 &&
        threshold < 1)) {
    stop("`threshold` must be a numeric scalar between 0 and 1.")
  }
}

prepare_model_data <- function(df, positive_class) {
  required_cols <- c("Patient_ID", "Subtype")
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }

  gene_cols <- grep("^Gene_", names(df), value = TRUE)
  if (length(gene_cols) == 0) {
    stop("No gene columns found. Expected columns named Gene_1, Gene_2, ...")
  }

  is_numeric_gene <- vapply(df[, gene_cols, drop = FALSE], is.numeric, logical(1))
  if (!all(is_numeric_gene)) {
    stop("All Gene_* columns must be numeric.")
  }

  labels <- as.character(df$Subtype)
  unique_labels <- sort(unique(labels))
  if (length(unique_labels) != 2) {
    stop("Subtype must contain exactly two classes for binary classification.")
  }
  if (!(positive_class %in% unique_labels)) {
    stop(paste0("positive_class '", positive_class, "' is not present in Subtype labels."))
  }

  negative_class <- setdiff(unique_labels, positive_class)
  X_all <- as.matrix(df[, gene_cols, drop = FALSE])
  y_all_binary <- ifelse(labels == positive_class, 1L, 0L)

  list(
    X_all = X_all,
    labels = labels,
    y_all_binary = y_all_binary,
    gene_cols = gene_cols,
    positive_class = positive_class,
    negative_class = negative_class
  )
}

make_stratified_split <- function(labels, train_frac, split_seed) {
  set.seed(split_seed)
  train_idx <- integer(0)
  test_idx <- integer(0)

  for (cls in unique(labels)) {
    cls_idx <- which(labels == cls)
    cls_idx <- sample(cls_idx, length(cls_idx), replace = FALSE)

    n_train_cls <- floor(length(cls_idx) * train_frac)
    n_train_cls <- max(1, n_train_cls)
    n_train_cls <- min(n_train_cls, length(cls_idx) - 1)

    train_idx <- c(train_idx, cls_idx[seq_len(n_train_cls)])
    test_idx <- c(test_idx, cls_idx[(n_train_cls + 1):length(cls_idx)])
  }

  list(train = sort(train_idx), test = sort(test_idx))
}

make_stratified_folds <- function(labels, k, fold_seed) {
  if (any(table(labels) < k)) {
    stop("Each class must have at least k_folds samples for stratified CV.")
  }

  set.seed(fold_seed)
  fold_id <- integer(length(labels))

  for (cls in unique(labels)) {
    cls_idx <- which(labels == cls)
    cls_idx <- sample(cls_idx, length(cls_idx), replace = FALSE)
    fold_assign <- rep(seq_len(k), length.out = length(cls_idx))
    fold_id[cls_idx] <- fold_assign
  }

  fold_id
}

select_top_genes <- function(X_train, y_train_binary, top_k) {
  if (length(unique(y_train_binary)) < 2) {
    stop("Training data has only one class; cannot select discriminative genes.")
  }

  pos_idx <- y_train_binary == 1L
  neg_idx <- y_train_binary == 0L

  pos_means <- colMeans(X_train[pos_idx, , drop = FALSE])
  neg_means <- colMeans(X_train[neg_idx, , drop = FALSE])
  scores <- abs(pos_means - neg_means)

  k_use <- min(top_k, ncol(X_train))
  names(sort(scores, decreasing = TRUE))[seq_len(k_use)]
}

fit_scaler <- function(X_train_selected) {
  means <- colMeans(X_train_selected)
  sds <- apply(X_train_selected, 2, sd)
  sds[is.na(sds) | sds == 0] <- 1
  list(means = means, sds = sds)
}

apply_scaler <- function(X_data, means, sds) {
  centered <- sweep(X_data, 2, means, FUN = "-")
  sweep(centered, 2, sds, FUN = "/")
}

compute_roc_auc <- function(y_true_binary, probs) {
  y_true_binary <- as.integer(y_true_binary)

  if (length(unique(y_true_binary)) < 2) {
    return(list(fpr = c(0, 1), tpr = c(0, 1), auc = NA_real_))
  }

  thresholds <- c(Inf, sort(unique(probs), decreasing = TRUE), -Inf)
  positives <- sum(y_true_binary == 1)
  negatives <- sum(y_true_binary == 0)

  tpr <- numeric(length(thresholds))
  fpr <- numeric(length(thresholds))

  for (i in seq_along(thresholds)) {
    pred <- ifelse(probs >= thresholds[i], 1L, 0L)
    tp <- sum(pred == 1L & y_true_binary == 1L)
    fp <- sum(pred == 1L & y_true_binary == 0L)
    tpr[i] <- tp / positives
    fpr[i] <- fp / negatives
  }

  ord <- order(fpr, tpr)
  fpr_sorted <- fpr[ord]
  tpr_sorted <- tpr[ord]

  fpr_unique <- sort(unique(fpr_sorted))
  tpr_unique <- sapply(fpr_unique, function(x) max(tpr_sorted[fpr_sorted == x], na.rm = TRUE))

  if (length(fpr_unique) < 2) {
    auc <- NA_real_
  } else {
    auc <- sum(diff(fpr_unique) * (head(tpr_unique, -1) + tail(tpr_unique, -1)) / 2)
  }

  list(fpr = fpr_unique, tpr = tpr_unique, auc = auc)
}

compute_confusion_stats <- function(y_true_binary, probs, threshold_value) {
  pred <- ifelse(probs >= threshold_value, 1L, 0L)

  tp <- sum(pred == 1L & y_true_binary == 1L)
  fp <- sum(pred == 1L & y_true_binary == 0L)
  tn <- sum(pred == 0L & y_true_binary == 0L)
  fn <- sum(pred == 0L & y_true_binary == 1L)

  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)

  list(pred = pred, tp = tp, fp = fp, tn = tn, fn = fn, precision = precision)
}

run_cv_on_training <- function(X_train, y_train_binary, labels_train, k_folds, top_k_genes, threshold, seed) {
  fold_id <- make_stratified_folds(labels_train, k_folds, seed + 1)
  cv_auc <- rep(NA_real_, k_folds)
  cv_precision <- rep(NA_real_, k_folds)
  cv_roc_curves <- vector(mode = "list", length = k_folds)

  for (fold in seq_len(k_folds)) {
    val_pos <- which(fold_id == fold)
    tr_pos <- which(fold_id != fold)

    X_fold_train <- X_train[tr_pos, , drop = FALSE]
    y_fold_train <- y_train_binary[tr_pos]
    X_fold_val <- X_train[val_pos, , drop = FALSE]
    y_fold_val <- y_train_binary[val_pos]

    selected_genes_fold <- select_top_genes(X_fold_train, y_fold_train, top_k_genes)

    X_fold_train_sel <- X_fold_train[, selected_genes_fold, drop = FALSE]
    X_fold_val_sel <- X_fold_val[, selected_genes_fold, drop = FALSE]

    scaler_fold <- fit_scaler(X_fold_train_sel)
    X_fold_train_scaled <- apply_scaler(X_fold_train_sel, scaler_fold$means, scaler_fold$sds)
    X_fold_val_scaled <- apply_scaler(X_fold_val_sel, scaler_fold$means, scaler_fold$sds)

    fold_train_df <- as.data.frame(X_fold_train_scaled)
    fold_train_df$Outcome <- y_fold_train

    fold_model <- suppressWarnings(
      glm(Outcome ~ ., data = fold_train_df, family = binomial())
    )
    fold_probs <- as.numeric(
      predict(fold_model, newdata = as.data.frame(X_fold_val_scaled), type = "response")
    )

    fold_roc <- compute_roc_auc(y_fold_val, fold_probs)
    fold_conf <- compute_confusion_stats(y_fold_val, fold_probs, threshold)

    cv_auc[fold] <- fold_roc$auc
    cv_precision[fold] <- fold_conf$precision
    cv_roc_curves[[fold]] <- fold_roc
  }

  list(
    auc_values = cv_auc,
    precision_values = cv_precision,
    roc_curves = cv_roc_curves,
    auc_mean = if (all(is.na(cv_auc))) NA_real_ else mean(cv_auc, na.rm = TRUE),
    auc_sd = if (sum(!is.na(cv_auc)) > 1) sd(cv_auc, na.rm = TRUE) else NA_real_,
    precision_mean = if (all(is.na(cv_precision))) NA_real_ else mean(cv_precision, na.rm = TRUE),
    precision_sd = if (sum(!is.na(cv_precision)) > 1) sd(cv_precision, na.rm = TRUE) else NA_real_
  )
}

run_final_on_test_set <- function(
  X_train,
  X_test,
  y_train_binary,
  y_test_binary,
  top_k_genes,
  threshold,
  positive_class,
  negative_class
) {
  selected_genes_final <- select_top_genes(X_train, y_train_binary, top_k_genes)
  X_train_sel <- X_train[, selected_genes_final, drop = FALSE]
  X_test_sel <- X_test[, selected_genes_final, drop = FALSE]

  scaler_final <- fit_scaler(X_train_sel)
  X_train_scaled <- apply_scaler(X_train_sel, scaler_final$means, scaler_final$sds)
  X_test_scaled <- apply_scaler(X_test_sel, scaler_final$means, scaler_final$sds)

  final_train_df <- as.data.frame(X_train_scaled)
  final_train_df$Outcome <- y_train_binary

  final_model <- suppressWarnings(
    glm(Outcome ~ ., data = final_train_df, family = binomial())
  )
  test_probs <- as.numeric(
    predict(final_model, newdata = as.data.frame(X_test_scaled), type = "response")
  )

  test_conf <- compute_confusion_stats(y_test_binary, test_probs, threshold)
  test_roc <- compute_roc_auc(y_test_binary, test_probs)

  pred_labels <- ifelse(test_conf$pred == 1L, positive_class, negative_class)
  actual_labels <- ifelse(y_test_binary == 1L, positive_class, negative_class)

  confusion_matrix <- table(
    Predicted = factor(pred_labels, levels = c(negative_class, positive_class)),
    Actual = factor(actual_labels, levels = c(negative_class, positive_class))
  )

  mse <- mean((y_test_binary - test_probs)^2)
  sse <- sum((y_test_binary - test_probs)^2)
  sst <- sum((y_test_binary - mean(y_test_binary))^2)
  r_squared <- if (sst == 0) NA_real_ else 1 - (sse / sst)

  list(
    selected_genes = selected_genes_final,
    final_model = final_model,
    confusion_matrix = confusion_matrix,
    tp = test_conf$tp,
    fp = test_conf$fp,
    tn = test_conf$tn,
    fn = test_conf$fn,
    probs = test_probs,
    y_true_binary = y_test_binary,
    pred_binary = test_conf$pred,
    precision = test_conf$precision,
    auc = test_roc$auc,
    roc_fpr = test_roc$fpr,
    roc_tpr = test_roc$tpr,
    mse = mse,
    r_squared = r_squared
  )
}

run_tumor_workflow <- function(df, seed, train_fraction, k_folds, top_k_genes, positive_class, threshold) {
  validate_model_config(train_fraction, k_folds, top_k_genes, positive_class, threshold)
  data_parts <- prepare_model_data(df, positive_class)

  split_idx <- make_stratified_split(data_parts$labels, train_fraction, seed)
  train_idx <- split_idx$train
  test_idx <- split_idx$test

  X_train <- data_parts$X_all[train_idx, , drop = FALSE]
  X_test <- data_parts$X_all[test_idx, , drop = FALSE]
  y_train_binary <- data_parts$y_all_binary[train_idx]
  y_test_binary <- data_parts$y_all_binary[test_idx]
  labels_train <- data_parts$labels[train_idx]
  labels_test <- data_parts$labels[test_idx]

  cv_results <- run_cv_on_training(
    X_train = X_train,
    y_train_binary = y_train_binary,
    labels_train = labels_train,
    k_folds = k_folds,
    top_k_genes = top_k_genes,
    threshold = threshold,
    seed = seed
  )

  test_results <- run_final_on_test_set(
    X_train = X_train,
    X_test = X_test,
    y_train_binary = y_train_binary,
    y_test_binary = y_test_binary,
    top_k_genes = top_k_genes,
    threshold = threshold,
    positive_class = data_parts$positive_class,
    negative_class = data_parts$negative_class
  )

  list(
    config = list(
      seed = seed,
      train_fraction = train_fraction,
      k_folds = k_folds,
      top_k_genes = top_k_genes,
      positive_class = data_parts$positive_class,
      negative_class = data_parts$negative_class,
      threshold = threshold
    ),
    dataset_summary = list(
      n_samples = nrow(df),
      n_genes = length(data_parts$gene_cols),
      class_counts = table(df$Subtype)
    ),
    split_summary = list(
      train_n = length(train_idx),
      test_n = length(test_idx),
      train_class_counts = table(labels_train),
      test_class_counts = table(labels_test),
      train_idx = train_idx,
      test_idx = test_idx
    ),
    cv = cv_results,
    selected_genes = test_results$selected_genes,
    test = list(
      confusion_matrix = test_results$confusion_matrix,
      tp = test_results$tp,
      fp = test_results$fp,
      tn = test_results$tn,
      fn = test_results$fn,
      probs = test_results$probs,
      y_true_binary = test_results$y_true_binary,
      precision = test_results$precision,
      auc = test_results$auc,
      roc_fpr = test_results$roc_fpr,
      roc_tpr = test_results$roc_tpr,
      mse = test_results$mse,
      r_squared = test_results$r_squared
    )
  )
}

auc_from_curve <- function(fpr, tpr) {
  if (length(fpr) < 2 || length(tpr) < 2) {
    return(NA_real_)
  }
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
}

summarize_cv_roc_curves <- function(cv_roc_curves, fpr_grid = seq(0, 1, by = 0.01)) {
  valid_curves <- Filter(
    f = function(curve_i) {
      !is.null(curve_i) &&
        !is.null(curve_i$fpr) &&
        !is.null(curve_i$tpr) &&
        length(curve_i$fpr) >= 2 &&
        length(curve_i$tpr) >= 2
    },
    x = cv_roc_curves
  )

  if (length(valid_curves) == 0) {
    return(NULL)
  }

  tpr_matrix <- matrix(NA_real_, nrow = length(valid_curves), ncol = length(fpr_grid))

  for (i in seq_along(valid_curves)) {
    curve_i <- valid_curves[[i]]
    interp_tpr <- approx(
      x = curve_i$fpr,
      y = curve_i$tpr,
      xout = fpr_grid,
      method = "linear",
      rule = 2,
      ties = "ordered"
    )$y
    tpr_matrix[i, ] <- interp_tpr
  }

  mean_tpr <- colMeans(tpr_matrix, na.rm = TRUE)
  sd_tpr <- apply(tpr_matrix, 2, sd, na.rm = TRUE)
  sd_tpr[is.na(sd_tpr)] <- 0

  lower_tpr <- pmax(0, mean_tpr - sd_tpr)
  upper_tpr <- pmin(1, mean_tpr + sd_tpr)
  mean_auc <- auc_from_curve(fpr_grid, mean_tpr)

  list(
    fpr_grid = fpr_grid,
    mean_tpr = mean_tpr,
    sd_tpr = sd_tpr,
    lower_tpr = lower_tpr,
    upper_tpr = upper_tpr,
    mean_auc = mean_auc,
    n_curves = length(valid_curves)
  )
}

plot_roc_on_current_device <- function(fpr, tpr, auc_value, cv_roc_curves = NULL) {
  cv_summary <- NULL
  if (!is.null(cv_roc_curves) && length(cv_roc_curves) > 0) {
    cv_summary <- summarize_cv_roc_curves(cv_roc_curves)
  }

  plot(
    c(0, 1), c(0, 1), type = "n",
    xlim = c(0, 1),
    ylim = c(0, 1),
    xaxs = "i",
    yaxs = "i",
    xlab = "False Positive Rate",
    ylab = "True Positive Rate",
    main = paste0("ROC Curves: CV Mean +/- SD Envelope + Test (Test AUC = ", format_metric(auc_value), ")")
  )
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

  legend_labels <- c(legend_labels, paste0("Test (AUC=", format_metric(auc_value), ")"))
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
    cex = 0.8
  )

  if (!is.null(cv_summary)) {
    legend(
      "bottomright",
      inset = c(0, 0.16),
      legend = "CV envelope (mean +/- 1 SD)",
      fill = rgb(0.45, 0.45, 0.45, alpha = 0.25),
      border = NA,
      bty = "n",
      cex = 0.8
    )
  }
}

plot_truth_table_on_current_device <- function(confusion_matrix) {
  mat <- as.matrix(confusion_matrix)
  if (!all(dim(mat) == c(2, 2))) {
    stop("Truth table plot currently expects a 2x2 confusion matrix.")
  }

  tn <- mat[1, 1]
  fn <- mat[1, 2]
  fp <- mat[2, 1]
  tp <- mat[2, 2]

  safe_ratio <- function(num, den) {
    if (den == 0) {
      return(NA_real_)
    }
    num / den
  }

  precision <- safe_ratio(tp, tp + fp)
  sensitivity <- safe_ratio(tp, tp + fn)
  specificity <- safe_ratio(tn, tn + fp)
  metrics_text <- paste0(
    "Precision=", format_metric(precision),
    "   Sensitivity=", format_metric(sensitivity),
    "   Specificity=", format_metric(specificity)
  )

  cell_labels <- matrix(c("TN", "FN", "FP", "TP"), nrow = 2, byrow = TRUE)

  max_count <- max(mat)
  color_scale <- colorRampPalette(c("#eef4fb", "#1f5a99"))(100)
  scale_index <- function(value) {
    if (max_count == 0) {
      return(1)
    }
    min(100, max(1, floor((value / max_count) * 99) + 1))
  }

  par(mar = c(6, 8, 6, 2))
  plot(c(0, 2), c(0, 2), type = "n", xaxt = "n", yaxt = "n",
       xlab = "Actual Class", ylab = "Predicted Class",
       main = "Truth Table (Confusion Matrix)")
  mtext(metrics_text, side = 3, line = 0.8, cex = 0.9, font = 2)

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
      text(
        x = x_left + 0.5, y = y_bottom + 0.64,
        labels = cell_labels[i, j], cex = 1.1, font = 2, col = text_col
      )
      text(
        x = x_left + 0.5, y = y_bottom + 0.38,
        labels = value, cex = 1.5, font = 2, col = text_col
      )
    }
  }

  axis(1, at = c(0.5, 1.5), labels = colnames(mat), cex.axis = 1)
  axis(2, at = c(1.5, 0.5), labels = rownames(mat), las = 1, cex.axis = 1)
  box()
}

save_roc_plot <- function(fpr, tpr, auc_value, file_path, cv_roc_curves = NULL) {
  png(filename = file_path, width = 800, height = 600)
  plot_roc_on_current_device(fpr, tpr, auc_value, cv_roc_curves)
  dev.off()
}

build_roc_ggplot <- function(fpr, tpr, auc_value, cv_roc_curves = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for ggplot ROC plots.")
  }

  cv_summary <- NULL
  if (!is.null(cv_roc_curves) && length(cv_roc_curves) > 0) {
    cv_summary <- summarize_cv_roc_curves(cv_roc_curves)
  }

  envelope_label <- "CV envelope (mean +/- 1 SD)"
  test_label <- paste0("Test (AUC=", format_metric(auc_value), ")")

  line_labels <- c("Reference (random)")
  line_colors <- c("Reference (random)" = "gray50")
  line_linetypes <- c(2)
  line_widths <- c(0.8)

  reference_df <- data.frame(
    fpr = c(0, 1),
    tpr = c(0, 1),
    curve = "Reference (random)",
    stringsAsFactors = FALSE
  )
  test_df <- data.frame(
    fpr = fpr,
    tpr = tpr,
    curve = test_label,
    stringsAsFactors = FALSE
  )

  cv_mean_df <- NULL
  cv_envelope_df <- NULL
  if (!is.null(cv_summary)) {
    cv_mean_label <- paste0(
      "CV mean (", cv_summary$n_curves, " folds, AUC=",
      format_metric(cv_summary$mean_auc), ")"
    )

    cv_mean_df <- data.frame(
      fpr = cv_summary$fpr_grid,
      tpr = cv_summary$mean_tpr,
      curve = cv_mean_label,
      stringsAsFactors = FALSE
    )

    cv_envelope_df <- data.frame(
      fpr = cv_summary$fpr_grid,
      lower_tpr = cv_summary$lower_tpr,
      upper_tpr = cv_summary$upper_tpr,
      band = envelope_label,
      stringsAsFactors = FALSE
    )

    line_labels <- c(line_labels, cv_mean_label)
    line_colors <- c(line_colors, setNames("gray35", cv_mean_label))
    line_linetypes <- c(line_linetypes, 1)
    line_widths <- c(line_widths, 1.2)
  }

  line_labels <- c(line_labels, test_label)
  line_colors <- c(line_colors, setNames("#1f5a99", test_label))
  line_linetypes <- c(line_linetypes, 1)
  line_widths <- c(line_widths, 1.4)

  p <- ggplot2::ggplot()

  if (!is.null(cv_envelope_df)) {
    p <- p +
      ggplot2::geom_ribbon(
        data = cv_envelope_df,
        mapping = ggplot2::aes(x = fpr, ymin = lower_tpr, ymax = upper_tpr, fill = band),
        alpha = 0.25,
        color = NA
      )
  }

  p <- p +
    ggplot2::geom_line(
      data = reference_df,
      mapping = ggplot2::aes(x = fpr, y = tpr, color = curve),
      linewidth = 0.8,
      linetype = 2
    )

  if (!is.null(cv_mean_df)) {
    p <- p +
      ggplot2::geom_line(
        data = cv_mean_df,
        mapping = ggplot2::aes(x = fpr, y = tpr, color = curve),
        linewidth = 1.2
      )
  }

  p <- p +
    ggplot2::geom_line(
      data = test_df,
      mapping = ggplot2::aes(x = fpr, y = tpr, color = curve),
      linewidth = 1.4
    ) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    ggplot2::scale_color_manual(values = line_colors, breaks = line_labels) +
    ggplot2::labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = paste0("ROC Curves: CV Mean +/- SD Envelope + Test (Test AUC = ", format_metric(auc_value), ")"),
      color = NULL,
      fill = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = ggplot2::element_rect(fill = "white", color = NA)
    ) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        order = 1,
        override.aes = list(linetype = line_linetypes, linewidth = line_widths)
      )
    )

  if (!is.null(cv_envelope_df)) {
    p <- p +
      ggplot2::scale_fill_manual(values = setNames(rgb(0.45, 0.45, 0.45, alpha = 0.25), envelope_label)) +
      ggplot2::guides(fill = ggplot2::guide_legend(order = 2))
  }

  p
}

plot_roc_ggplot_on_current_device <- function(fpr, tpr, auc_value, cv_roc_curves = NULL) {
  print(build_roc_ggplot(fpr, tpr, auc_value, cv_roc_curves))
}

save_roc_plot_ggplot <- function(fpr, tpr, auc_value, file_path, cv_roc_curves = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("Skipping ggplot ROC plot: package 'ggplot2' is not installed.")
    return(invisible(FALSE))
  }

  roc_plot <- build_roc_ggplot(fpr, tpr, auc_value, cv_roc_curves)
  ggplot2::ggsave(
    filename = file_path,
    plot = roc_plot,
    width = 8,
    height = 6,
    dpi = 120
  )
  invisible(TRUE)
}

save_truth_table_plot <- function(confusion_matrix, file_path) {
  png(filename = file_path, width = 900, height = 700)
  plot_truth_table_on_current_device(confusion_matrix)
  dev.off()
}

print_workflow_summary <- function(results, data_source_label) {
  cat("\n=== Configuration ===\n")
  cat("Data source: ", data_source_label, "\n", sep = "")
  cat("Seed: ", results$config$seed, "\n", sep = "")
  cat("Train fraction: ", results$config$train_fraction, "\n", sep = "")
  cat("k folds: ", results$config$k_folds, "\n", sep = "")
  cat("Top K genes: ", results$config$top_k_genes, "\n", sep = "")
  cat("Positive class: ", results$config$positive_class, "\n", sep = "")
  cat("Threshold: ", results$config$threshold, "\n", sep = "")

  cat("\n=== Dataset Summary ===\n")
  cat("Samples: ", results$dataset_summary$n_samples, "\n", sep = "")
  cat("Gene features: ", results$dataset_summary$n_genes, "\n", sep = "")
  cat("Class counts:\n")
  print(results$dataset_summary$class_counts)

  cat("\n=== Stratified Split ===\n")
  cat("Train samples: ", results$split_summary$train_n, "\n", sep = "")
  cat("Test samples: ", results$split_summary$test_n, "\n", sep = "")
  cat("Train class counts:\n")
  print(results$split_summary$train_class_counts)
  cat("Test class counts:\n")
  print(results$split_summary$test_class_counts)

  cat("\n=== Cross-Validation (Train Only) ===\n")
  for (fold in seq_len(length(results$cv$auc_values))) {
    cat(
      "Fold ", fold,
      ": AUC=", format_metric(results$cv$auc_values[fold]),
      ", Precision=", format_metric(results$cv$precision_values[fold]),
      "\n", sep = ""
    )
  }
  cat(
    "CV AUC mean +/- sd: ",
    format_metric(results$cv$auc_mean), " +/- ", format_metric(results$cv$auc_sd), "\n",
    sep = ""
  )
  cat(
    "CV Precision mean +/- sd: ",
    format_metric(results$cv$precision_mean), " +/- ", format_metric(results$cv$precision_sd), "\n",
    sep = ""
  )

  cat("\n=== Final Test Evaluation ===\n")
  cat("Confusion matrix (rows=Predicted, cols=Actual):\n")
  print(results$test$confusion_matrix)
  cat(
    "TP: ", results$test$tp,
    "  FP: ", results$test$fp,
    "  TN: ", results$test$tn,
    "  FN: ", results$test$fn, "\n", sep = ""
  )
  cat("Precision: ", format_metric(results$test$precision), "\n", sep = "")
  cat("ROC AUC: ", format_metric(results$test$auc), "\n", sep = "")
  cat("MSE (probability-based): ", format_metric(results$test$mse), "\n", sep = "")
  cat("R^2 (probability-based): ", format_metric(results$test$r_squared), "\n", sep = "")
}
