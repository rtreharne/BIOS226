# ============================================================
# BIOS226 - Week 5
# Main script (heavily annotated): train and evaluate a classifier
# ============================================================
#
# WHY THIS FILE EXISTS
# --------------------
# This file is intentionally verbose so you can understand each step.
# It is the "teaching runner":
# - you edit parameters here,
# - then run this script end-to-end.
#
# All detailed modeling logic lives in:
#   `week_5/scripts/model_workflow_helpers.R`
# and data generation lives in:
#   `week_5/scripts/generate_tumor_dataset.R`
#
# This separation keeps the main script readable.


# ============================================================
# STEP 0) Paths to support scripts
# ============================================================
#
# These paths auto-resolve whether you run from project root
# or from inside `week_5/scripts`.

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
# STEP 1) Choose data source mode
# ============================================================
#
# `use_generator`
# TRUE  -> generate a fresh synthetic dataset now (in memory)
# FALSE -> load an existing CSV from `input_csv`
#
# Recommended for learning:
# TRUE, so you can tweak simulation settings and rerun immediately.

use_generator <- TRUE

# Used only if `use_generator <- FALSE`.
input_csv <- if (file.exists("week_5/scripts/tumor_dataset_seed_123.csv")) {
  "week_5/scripts/tumor_dataset_seed_123.csv"
} else {
  "tumor_dataset_seed_123.csv"
}


# ============================================================
# STEP 2) Dataset generation parameters (only used when TRUE above)
# ============================================================
#
# These map directly to `generate_tumor_dataset(...)`.
# Change these when you want easier/harder synthetic tasks.

# Reproducibility seed for synthetic dataset generation.
# Same seed + same generation parameters = same dataset.
dataset_seed <- 123

# Number of samples (patients / rows).
dataset_n_samples <- 120

# Number of gene features (Gene_1 ... Gene_n).
# High value simulates high-dimensional omics (p >> n).
dataset_n_genes <- 10000

# Number of truly informative genes carrying class signal.
# Remaining genes are noise-only.
dataset_n_informative <- 50

# Proportion of Luminal_A class in generated data.
# 0.6 means about 60% Luminal_A and 40% Basal_like.
dataset_class_proportion <- 0.6

# Background noise level in expression values (standard deviation).
# Larger value = harder separation between classes.
dataset_noise_sd <- 1.5


# ============================================================
# STEP 3) Modeling / evaluation parameters
# ============================================================
#
# These control splitting, CV, feature selection, and classification threshold.

# Reproducibility seed for model split and CV assignment.
model_seed <- 123

# Fraction of samples used for training.
# Remaining fraction is held out for final test evaluation.
train_fraction <- 0.8

# Number of CV folds used on training set only.
k_folds <- 5

# Number of top genes selected using train-only signal ranking.
top_k_genes <- 25

# Positive class label for binary metrics (precision, ROC, AUC).
positive_class <- "Basal_like"

# Probability threshold for class predictions.
# Example: probability >= 0.5 -> predict positive class.
threshold <- 0.85


# ============================================================
# STEP 4) Plot behavior settings
# ============================================================
#
# `show_plots_live`
# TRUE  -> display figures in the current graphics device (RStudio plot pane, etc.)
# FALSE -> do not display live plots
#
# NOTE:
# Live plotting requires an interactive R session (RStudio or interactive R).
# If you run via `Rscript`, plots are saved to files but not shown live.

show_plots_live <- TRUE

# Optional: if TRUE and interactive, open a new graphics window per plot.
# In many setups (e.g., RStudio), keeping FALSE is cleaner because plots
# appear in the existing Plot pane.
open_new_device_for_live_plots <- FALSE


# ============================================================
# STEP 5) Load external function definitions
# ============================================================
#
# `source(...)` executes another R script and imports its functions
# into the current session.

source(generator_script)
source(helper_script)


# ============================================================
# STEP 6) Build or load the dataset as data frame `df`
# ============================================================

if (use_generator) {
  # Generate synthetic data now.
  # This also writes a CSV file to your current working directory
  # with name: tumor_dataset_seed_<seed>.csv
  df <- generate_tumor_dataset(
    seed = dataset_seed,
    n_samples = dataset_n_samples,
    n_genes = dataset_n_genes,
    n_informative = dataset_n_informative,
    class_proportion = dataset_class_proportion,
    noise_sd = dataset_noise_sd
  )

  # Text label for summary output so you know where data came from.
  data_source_label <- paste0(
    "Generated in script (seed=", dataset_seed,
    "); CSV saved as tumor_dataset_seed_", dataset_seed, ".csv in current working directory"
  )
} else {
  # Load existing CSV mode.
  if (!file.exists(input_csv)) {
    stop(paste0("Input file not found: ", input_csv))
  }

  # Keep strings as characters and preserve exact column names.
  df <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
  data_source_label <- paste0("Loaded from CSV: ", input_csv)
}


# ============================================================
# STEP 7) Run leakage-safe training and evaluation
# ============================================================
#
# `run_tumor_workflow(...)` performs all core analysis:
# 1) validates schema
# 2) stratified train/test split
# 3) CV on training set only
# 4) train-only feature selection and scaling
# 5) logistic regression fit
# 6) final held-out test metrics
#
# Returned object `results` is a list containing:
# - configuration snapshot
# - dataset and split summaries
# - fold-wise and aggregate CV metrics
# - final test metrics
# - confusion matrix and ROC coordinates

results <- run_tumor_workflow(
  df = df,
  seed = model_seed,
  train_fraction = train_fraction,
  k_folds = k_folds,
  top_k_genes = top_k_genes,
  positive_class = positive_class,
  threshold = threshold
)


# ============================================================
# STEP 8) Save figures to disk
# ============================================================
#
# We always save files so outputs are reproducible and shareable.
# ROC output contains:
# - CV mean ROC line
# - shaded CV envelope (mean +/- 1 SD)
# - final test ROC line
# We also save a ggplot2 version of the same ROC content for comparison.

roc_plot_file <- paste0("roc_curve_test_seed_", model_seed, ".png")
roc_plot_file_ggplot <- paste0("roc_curve_test_seed_", model_seed, "_ggplot2.png")
truth_table_plot_file <- paste0("truth_table_test_seed_", model_seed, ".png")

save_roc_plot(
  fpr = results$test$roc_fpr,
  tpr = results$test$roc_tpr,
  auc_value = results$test$auc,
  file_path = roc_plot_file,
  cv_roc_curves = results$cv$roc_curves
)

ggplot_roc_saved <- save_roc_plot_ggplot(
  fpr = results$test$roc_fpr,
  tpr = results$test$roc_tpr,
  auc_value = results$test$auc,
  file_path = roc_plot_file_ggplot,
  cv_roc_curves = results$cv$roc_curves
)

save_truth_table_plot(
  confusion_matrix = results$test$confusion_matrix,
  file_path = truth_table_plot_file
)


# ============================================================
# STEP 9) Optionally display figures live
# ============================================================
#
# "Live" means drawn to the current graphics device immediately.
# This is useful during teaching/debugging while still keeping saved files.

if (show_plots_live) {
  if (interactive()) {
    if (open_new_device_for_live_plots) {
      dev.new(width = 8, height = 6)
    }
    # Live ROC view includes CV mean + envelope + final test ROC.
    plot_roc_on_current_device(
      fpr = results$test$roc_fpr,
      tpr = results$test$roc_tpr,
      auc_value = results$test$auc,
      cv_roc_curves = results$cv$roc_curves
    )

    if (open_new_device_for_live_plots) {
      dev.new(width = 8, height = 6)
    }
    plot_truth_table_on_current_device(
      confusion_matrix = results$test$confusion_matrix
    )
  } else {
    cat(
      "\nLive plotting skipped: session is non-interactive.\n",
      "Tip: run with `source(\"week_5/scripts/train_evaluate_tumor_model.R\")` in RStudio ",
      "to see plots in the Plot pane.\n",
      sep = ""
    )
  }
}


# ============================================================
# STEP 10) Print textual summary
# ============================================================
#
# This prints:
# - configuration
# - class balance and split summary
# - CV performance
# - final test confusion matrix and metrics

print_workflow_summary(results, data_source_label)

cat("\nSaved ROC plot: ", roc_plot_file, "\n", sep = "")
if (isTRUE(ggplot_roc_saved)) {
  cat("Saved ROC plot (ggplot2): ", roc_plot_file_ggplot, "\n", sep = "")
} else {
  cat("Saved ROC plot (ggplot2): skipped (package 'ggplot2' not installed)\n", sep = "")
}
cat("Saved truth table plot: ", truth_table_plot_file, "\n", sep = "")
cat("Done.\n")
