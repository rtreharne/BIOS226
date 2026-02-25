# ============================================================================
# Script: run_tumor_model_quick.R
# Purpose:
#   Fast, teaching-focused runner for the Topic 5 supervised learning workflow.
#
# What this script does:
#   1) Sources the data generator and modeling helper scripts.
#   2) Builds a synthetic tumor dataset using user-editable parameters.
#   3) Runs a leakage-safe train/CV/test workflow.
#   4) Plots:
#      - ROC with random baseline, CV mean curve, CV envelope (+/- 1 SD), and
#        final held-out test ROC.
#      - Truth table (confusion matrix) heatmap.
#   5) Adds the active threshold value under the truth table.
#
# Required files (same folder as this script):
#   - generate_tumor_dataset.R
#   - model_workflow_helpers.R
#
# How to run (RStudio Console):
#   source("run_tumor_model_quick.R")
#
# Outputs:
#   - Plot pane only (no PNG files are saved by this script).
#   - In-memory objects: `df` (dataset) and `results` (workflow outputs).
# ============================================================================

# Load dataset generator function: generate_tumor_dataset(...)
source("generate_tumor_dataset.R")

# Load workflow functions: run_tumor_workflow(...), plotting helpers, formatting
source("model_workflow_helpers.R")

# Dataset parameters (edit these)
data_seed <- 123            # Reproducibility seed for synthetic data creation.
n_samples <- 500          # Number of samples/rows (patients) to simulate.
n_genes <- 10000            # Number of Gene_* feature columns to generate.
n_informative <- 50         # Number of truly signal-carrying genes.
class_proportion <- 0.6     # Proportion for Luminal_A class (0 < value < 1).
noise_sd <- 2.0          # Noise SD in gene values; higher = harder problem.

# Model parameters (edit these)
model_seed <- 123           # Seed for train/test split and CV fold assignment.
train_fraction <- 0.8       # Fraction sent to training set (rest held out test).
k_folds <- 5                # Number of CV folds used on training data only.
top_k_genes <- 25           # Number of top-ranked genes kept for modeling.
positive_class <- "Basal_like"  # Label treated as positive class (binary = 1).
threshold <- 0.5# Probability cutoff for positive class prediction.

# Generate synthetic dataset from the parameter block above.
df <- generate_tumor_dataset(
  seed = data_seed,                 # Generator seed.
  n_samples = n_samples,            # Total samples.
  n_genes = n_genes,                # Total gene features.
  n_informative = n_informative,    # Informative signal features.
  class_proportion = class_proportion,  # Luminal_A fraction.
  noise_sd = noise_sd               # Noise level.
)

# Run leakage-safe modeling workflow and collect all outputs.
results <- run_tumor_workflow(
  df = df,                          # Input dataset.
  seed = model_seed,                # Split/CV reproducibility seed.
  train_fraction = train_fraction,  # Train/test partition ratio.
  k_folds = k_folds,                # CV folds.
  top_k_genes = top_k_genes,        # Feature selection size.
  positive_class = positive_class,  # Positive class label.
  threshold = threshold             # Classification threshold.
)

# Plot ROC: random baseline + CV mean/envelope + final test ROC curve.
plot_roc_on_current_device(
  fpr = results$test$roc_fpr,
  tpr = results$test$roc_tpr,
  auc_value = results$test$auc,
  cv_roc_curves = results$cv$roc_curves
)

# Plot confusion matrix truth table on current graphics device.
plot_truth_table_on_current_device(
  confusion_matrix = results$test$confusion_matrix
)

# Annotate truth table with the threshold used for predicted classes.
mtext(
  paste0(
    "  Threshold=", format_metric(threshold)
  ),
  side = 1, line = 4.2, cex = 0.78, font = 2
)
