# Quick Topic 5 runner: live plots in RStudio, no plot files saved.

if (!interactive()) {
  stop("Run with source('week_5/scripts/run_tumor_model_quick.R') in RStudio.")
}

generator_candidates <- c(
  "week_5/scripts/generate_tumor_dataset.R",
  "generate_tumor_dataset.R"
)
helper_candidates <- c(
  "week_5/scripts/model_workflow_helpers.R",
  "week_5/scripts/model_workflow_helper.R",
  "model_workflow_helpers.R",
  "model_workflow_helper.R"
)

generator_script <- generator_candidates[file.exists(generator_candidates)][1]
helper_script <- helper_candidates[file.exists(helper_candidates)][1]
if (is.na(generator_script) || is.na(helper_script)) {
  stop("Could not find generate_tumor_dataset.R and/or model_workflow_helpers.R.")
}

source(generator_script)
source(helper_script)

# Dataset parameters (edit these)
data_seed <- 123
n_samples <- 120
n_genes <- 5000
n_informative <- 50
class_proportion <- 0.6
noise_sd <- 1

# Model parameters (edit these)
model_seed <- 123
train_fraction <- 0.8
k_folds <- 5
top_k_genes <- 25
positive_class <- "Basal_like"
threshold <- 0.85

df <- generate_tumor_dataset(
  seed = data_seed,
  n_samples = n_samples,
  n_genes = n_genes,
  n_informative = n_informative,
  class_proportion = class_proportion,
  noise_sd = noise_sd
)

# Generator writes a CSV by design; remove it to keep this run display-only.
generated_csv <- paste0("tumor_dataset_seed_", data_seed, ".csv")
if (file.exists(generated_csv)) {
  unlink(generated_csv)
}

results <- run_tumor_workflow(
  df = df,
  seed = model_seed,
  train_fraction = train_fraction,
  k_folds = k_folds,
  top_k_genes = top_k_genes,
  positive_class = positive_class,
  threshold = threshold
)

plot_roc_on_current_device(
  fpr = results$test$roc_fpr,
  tpr = results$test$roc_tpr,
  auc_value = results$test$auc,
  cv_roc_curves = results$cv$roc_curves
)

plot_truth_table_on_current_device(
  confusion_matrix = results$test$confusion_matrix
)

safe_ratio <- function(num, den) if (den == 0) NA_real_ else num / den
tp <- results$test$tp
tn <- results$test$tn
fp <- results$test$fp
fn <- results$test$fn
sensitivity <- safe_ratio(tp, tp + fn)
precision <- safe_ratio(tp, tp + fp)
accuracy <- safe_ratio(tp + tn, tp + tn + fp + fn)

mtext(
  paste0(
    "TP=", tp, "  TN=", tn, "  FP=", fp, "  FN=", fn,
    "  Sensitivity=", format_metric(sensitivity),
    "  Accuracy=", format_metric(accuracy),
    "  Precision=", format_metric(precision),
    "  Threshold=", format_metric(threshold)
  ),
  side = 1, line = 4.2, cex = 0.78, font = 2
)
