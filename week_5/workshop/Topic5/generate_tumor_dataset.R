# ============================================================
# BIOS226 - Week 5
# Synthetic RNA-seq-style tumor subtype dataset generator
# ============================================================
#
# This script creates a fully reproducible high-dimensional dataset
# for supervised binary classification teaching in breast cancer
# (Luminal_A vs Basal_like).
# It simulates log-expression-like values that are already normalized,
# so no additional normalization step is applied in this script.
#
# The setup reflects a common biological context where p >> n:
# many genes (features) and comparatively few samples.

generate_tumor_dataset <- function(
  seed = 123,
  n_samples = 120,
  n_genes = 5000,
  n_informative = 50,
  class_proportion = 0.6,
  noise_sd = 1
) {
  # ------------------------------
  # Input validation (base R only)
  # ------------------------------
  is_positive_integer <- function(x) {
    is.numeric(x) && length(x) == 1 && is.finite(x) && x > 0 && x == as.integer(x)
  }

  if (!(is.numeric(seed) && length(seed) == 1 && is.finite(seed))) {
    stop("`seed` must be a numeric scalar.")
  }
  if (!is_positive_integer(n_samples)) {
    stop("`n_samples` must be a positive integer.")
  }
  if (!is_positive_integer(n_genes)) {
    stop("`n_genes` must be a positive integer.")
  }
  if (!is_positive_integer(n_informative)) {
    stop("`n_informative` must be a positive integer.")
  }
  if (n_informative > n_genes) {
    stop("`n_informative` must be less than or equal to `n_genes`.")
  }
  if (!(is.numeric(class_proportion) &&
        length(class_proportion) == 1 &&
        is.finite(class_proportion) &&
        class_proportion > 0 &&
        class_proportion < 1)) {
    stop("`class_proportion` must be a numeric scalar between 0 and 1 (exclusive).")
  }
  if (!(is.numeric(noise_sd) &&
        length(noise_sd) == 1 &&
        is.finite(noise_sd) &&
        noise_sd > 0)) {
    stop("`noise_sd` must be a positive numeric scalar.")
  }

  # Full reproducibility for class labels and expression values.
  set.seed(seed)

  # -------------------------------------------------------
  # Generate labels with a slight class imbalance.
  # class_proportion gives the target proportion of Luminal_A.
  # -------------------------------------------------------
  subtype_a_name <- "Luminal_A"
  subtype_b_name <- "Basal_like"

  n_A <- round(n_samples * class_proportion)
  n_B <- n_samples - n_A

  subtype_labels <- sample(
    c(rep(subtype_a_name, n_A), rep(subtype_b_name, n_B)),
    size = n_samples,
    replace = FALSE
  )

  # -------------------------------------------------------
  # Generate baseline expression (noise genes):
  # approximately normal values with mean 0 and sd = noise_sd,
  # representing already log-transformed / normalized expression.
  # -------------------------------------------------------
  expression_matrix <- matrix(
    rnorm(n = n_samples * n_genes, mean = 0, sd = noise_sd),
    nrow = n_samples,
    ncol = n_genes
  )

  # -------------------------------------------------------
  # Add signal to exactly n_informative genes:
  # for Basal_like samples, shift the informative genes by +1.0.
  # All non-informative genes remain pure random noise.
  # -------------------------------------------------------
  informative_gene_idx <- seq_len(n_informative)
  subtype_b_idx <- which(subtype_labels == subtype_b_name)

  if (length(subtype_b_idx) > 0 && n_informative > 0) {
    expression_matrix[subtype_b_idx, informative_gene_idx] <-
      expression_matrix[subtype_b_idx, informative_gene_idx] + 1.0
  }

  # Gene columns are named Gene_1 ... Gene_n.
  colnames(expression_matrix) <- paste0("Gene_", seq_len(n_genes))

  # Build final data frame with Patient_ID first and Subtype second.
  # Gene columns follow after these identifiers.
  gene_df <- as.data.frame(expression_matrix)
  df <- data.frame(
    Patient_ID = paste0("Patient_", seq_len(n_samples)),
    Subtype = factor(subtype_labels, levels = c(subtype_a_name, subtype_b_name)),
    gene_df,
    check.names = FALSE
  )

  # -------------------------------------------------------
  # Write CSV output with seed in filename.
  # The file is written to the current working directory.
  # -------------------------------------------------------
  output_file <- paste0("tumor_dataset_seed_", seed, ".csv")
  write.csv(df, output_file, row.names = FALSE)

  # Return dataset for immediate use in R.
  return(df)
}

# Example usage (run these lines manually in the console):
# df <- generate_tumor_dataset(seed = 123)
# dim(df)
# table(df$Subtype)
