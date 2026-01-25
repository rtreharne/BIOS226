# ============================================================
# BIOS226 – Week 1
# Generate example biological dataset (4 groups)
# ============================================================

# ------------------------------------------------------------
# Obtain student ID (works in RStudio AND terminal)
# ------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 1 && grepl("^[0-9]{9}$", args[1])) {
  # Running via Rscript with argument
  student_id <- args[1]
  
} else if (interactive()) {
  # Running interactively (e.g. RStudio)
  repeat {
    student_id <- readline("Enter your 9-digit student ID: ")
    if (grepl("^[0-9]{9}$", student_id)) break
    cat("Invalid input. Please enter exactly 9 digits.\n")
  }
  
} else {
  stop("Usage: Rscript generate_example_data.R <9-digit-student-ID>")
}

# Set seed for reproducibility
set.seed(as.numeric(student_id))


# ------------------------------------------------------------
# Experimental design
# ------------------------------------------------------------

n_per_group <- 20   # >= 15 as requested

condition <- factor(rep(
  c("Control", "Low_Treatment", "Medium_Treatment", "High_Treatment"),
  each = n_per_group
))

# Temperature values (°C) – shared range across groups
temperature <- rnorm(
  4 * n_per_group,
  mean = 30,
  sd = 2
)

# ------------------------------------------------------------
# Generate biologically plausible growth rates
# ------------------------------------------------------------

# Baseline relationship with temperature
baseline <- 0.80 + 0.01 * (temperature - 30)

growth_rate <- c(
  rnorm(n_per_group, mean = baseline[1:n_per_group] + 0.00, sd = 0.08),
  rnorm(n_per_group, mean = baseline[(n_per_group + 1):(2 * n_per_group)] + 0.10, sd = 0.08),
  rnorm(n_per_group, mean = baseline[(2 * n_per_group + 1):(3 * n_per_group)] + 0.20, sd = 0.08),
  rnorm(n_per_group, mean = baseline[(3 * n_per_group + 1):(4 * n_per_group)] + 0.30, sd = 0.08)
)

# ------------------------------------------------------------
# Create data frame
# ------------------------------------------------------------

example_data <- data.frame(
  sample_id   = paste0("S", seq_len(4 * n_per_group)),
  condition   = condition,
  temperature = round(temperature, 1),
  growth_rate = round(growth_rate, 3)
)

# ------------------------------------------------------------
# Inspect dataset
# ------------------------------------------------------------

head(example_data)
str(example_data)
summary(example_data)

# ------------------------------------------------------------
# Write to CSV
# ------------------------------------------------------------

write.csv(
  example_data,
  file = "example_bio_data.csv",
  row.names = FALSE
)

cat("\nDataset 'example_bio_data.csv' created successfully.\n")
