# biostring_test.R
# BIOS226 — Biostrings sanity + wow test
# Generates a reproducible DNA sequence from your student ID
# and visualises k-mer composition

# ---- Load package ----
if (!requireNamespace("Biostrings", quietly = TRUE)) {
  stop("Biostrings is not installed. Please install it before running this script.")
}
library(Biostrings)

# ---- Get student ID ----
student_id <- readline("Enter your 9-digit student ID: ")

if (!grepl("^\\d{9}$", student_id)) {
  stop("Student ID must be exactly 9 digits.")
}

# ---- Use ID to set reproducible seed ----
set.seed(as.integer(substr(student_id, 1, 7)))

# ---- Generate DNA sequence ----
dna <- DNAString(
  paste(
    sample(c("A", "C", "G", "T"), size = 1000, replace = TRUE),
    collapse = ""
  )
)

# ---- Compute k-mer frequencies (trinucleotides) ----
k <- 3
kmers <- oligonucleotideFrequency(dna, width = k)
kmers <- sort(kmers, decreasing = TRUE)

top_kmers <- kmers[1:20]

# ---- Plot ----
par(mar = c(10, 4, 4, 1))

barplot(
  top_kmers,
  col = "steelblue",
  las = 2,
  ylab = "Frequency",
  main = paste(
    "Top trinucleotide frequencies\nStudent ID:",
    student_id
  )
)

# ---- Message ----
cat("\nBiostrings is working correctly.\n")
cat("DNA length:", length(dna), "bp\n")
cat("Top trinucleotide:", names(top_kmers)[1], "\n")

