---
marp: true
title: "BIOS226 - Topic 5 - Supervised Learning (Part 2)"
theme: default
paginate: true
size: 16:9
---

<style>
section::before {
  content: "";
  position: absolute;
  top: 20px;
  right: 24px;
  width: 180px;
  height: 90px;
  background: url("img/uol_logo.png") no-repeat right top;
  background-size: contain;
  pointer-events: none;
}

section.title::before {
  display: none;
}

section::after {
  bottom: 20px;
  content: attr(data-marpit-pagination) " / " attr(data-marpit-pagination-total);
  font-size: 0.7em;
  position: absolute;
  right: 24px;
}

section:not([data-marpit-pagination])::after {
  display: none;
}

section {
  padding-bottom: 76px;
}

section ul {
  line-height: 1.25;
}

.title-logo {
  --logo-crop-left: 20px;
  height: 240px;
  margin-bottom: 12px;
  margin-left: calc(var(--logo-crop-left) * -1);
  clip-path: inset(0 0 0 var(--logo-crop-left));
}

.figure img {
  width: 68%;
  max-height: 350px;
  object-fit: contain;
  display: block;
  margin: 8px auto 14px auto;
}

section.figure {
  text-align: center;
}

section.figure h1 {
  margin-bottom: 0.35em;
}

section.figure p {
  margin-top: 0.2em;
}

section.truth img {
  width: 56%;
  max-height: 315px;
}

section.roc img {
  width: 62%;
  max-height: 330px;
}

section.fullscreen {
  padding: 0;
  background-color: #fff;
}

section.fullscreen::before,
section.fullscreen::after {
  display: none;
}
</style>

<!-- _class: title -->
<img src="img/uol_logo.png" class="title-logo" alt="University of Liverpool logo">

# BIOS226 - Topic 5 - Supervised Learning (Part 2)
## The Tumor Subtype Classification Pipeline

Dr. Robert Treharne

---

# From Data to Decision: The Pipeline

We will use our synthetic breast cancer gene expression dataset generated in R.

- 120 patients
- 10,000 gene features
- Subtypes: Luminal_A vs Basal_like
- Logistic regression classifier

Goal: Predict tumor subtype from gene expression profiles.

---

# 1. Define the Prediction Problem

Clinical question: Can we classify a tumor as *Basal_like* or *Luminal_A* based on gene expression?

Model task: Binary classification.

- Input: 10,000 gene expression values per patient
- Output: Probability tumor is Basal_like

Positive class: **Basal_like**

---

# 2. Validate the Data Structure

Before modeling, verify:

- Column structure (`Patient_ID`, `Subtype`, `Gene_1 ... Gene_n`)
- Exactly two classes present
- All gene columns are numeric
- No schema inconsistencies

Why? Garbage in leads to garbage out.

---

# 3. Stratified Train/Test Split

Split the data:

- 80% training
- 20% test
- Stratified by subtype

Why stratified? It maintains class balance in both sets.

The test set is held out and **never** used during training.

---

# 4. Feature Selection (Train Only)

Rank genes by absolute difference in mean expression between classes.

Select top K genes (e.g., 25).

Important: Feature selection is performed on training data only.

Why? To avoid data leakage.

---

# 5. Scaling & Normalisation (Train Only)

Gene expression values vary in magnitude.

Standardise each selected gene:

- Subtract mean
- Divide by standard deviation

Scaling parameters are learned from training data only and then applied to test data.

---

# 6. Cross-Validation on Training Data

Perform k-fold cross-validation (e.g., 5-fold):

1. Split training set into 5 parts
2. Train on 4 folds
3. Validate on 1 fold
4. Repeat 5 times

Outputs:
- AUC per fold
- Precision per fold

Purpose: Estimate **model stability** before touching test set.

---

# 7. Fit the Logistic Regression Model

Logistic regression estimates:

`P(Basal_like | gene expression)`

Model form:

`log(p / (1 - p)) = beta0 + beta1x1 + beta2x2 + ...`

**Output**: Probability between 0 and 1 for each patient.

---

# 8. Confusion Matrix: Understanding Errors

Using chosen threshold (e.g., 0.85 - high specificity! Why?), compute:

- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)

From this we calculate:

- Precision
- Sensitivity
- Specificity

This describes performance at one threshold.

---

<!-- _class: fullscreen -->
![bg contain](img/truth_table_test_seed_123.png)

---

# 9. ROC Curve & AUC

The model outputs probabilities.

If threshold varies, sensitivity and false positive rate change.

ROC curve shows performance across all thresholds:

- X-axis: False Positive Rate
- Y-axis: True Positive Rate

AUC measures discrimination ability:

- 0.5 = random
- 1.0 = perfect

---

<!-- _class: fullscreen -->
![bg contain](img/roc_curve_test_seed_123.png)

---

# 10. Final Test Evaluation and Discipline

After all training steps, evaluate once on the held-out test set.

Report:

- Confusion matrix
- Precision
- ROC AUC
- MSE (probability-based)
- R squared (probability-based)

This gives an unbiased estimate of real-world performance.

---

# Pipeline Summary

1. Define question
2. Validate data
3. Split train/test
4. Select features (train only)
5. Scale data (train only)
6. Cross-validate
7. Fit model
8. Evaluate with confusion matrix
9. Evaluate with ROC/AUC
10. Report final test performance
