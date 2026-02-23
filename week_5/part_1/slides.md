---
marp: true
title: "BIOS226 - Topic 5 - Supervised Learning (Part 1)"
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

.question-box {
  margin-top: 12px;
  max-width: 64%;
  border: 1.5px solid #1f5a99;
  border-radius: 8px;
  background: #f4f8fd;
  padding: 8px 10px;
  font-size: 0.8em;
}

.question-box p {
  margin: 0 0 4px 0;
  font-weight: 700;
  color: #133b63;
}

.question-box ul {
  margin: 0;
  padding-left: 18px;
}
</style>

<!-- _class: title -->
<img src="img/uol_logo.png" class="title-logo" alt="University of Liverpool logo">

# BIOS226 - Topic 5 - Supervised Learning (Part 1)
## Foundations for Biological Model Evaluation

Dr. Robert Treharne

---

# From Exploration to Prediction

- PCA and clustering show structure in biological data.
- Supervised learning moves us from pattern finding to outcome prediction.
- Practical impact: treatment and risk decisions can be data-driven.

**Example: Oncotype DX**
- Gene expression from tumour tissue is used to estimate recurrence risk.
- That estimate supports chemotherapy decision-making.

<div class="question-box">
<p>Questions</p>
<ul>
  <li>What does Oncotype DX measure?</li>
  <li>Is this classification or regression?</li>
</ul>
</div>

---

# Unsupervised vs Supervised

**Unsupervised learning**
- Finds structure without known labels (for example PCA).

**Supervised learning**
- Learns from labelled examples to predict outcomes.

**Example: The Cancer Genome Atlas (TCGA) subtype work**
- Molecular profiles improved subtype definitions beyond histology.

---

# What Are X and Y?

- **X**: measurable input features
- **Y**: outcome to predict

**Biological examples**
- SNPs -> disease risk
- Gene expression -> cancer subtype
- Clinical + genomic data -> prognosis

<div class="question-box">
<p>Questions</p>
<ul>
  <li>What is a SNP in practical terms?</li>
  <li>Why can many small SNP effects still become predictive?</li>
</ul>
</div>

---

# Classification vs Regression

**Classification**
- Predicts a category (for example subtype A vs B).

**Regression**
- Predicts a continuous value (for example drug response level).

**Example**
- Drug-response models estimate the strength of expected treatment effect.

---

# What Is a Model?

- A model is a function that maps **X -> Y**.
- It learns from training data, then predicts on new samples.

**Common models**
- Logistic regression
- Random forest
- Support vector machine

**Clinical example**
- Early-warning systems can estimate sepsis risk before diagnosis.

---

# The High-Dimensional Problem (p >> n)

- In omics, features (genes) can far exceed patients.
- This can make models unstable and easy to overfit.
- Apparent high accuracy may not replicate in new datasets.

**Example**
- Early microarray studies often reported strong results that later failed replication.

<div class="question-box">
<p>Questions</p>
<ul>
  <li>Why does p &gt;&gt; n increase overfitting risk?</li>
  <li>How can noise create false signatures in small cohorts?</li>
</ul>
</div>

---

# Why Biology Is Harder

Biological datasets are often:

- noisy
- small
- batch-affected
- biologically heterogeneous

**Key risk**
- A model may learn technical artefacts (for example batch or depth) instead of biology.

**Take-home**
- Good validation design is as important as model choice.

---

# Common Non-Biological Applications

Supervised learning is widely used outside biology, for example:

- Email spam filtering
- Fraud detection in banking and payments
- Credit-risk scoring for lending
- Product and content recommendation systems
- Demand forecasting and inventory planning
- Predictive maintenance for machines and vehicles
- Speech recognition and language translation
- Computer vision for quality control in manufacturing

---

# Biological Applications Beyond Omics

Other supervised-learning scenarios in biology include:

- Medical image diagnosis (for example chest X-ray, retinal imaging, MRI)
- Digital pathology slide classification
- ECG-based arrhythmia detection
- ICU deterioration and sepsis risk prediction from vital signs
- Microscopy image classification (cell type, morphology, localisation)
- Crop disease detection from plant images
- Wildlife species identification from camera traps or acoustic recordings
