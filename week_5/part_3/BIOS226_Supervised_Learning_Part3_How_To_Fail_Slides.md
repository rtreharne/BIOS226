---
marp: true
title: "BIOS226 - Topic 5 - Supervised Learning (Part 3)"
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
  max-height: 270px;
  object-fit: contain;
  display: block;
  margin: 16px auto 46px auto;
}

section.figure {
  text-align: center;
  padding-bottom: 140px;
}

section.figure h1 {
  margin-bottom: 0.3em;
}

section.figure p {
  margin-top: 0.2em;
}

section.overview {
  padding-bottom: 120px;
}
</style>

<!-- _class: title -->
<img src="img/uol_logo.png" class="title-logo" alt="University of Liverpool logo">

# BIOS226 - Topic 5 - Supervised Learning (Part 3)
## How To Fail

Dr. Robert Treharne

---

<!-- _class: overview -->
# Overview

In supervised learning, models rarely fail loudly. They fail silently.

Common failure modes:

- Overfitting (learning noise instead of signal)
- Underfitting (model too simple)
- Data leakage (information from test data enters training)
- Ignoring class imbalance
- Choosing the wrong threshold for the clinical context
- Evaluating on training data only

---

<!-- _class: figure -->
![ROC Scenario 0](img/roc_scenario_0_perfect_high_n_high_p_seed_1006.png)

---

<!-- _class: figure -->
![ROC Scenario 1 seed 1201](img/roc_scenario_1_overfitting_seed_1201.png)

---

<!-- _class: figure -->
![ROC Scenario 2](img/roc_scenario_2_underfitting_seed_1002.png)

---

<!-- _class: figure -->
![ROC Scenario 3](img/roc_scenario_3_wrong_labels_seed_1003.png)

---

<!-- _class: figure -->
![ROC Scenario 4](img/roc_scenario_4_feature_leakage_seed_1000.png)

---

<!-- _class: figure -->
![ROC Scenario 5](img/roc_scenario_5_ignore_imbalance_90_10_seed_1005.png)

---

# By the end of this topic, you should be able to:

- Explain the difference between exploratory and supervised learning in biological data
- Describe a leakage-safe supervised learning pipeline from raw data to final evaluation
- Interpret a confusion matrix and ROC curve in a clinical context
- Recognise common failure modes (overfitting, leakage, imbalance, label errors)
- Critically evaluate whether a model is genuinely generalisable or silently flawed
