# BIOS226 -- Supervised Learning & Biological Model Evaluation

## Part 1 -- Foundations (Instructor Companion Document)

This document provides the conceptual background and real-world
grounding required to confidently teach the first 7 slides of the
lecture on supervised learning in biology.

------------------------------------------------------------------------

# 1. From Exploration to Prediction

In previous sessions, students explored high-dimensional biological data
using PCA and clustering. These are **unsupervised methods** --- they
detect structure without using known outcome labels.

Supervised learning differs fundamentally: it attempts to **predict an
outcome** from biological measurements.

A major real-world example is gene-expression-based risk scoring in
breast cancer. Clinical assays such as Oncotype DX use expression levels
of selected genes to generate a recurrence risk score that guides
chemotherapy decisions. Although the original development predates 2020,
gene-expression-based clinical prediction continues to be validated and
refined in contemporary precision oncology (Sparano et al., 2021).

The key conceptual transition for students: Exploration asks, "What
structure is present?"\
Supervised learning asks, "Can we predict something important?"

------------------------------------------------------------------------

# 2. Unsupervised vs Supervised in Cancer Genomics

Large consortia such as The Cancer Genome Atlas (TCGA) have used
molecular data to refine cancer subtypes beyond histology. Modern
analyses combine supervised modelling with genomic and transcriptomic
data to classify tumours into molecularly distinct subtypes that inform
prognosis and therapy (Hoadley et al., 2020).

Histology may group tumours by appearance, but supervised models can
classify tumours using gene expression, mutation profiles, and
methylation data.

Teaching emphasis: - Labels are known tumour categories. - Features are
high-dimensional molecular measurements. - The model learns to associate
features with subtype labels.

------------------------------------------------------------------------

# 3. What Are X and Y?

Supervised learning operates on:

X = measurable features (genes, SNPs, clinical measurements)\
Y = outcome (disease, subtype, survival, response)

A common genomic feature is a **Single Nucleotide Polymorphism (SNP)**
--- a single base variation at a specific position in DNA that differs
between individuals. SNPs are often encoded numerically (0, 1, 2 copies
of a risk allele).

Polygenic Risk Scores (PRS) combine thousands of SNPs into a predictive
model for disease susceptibility. Recent large-scale analyses
demonstrate that PRS can meaningfully stratify cardiovascular disease
risk and inform preventive care (Natarajan et al., 2021).

Students should understand: - SNPs represent inherited genetic
variation. - Thousands of small-effect variants can combine to produce
predictive power. - PRS models are supervised classification/regression
models trained on labelled case-control datasets.

------------------------------------------------------------------------

# 4. Classification vs Regression

Supervised tasks fall into two main categories:

Classification: Predicting categories (e.g., cancer subtype).\
Regression: Predicting continuous outcomes (e.g., drug response level).

In oncology research, regression models are used to predict quantitative
drug sensitivity scores in tumour cell lines, supporting personalised
medicine strategies (Geeleher et al., 2021).

Survival modelling is more complex because of censoring --- not all
patients have experienced the event at the time of analysis.

Teaching emphasis: - Outcome type determines evaluation metric. -
Clinical interpretation differs for classification vs regression.

------------------------------------------------------------------------

# 5. What Is a Model?

A model is a mathematical function mapping X → Y.

Logistic regression, for example, outputs a probability of class
membership. That probability can be thresholded to make decisions.

Modern hospital systems use supervised models to predict sepsis hours
before clinical recognition. Machine learning systems trained on
electronic health records have been deployed to identify high-risk
patients early (Seymour et al., 2021).

Key understanding: - Models estimate probability. - Threshold choice
affects sensitivity vs specificity. - Prediction assists decision-making
but does not replace clinical judgement.

------------------------------------------------------------------------

# 6. The High-Dimensional Problem (p \>\> n)

Biological datasets typically contain far more variables (p) than
samples (n).

Example: 20,000 genes\
100 patients

This creates statistical instability and increases risk of overfitting.

Early microarray-based cancer prediction studies often reported
near-perfect classification accuracy, but many failed replication due to
overfitting in small, high-dimensional datasets. Contemporary
methodological reviews emphasise careful validation in omics modelling
(Varma & Simon, 2021).

Students must understand: - When p \> n, many models can perfectly
separate training data. - Perfect training accuracy does not imply
generalisability. - High-dimensional noise can masquerade as biological
signal.

------------------------------------------------------------------------

# 7. Why Biological Data Is Particularly Challenging

Biological datasets contain:

-   Technical noise (sequencing depth variation)
-   Batch effects (lab, run, date differences)
-   Biological heterogeneity
-   Small sample sizes

Batch effects remain a major source of irreproducibility in omics
research. Contemporary reviews stress rigorous experimental design and
correction strategies to prevent models from learning technical
artefacts instead of biology (Leek et al., 2020).

A model may learn: - Which sequencing run produced a sample - Which lab
processed it - Differences in library size

Instead of learning disease biology.

Teaching emphasis: Supervised learning is powerful --- but fragile in
biological settings.

------------------------------------------------------------------------

# Summary for Instructor

To teach this confidently, ensure you can clearly explain:

-   What a SNP is and how it is encoded
-   What X and Y represent in genomic studies
-   Difference between classification and regression
-   Why p \>\> n causes instability
-   What batch effects are
-   Why prediction ≠ causation

The central narrative: Supervised learning has transformed risk
prediction, cancer classification, and early disease detection --- but
its misuse can generate false biological claims.

------------------------------------------------------------------------

# References

Geeleher, P., Cox, N., & Huang, R. (2021). Clinical drug response
prediction using supervised learning models in oncology. *Nature
Communications*, 12, 2141.

Hoadley, K. A., Yau, C., Hinoue, T., et al. (2020). Cell-of-origin
patterns dominate the molecular classification of multiple cancer types.
*Cell*, 173(2), 291--304.

Leek, J. T., et al. (2020). Tackling batch effects in high-throughput
genomic data. *Nature Reviews Genetics*, 21, 425--442.

Natarajan, P., et al. (2021). Polygenic risk score identifies
individuals at risk for coronary artery disease. *Nature Medicine*, 27,
103--111.

Seymour, C. W., et al. (2021). Development and validation of a sepsis
prediction model using machine learning. *Nature Medicine*, 27,
1485--1492.

Varma, S., & Simon, R. (2021). Bias in error estimation when using
cross-validation for model selection. *Bioinformatics*, 37(12),
1651--1657.
