# BIOS226 -- Supervised Learning & Biological Model Evaluation

## Part 1 -- Foundations (Slides 1--7)

------------------------------------------------------------------------

## Slide 1 -- From Exploration to Prediction

We have explored structure in biological data using PCA and clustering.
Now we move from **exploration** to **prediction**.

**Example: Breast Cancer Recurrence (Oncotype DX)**\
Gene expression from tumour tissue is used to predict risk of recurrence
and guide chemotherapy decisions.

Supervised learning already influences: - Chemotherapy decisions -
Cancer subtype diagnosis - Drug response prediction

**Preparation Questions:** - What is Oncotype DX measuring? - Is this
classification or regression? - How is gene expression used as
predictive input?

------------------------------------------------------------------------

## Slide 2 -- Unsupervised vs Supervised

**Unsupervised:** Detect structure without labels (e.g., PCA).\
**Supervised:** Learn relationship between features and known outcomes.

**Example: TCGA Tumour Subtype Classification**\
Supervised models refined cancer subtype definitions beyond histology.

**Preparation Questions:** - Why might histology miss molecular
subtypes? - What are the "labels" in tumour subtype classification? -
What biological data serve as input features?

------------------------------------------------------------------------

## Slide 3 -- What Are X and Y?

X = Measurable biological features\
Y = Outcome we care about

Examples: - SNPs → Predict disease risk\
- Gene expression → Predict cancer subtype\
- Clinical + genomic data → Predict survival

**Example: Polygenic Risk Scores**\
Thousands of SNPs combined to estimate coronary artery disease risk.

**Preparation Questions:** - What is a SNP? - How are SNPs encoded
numerically? - Why can many small genetic effects combine into
predictive power?

------------------------------------------------------------------------

## Slide 4 -- Classification vs Regression

**Classification:** Categorical outcome (e.g., tumour subtype)\
**Regression:** Continuous outcome (e.g., drug response level)

**Example: Drug Response Prediction**\
Models predict how strongly a tumour may respond to chemotherapy.

**Preparation Questions:** - Why is survival modelling more complex than
simple regression? - When might regression outputs be converted into
risk categories?

------------------------------------------------------------------------

## Slide 5 -- What Is a Model?

A model is a mathematical function mapping X → Y.

Examples: - Logistic regression - Random forest - Support vector machine

**Example: Sepsis Early Warning Systems**\
Hospitals use supervised models to predict sepsis hours before clinical
diagnosis.

**Preparation Questions:** - What does logistic regression output? - Why
is probability useful in clinical decisions? - How does threshold choice
affect treatment decisions?

------------------------------------------------------------------------

## Slide 6 -- The High-Dimensional Problem (p \>\> n)

In omics: - Thousands of genes - Few patients

This creates statistical instability.

**Example: Early Microarray Studies (2000s)**\
Many reported near-perfect cancer classification accuracy, but failed
replication due to overfitting.

**Preparation Questions:** - Why does p \>\> n increase overfitting
risk? - What is noise in RNA-seq? - Why can small datasets produce false
gene signatures?

------------------------------------------------------------------------

## Slide 7 -- Why Biology Is Harder Than Machine Learning Competitions

Biological data is: - Noisy - Small - Batch-affected - Biologically
heterogeneous

**Example: Batch Effects in Gene Expression Studies**\
Models sometimes learn laboratory differences instead of disease
biology.

**Preparation Questions:** - What is a batch effect? - Could a model
learn sequencing depth instead of disease? - How do we detect and
prevent this?
