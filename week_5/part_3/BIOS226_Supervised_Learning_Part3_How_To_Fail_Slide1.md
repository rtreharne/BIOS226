---
marp: true
paginate: true
size: "16:9"
theme: default
---

# BIOS226 -- Topic 5 -- Supervised Learning (Part 3)

## How To Fail

In supervised learning, models rarely fail loudly.

They fail silently.

Common failure modes:

-   Overfitting (learning noise instead of signal)
-   Underfitting (model too simple)
-   Data leakage (information from test data enters training)
-   Feature selection before splitting
-   Scaling before splitting
-   Ignoring class imbalance
-   Choosing the wrong threshold for the clinical context
-   Evaluating on training data only
-   Reporting only the best metric
-   Not controlling randomness (no reproducibility)

In this section, we will deliberately create and visualise these
failures using ROC curves.
