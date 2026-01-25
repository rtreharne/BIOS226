## 8. A first statistical model: ANOVA

In this section, you will run your **first formal statistical model** in BIOS226.

The goal here is **not** to master statistics.  
It is to understand **what question the model is asking**, **what the output means**, and **why we use it**.

---

### What is ANOVA?

ANOVA stands for **Analysis of Variance**.

It is used when you want to test whether the **mean value of a response variable** differs between **two or more groups**.

In our experiment:
- The **response variable** is `growth_rate`
- The **grouping variable** is `condition`
- There are multiple treatment groups

ANOVA answers the question:

Are the differences in mean growth rate between treatment groups larger than we would expect by chance alone?

---

### Running the ANOVA

Add the following lines to your script and run them:

```r
anova_model <- aov(growth_rate ~ condition, data = data)
summary(anova_model)
```

---

### What does this code do?

- `aov()` fits an ANOVA model
- `growth_rate ~ condition` specifies the model structure
- `data = data` tells R where to find the variables
- `anova_model` stores the fitted model
- `summary(anova_model)` displays the results

---

### Interpreting the output

The key value to focus on is the **p-value**.

- A **small p-value** (typically less than 0.05) suggests that at least one group mean differs
- A **large p-value** suggests that observed differences could plausibly be due to random variation

Important:
- ANOVA does **not** tell you which groups differ
- It only tells you whether **any difference exists**

---

### What if the ANOVA is significant?

If the ANOVA result is statistically significant, the next step is to perform **post-hoc tests**.

Post-hoc tests allow you to:
- Compare groups pairwise
- Identify which specific treatments differ from each other

You will encounter these ideas in more detail elsewhere in your degree.

For now, remember:
ANOVA is a starting point, not the end of the analysis.

---

### What about assumptions?

Like all statistical tests, ANOVA relies on assumptions, including:
- Independence of observations
- Normally distributed residuals
- Similar variance across groups

You will cover these assumptions formally in **BIOS203**.

For this workshop:
- Do not worry about checking assumptions yet
- Focus on understanding what the model is doing conceptually

---

### Why are we doing this now?

ANOVA introduces several ideas that will appear repeatedly in BIOS226:

- Modelling biological data
- Comparing experimental groups
- Interpreting statistical output
- Deciding what to do next in an analysis

Later in the module, these same ideas will extend to:
- More complex models
- High-dimensional data
- Supervised learning and model evaluation

This is your first step into statistical modelling.
