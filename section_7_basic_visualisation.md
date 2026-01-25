## 7. Basic visualisation

Visualisation is one of the most important steps in data analysis.  
Before running statistical tests or models, we should **look at the data**.

Plots help us to:
- understand patterns and trends
- compare experimental groups
- identify variability and potential outliers

---

### Response variables and independent variables

In most biological experiments, we distinguish between two types of variables:

- **Response variable**  
  This is the outcome you measure.  
  In this dataset, the response variable is:

  growth_rate

- **Independent variable**  
  This is a variable that may influence the response.  
  In this dataset, independent variables include:

  condition  
  temperature  

When we make plots, we usually place:
- the **independent variable** on the x-axis
- the **response variable** on the y-axis

---

### Plot 1: Grouped boxplot (growth rate by treatment)

A boxplot is useful for comparing a response variable across discrete groups.

To create a boxplot of growth rate for each treatment condition, add and run:

```r
boxplot(growth_rate ~ condition,
        data = data,
        xlab = "Treatment condition",
        ylab = "Growth rate",
        main = "Growth rate by treatment condition")
```

What this plot shows:
- one box per treatment group
- the median growth rate in each group
- the spread and variability within each group
- potential outliers

This plot is especially useful when preparing for statistical tests such as ANOVA.

---

### Plot 2: Scatterplot (temperature vs growth rate)

A scatterplot is used to explore the relationship between two **continuous variables**.

To plot temperature against growth rate, add and run:

```r
plot(data$temperature,
     data$growth_rate,
     xlab = "Temperature (°C)",
     ylab = "Growth rate",
     main = "Growth rate vs temperature")
```

In this plot:
- temperature is the independent variable (x-axis)
- growth rate is the response variable (y-axis)
- each point represents a single sample

Scatterplots help you assess whether changes in an independent variable are associated with changes in the response variable.

---

### Why these plots matter

Together, these two plots allow you to:
- compare growth rate across treatment groups
- explore how growth rate varies with temperature
- develop intuition before formal statistical analysis

Visualisation is not optional — it is a core part of good biological data analysis.
