# Sure Tomorrow — Insurance Machine Learning & Data Obfuscation

## Project Overview

This project explores how machine learning and linear algebra can help the insurance company **Sure Tomorrow** identify similar customers, predict insurance benefit outcomes, and transform personal information while preserving predictive performance.

The project was developed in Python using Jupyter Notebook as part of a Data Science learning program.

## Business Objectives

The analysis addresses four main tasks:

1. **Customer similarity:** Identify customers with similar characteristics using k-nearest neighbors.
2. **Insurance benefit classification:** Predict whether a customer is likely to receive at least one insurance benefit.
3. **Insurance benefit regression:** Predict the number of benefits a customer may receive.
4. **Data obfuscation:** Apply an invertible matrix transformation to personal data and demonstrate that ordinary linear regression predictions remain effectively unchanged.

## Dataset

The dataset contains **5,000 insurance customer records** with the following variables:

| Feature | Description |
|---|---|
| `gender` | Encoded customer gender |
| `age` | Customer age |
| `income` | Customer income |
| `family_members` | Number of family members |
| `insurance_benefits` | Number of insurance benefits received |

### Data Preparation

- Checked for missing values and duplicate records.
- Reviewed numerical ranges and potential outliers.
- Preserved records without sufficient evidence of data entry errors.
- Applied `MaxAbsScaler` where appropriate.
- Used group-aware splitting to prevent identical feature profiles from appearing in both training and test sets.

The binary classification target was defined as:

`insurance_benefits_received = 1` if `insurance_benefits > 0`, and `0` otherwise.

Approximately **11.28%** of customers received at least one benefit, indicating a class imbalance.

## Methodology

### 1. Customer Similarity — k-Nearest Neighbors

Customer similarity was evaluated using:

- Euclidean distance with original features.
- Manhattan distance with original features.
- Euclidean distance with scaled features.
- Manhattan distance with scaled features.

**Finding:** Feature scaling significantly affected nearest-neighbor selection. Without scaling, income tended to dominate distances because of its larger numerical magnitude.

### 2. Binary Classification

A k-nearest neighbors classifier was used to predict whether a customer received at least one insurance benefit.

The evaluation included:

- A group-aware training/test split.
- Five-fold group-based cross-validation.
- Hyperparameter evaluation for `k = 1` through `10`.
- Scaling performed within the cross-validation pipeline.
- Comparison against a randomized dummy baseline.
- F1 score as the primary classification metric.

### 3. Linear Regression

A custom ordinary least squares regression model was implemented using matrix operations.

Models were evaluated on original and scaled features using:

- Root Mean Squared Error (RMSE).
- Coefficient of Determination (R²).

### 4. Personal Data Obfuscation

An invertible random matrix was used to transform the four personal features.

The project demonstrates that, under the appropriate invertibility conditions, the regression coefficients can be transformed to preserve the same predictions.

The mathematical proof explicitly accounts for the model intercept. Computational results were also used to verify the theoretical conclusion.

## Key Results

| Task | Metric | Result |
|---|---|---|
| Classification | Best `k` | 1 |
| Classification | Mean cross-validation F1, scaled kNN | 0.9283 |
| Classification | Test F1, scaled kNN | 0.95 |
| Classification | Test F1, randomized dummy baseline | 0.12 |
| Regression | RMSE, original features | 0.36 |
| Regression | RMSE, scaled features | 0.36 |
| Regression | R² | 0.43 |
| Data obfuscation | Maximum prediction difference | Approximately 5.23 × 10⁻⁸ |
| Data obfuscation | Prediction equivalence at absolute tolerance 10⁻⁶ | True |

The classification model substantially outperformed the randomized baseline in the evaluated test set.

Scaling improved the performance of distance-based classification, while ordinary linear regression retained essentially the same predictive performance after scaling.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Linear Algebra
- Machine Learning

## Repository Structure

```text
Sure-Tomorrow-Insurance-ML/
├── notebook.ipynb
└── README.md
```

## How to Explore the Project

1. Open [notebook.ipynb](notebook.ipynb) to review the analysis, code, visualizations, and results.
2. To execute the notebook locally, install the required Python libraries.
3. Obtain the dataset from its authorized source.
4. Update the dataset path in the notebook to match your local environment.
5. Run the cells in order.

The original notebook references a dataset path provided by the training environment. The dataset is not included in this public repository.

## Limitations

- The classification target is imbalanced.
- The dataset does not include unique customer identifiers, limiting the interpretation of duplicate records.
- The final classifier was evaluated using one held-out test split.
- Regression showed moderate explanatory power, with an R² of approximately 0.43.
- The obfuscation approach is reversible and **does not constitute secure encryption**.
- Additional validation would be necessary before deploying these models in a real insurance environment.

## Conclusions

This project demonstrates that data preparation and feature scaling can strongly influence distance-based machine learning algorithms.

Scaled kNN achieved strong classification performance compared with the randomized baseline. In contrast, linear regression maintained essentially unchanged predictive performance after feature scaling.

The linear algebra component also demonstrated that an invertible transformation of personal features can preserve ordinary least squares predictions, provided that the corresponding model coefficients are transformed consistently.

Overall, the project combines exploratory data analysis, supervised machine learning, mathematical reasoning, and practical evaluation techniques to address insurance-related analytical tasks.

---

**Project Status:** Completed and academically approved.

**Author:** [ElamDay](https://github.com/ElamDay)
