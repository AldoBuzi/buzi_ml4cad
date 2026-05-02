# Comparative Study of Machine Learning Indicators for Cardiac Mortality Prediction in Ischemic Heart Disease Patients

> Bachelor's thesis — **University of Pisa, Department of Computer Science**
>
> **Author:** Donaldo (Aldo) Buzi
>
> **Advisors:** Prof. Giuseppe Prencipe, Prof.ssa Alina Sîrbu
>
> **Academic Year:** 2023/2024

A comparative machine learning study for predicting **7-year cardiovascular mortality** in patients with **Ischemic Heart Disease (IHD)**. The work compares two risk indicators:

- **17-parameter indicator** — based purely on cardiovascular features (adapted from [2])
- **26-parameter indicator** — extends the above with thyroid function parameters

An **ensemble model** is built by combining the best classifiers. The study also includes **Kaplan-Meier survival analysis** and **Cox regression** for variable importance assessment.

---

## Table of Contents

- [Abstract](#abstract)
- [Contributions](#contributions)
- [Dataset & Feature Sets](#dataset--feature-sets)
- [Pipeline](#pipeline)
- [Models](#models)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [References](#references)

---

## Abstract

Cardiovascular diseases, particularly Ischemic Heart Disease (IHD), remain a leading cause of death worldwide. This study evaluates the ability of machine learning to identify high-risk IHD patients and compares two risk indicators:

1. A **purely cardiovascular indicator** based on 17 clinical parameters
2. A **thyroid-enhanced indicator** that adds thyroid function parameters (26 total)

The goal is to determine whether thyroid data improves prediction accuracy. The best-performing models are combined into an ensemble, and survival analysis validates the clinical stratification.

---

## Contributions

- **Data processing** — cleaning, filtering CVD deaths, handling missing values
- **ML pipeline** — training and evaluating 8 classifiers across multiple feature sets
- **Ensemble construction** — exhaustive search over all model combinations to find the best ensemble by Brier score and macro F1
- **Ablation tests** — univariate and multivariate feature importance via a custom *importance* metric
- **Survival analysis** — Kaplan-Meier estimation and Cox Proportional Hazards model for risk stratification and variable analysis

---

## Dataset & Feature Sets

Three raw data sources are used:
- `raw_data.xlsx` — main clinical dataset
- `data_prelievo.xlsx` — blood draw dates
- `creatina_more_columns.xlsx` — additional creatinine measurements

After filtering out non-CVD deaths and irrelevant features, four feature sets are derived.

### Main indicators (thesis focus)

| Indicator | Parameters | Features (incl. target) | Description |
|-----------|-----------|------------------------|-------------|
| **Cardiovascular** | 17 | 18 (17 + `Survive7Y`) | Original cardiovascular parameters from [2], minus creatinine (removed due to missing values) |
| **Thyroid-enhanced** | 26 | 27 (26 + `Survive7Y`) | Cardiovascular + thyroid panel (TSH, fT3, fT4, Euthyroid, SCH, SCT, Low T3, Hypothyroidism, Hyperthyroidism) |

### Secondary variants (exploratory)

| Variant | Features (incl. target) | Description |
|---------|------------------------|-------------|
| **22-parameter** | 23 | Cardiovascular + creatinine & lipid panel (Total cholesterol, HDL, LDL, Triglycerides, Creatinine). Excludes thyroid. |
| **31-parameter** | 32 | All available features (cardiovascular + lipids + thyroid). |

> **Note on nomenclature:** The code and directories use the full feature count including the target (18, 23, 27, 32 features). The thesis refers to the parameter count excluding the target (17, 22, 26, 31 parameters).

### Missing data handling

For the 23- and 32-feature sets (which contain creatinine and lipid columns with missing values), two strategies are tested separately:
- **Mean imputation**
- **Drop rows with NAs**

---

## Pipeline

### 1. Data Processing
**Notebook:** `1_data_process.ipynb`

- Loads raw Excel files, converts dates
- Removes non-CVD death patients
- Removes irrelevant features, creates `Survive7Y` target
- Splits into train/validation/test for each feature set

### 2. Training
**Notebook:** `2_classifiers.ipynb`  
**Module:** `train.py`

Each classifier is trained via a `Pipeline` (`StandardScaler` → model) with `RandomizedSearchCV` (2-fold CV, 5000 iterations). Hyperparameter spaces are defined in `hyperparameters.py` using SciPy statistical distributions for continuous parameters.

### 3. Data Sampling
**Notebook:** `3.1_data_sampling.ipynb`

To address severe class imbalance (~13% events), three oversampling techniques are applied:

| Oversampler | Strategy |
|-------------|----------|
| **SMOTE** | Standard synthetic minority oversampling |
| **Borderline-SMOTE** | Focuses on borderline minority samples |
| **SVM-SMOTE** | Uses SVM to identify the borderline region |

Each resampled model is trained 5 times and the mean macro F1 is logged.

### 4. Calibration
**Notebook:** `3_ensemble_ablations.ipynb`

Top models are calibrated via `CalibratedClassifierCV`. Calibration curves are plotted for the best ensembles.

### 5. Ensemble Search
**Notebook:** `3.2_find_best_ensemble.ipynb`  
**Module:** `ensemble.py`

All unique combinations of 2+ models are evaluated:
- Predictions are averaged (simple mean of probabilities)
- Ranked by **Brier score** (calibration quality) and **macro F1**
- The top 5–10 ensembles are saved and analyzed

### 6. Feature Clustering & Ablation
**Notebook:** `4_feature_cluster.ipynb`

- Computes Spearman correlation matrix
- Hierarchical clustering of correlated features
- Univariate and multivariate ablation tests using a custom *importance* metric
- Identifies the most predictive feature groups

### 7. Survival Analysis
**Notebook:** `5_survival_analysis.ipynb`

- **Kaplan-Meier estimator** — computes survival functions for all patients and for high-risk vs. low-risk groups stratified by the ML model
- **Cox Proportional Hazards model** — univariate and multivariate analysis to assess the statistical significance and impact of each variable
- Compares ML-based risk stratification with traditional statistical methods

---

## Models

| Model | Abbreviation |
|-------|-------------|
| Logistic Regression | `lr` |
| Support Vector Classifier | `svc` |
| K-Nearest Neighbors | `knn` |
| Random Forest | `rf` |
| Adaptive Boosting | `adaboost` |
| Multilayer Perceptron (Neural Network) | `nn` |
| Gradient Boosting | `gb` |
| XGBoost | `xgb` |

Each model is trained:
1. **Without sampling** — on the original imbalanced data
2. **With SMOTE** — synthetic oversampling
3. **With Borderline-SMOTE** — focused oversampling
4. **With SVM-SMOTE** — SVM-guided oversampling

---

## Project Structure

```
.
├── 1_data_process.ipynb           # Data cleaning & feature set creation
├── 2_classifiers.ipynb            # Base classifier training
├── 3.1_data_sampling.ipynb        # Oversampling with SMOTE variants
├── 3.2_find_best_ensemble.ipynb   # Optimal ensemble search
├── 3_ensemble_ablations.ipynb     # Calibration & ablation analysis
├── 4_feature_cluster.ipynb        # Feature correlation & clustering
├── 5_survival_analysis.ipynb      # Kaplan-Meier & Cox regression
│
├── train.py                       # Training pipeline (RandomizedSearchCV)
├── ensemble.py                    # Ensemble building & evaluation
├── utils.py                       # Preprocessing & sampler utilities
├── hyperparameters.py             # Hyperparameter search spaces
│
├── data/
│   ├── raw/                       # Raw Excel sources
│   ├── 18features/                # 17-parameter indicator (cardiovascular)
│   ├── 23features/                # 22-parameter variant (+lipids)
│   ├── 27features/                # 26-parameter indicator (+thyroid)
│   └── 32features/                # 31-parameter variant (all features)
│
├── models/                        # Serialized .joblib files
├── models_output/                 # Training logs (.txt)
├── figures/                       # Calibration & survival plots
└── README.md
```

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scipy matplotlib scikit-learn imbalanced-learn xgboost joblib openpyxl lifelines
```

### Execution order

```bash
jupyter notebook 1_data_process.ipynb
jupyter notebook 2_classifiers.ipynb
jupyter notebook 3.1_data_sampling.ipynb
jupyter notebook 3.2_find_best_ensemble.ipynb
jupyter notebook 3_ensemble_ablations.ipynb
jupyter notebook 4_feature_cluster.ipynb
jupyter notebook 5_survival_analysis.ipynb
```

---

## Results Summary

- **Single best model** varies by feature set, but tree-based models (GB, XGB, RF) consistently rank among the top.
- **Ensembles of 3–4 diverse models** outperform single classifiers, especially combinations of tree-based + linear models.
- **SVM-SMOTE oversampling** tends to produce the best-calibrated models (lowest Brier score).
- **Thyroid panel features** (26-parameter indicator) contribute meaningful predictive value in ablation studies.
- **Kaplan-Meier survival curves** confirm that the model's high-risk group has significantly lower survival probability, validating the clinical utility of the ML-based stratification.

---

## References

[2] *Original cardiovascular risk indicator paper* (cited in the thesis).

Full details on methodology, clinical background, and extended results are available in the thesis document.
