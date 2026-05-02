# ML4CAD — Machine Learning for Coronary Artery Disease Prediction

**Author:** [Aldo Buzi](https://github.com/AldoBuzi)

A comprehensive machine learning pipeline for predicting **7-year mortality** in patients with Coronary Artery Disease (CAD), using clinical, biochemical, and thyroid-function data. The project explores:

- Multiple feature subsets (18 / 23 / 27 / 32 features)
- 7 classifiers + hyperparameter tuning via `RandomizedSearchCV`
- Oversampling strategies (SMOTE, Borderline-SMOTE, SVM-SMOTE) to handle class imbalance
- Ensemble learning (best combinations of 2+ models)
- Feature clustering and ablation studies
- Kaplan–Meier survival analysis for risk stratification

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset & Feature Subsets](#dataset--feature-subsets)
- [Pipeline](#pipeline)
  - [1. Data Processing](#1-data-processing)
  - [2. Classification](#2-classification)
  - [3. Sampling & Imbalance Handling](#3-sampling--imbalance-handling)
  - [4. Ensemble Search](#4-ensemble-search)
  - [5. Feature Clustering & Ablation](#5-feature-clustering--ablation)
  - [6. Survival Analysis](#6-survival-analysis)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)

---

## Project Overview

**Goal:** Predict whether a CAD patient will die from cardiovascular causes within 7 years of a clinical visit (target: `Survive7Y`).

**Key challenge:** Severe class imbalance — only ≈13% of patients experience the event.

**Approach:** A multi-stage pipeline that:
1. Cleans and filters raw clinical data
2. Trains and tunes multiple classifiers on different feature subsets
3. Applies oversampling to handle imbalance
4. Searches for optimal ensemble combinations
5. Validates via feature ablation and survival curve analysis

---

## Dataset & Feature Subsets

Data comes from **three raw sources**:
- `raw_data.xlsx` — main clinical dataset
- `data_prelievo.xlsx` — blood draw dates
- `creatina_more_columns.xlsx` — additional creatinine measurements

After cleaning (removing non-CVD deaths, irrelevant features), four feature subsets are created:

| Subset | # Features | Description |
|--------|-----------|-------------|
| **18 features** | 18 | Core clinical features (age, vessels, history, etc.) — **excludes** both lipid panel and thyroid panel |
| **23 features** | 23 | 18 features + **creatinine & lipid panel** (Total cholesterol, HDL, LDL, Triglycerides, Creatinine) |
| **27 features** | 27 | 18 features + **thyroid panel** (TSH, fT3, fT4, Euthyroid, SCH, SCT, Low T3, Hypothyroidism, Hyperthyroidism) |
| **32 features** | 32 | All available features (18 + lipids + thyroid) |

**Missing data handling:**
- For 23- and 32-feature sets: **mean imputation** vs. **dropping NAs** — both strategies are tested separately.

---

## Pipeline

### 1. Data Processing
**Notebook:** `1_data_process.ipynb`

- Loads raw Excel files (`raw_data.xlsx`, `data_prelievo.xlsx`, `creatina_more_columns.xlsx`)
- Filters to **CVD death patients only** (removes non-CVD deaths)
- Removes irrelevant features, creates target `Survive7Y`
- Splits data into **feature subsets** (18, 23, 27, 32 features)
- Saves train/validation/test splits for each subset

### 2. Classification
**Notebook:** `2_classifiers.ipynb`  
**Module:** `train.py`

For each feature subset:
- Builds a `sklearn` `Pipeline` with:
  - `ColumnTransformer` + `StandardScaler` for numerical features
  - The chosen classification model
- Runs **`RandomizedSearchCV`** (2-fold CV, 5000 iterations) over a wide hyperparameter space
- Evaluates on **validation set** (classification report, AUROC, confusion matrix)
- Saves the best model as a `.joblib` file

**Hyperparameter search spaces** are defined in `hyperparameters.py` using SciPy distributions for continuous parameters.

### 3. Sampling & Imbalance Handling
**Notebook:** `3.1_data_sampling.ipynb`

Applies **oversampling** to the training set before re-training all classifiers:

| Oversampler | Description |
|-------------|-------------|
| **SMOTE** | Standard Synthetic Minority Oversampling Technique |
| **Borderline-SMOTE** | SMOTE focused on borderline minority samples |
| **SVM-SMOTE** | SMOTE using SVM to identify borderline region |

Each classifier is trained 5 times with different resampled data, and the **mean macro F1** is logged. Results are saved as:
- `*_sampling.txt` — full results with oversampling
- `*_only_oversampling.txt` — oversampling only (no undersampling)
- Best models saved as `*_random_smote_*.joblib`, `*_random_bordersmote_*.joblib`, `*_random_svmsmote_*.joblib`

### 4. Ensemble Search
**Notebook:** `3.2_find_best_ensemble.ipynb` / `3_ensemble_ablations.ipynb`  
**Module:** `ensemble.py`

After individual models are trained on resampled data, the best ones are combined into **ensembles**:

1. Generates all **unique combinations** of 2+ models without duplicates
2. For each combination, predicts via **simple averaging** of probabilities
3. Evaluates on: **AUROC**, **macro F1**, **Brier score**
4. Ranks the top 5–10 ensembles by **Brier score** (calibration quality) and macro F1
5. Produces **calibration plots** for the best ensembles

This approach finds that ensembles of 3–4 diverse models consistently outperform single classifiers.

### 5. Feature Clustering & Ablation
**Notebook:** `4_feature_cluster.ipynb`

To understand feature importance:
- Computes **correlation matrix** (Pearson) among features
- Applies **hierarchical clustering** to group correlated features
- Performs **univariate and multivariate ablation**: removes feature clusters and measures performance drop
- Identifies the most predictive feature groups

Output files:
- `feat_cluster_hier.df` — cluster assignments
- `cluster.tiff` — cluster visualization
- `extra_ablation_uni_test.csv` — univariate ablation results
- `multivariate_ablation_hier_nomeds.csv` — multivariate ablation results

### 6. Survival Analysis
**Notebook:** `5_survival_analysis.ipynb`

Generates **Kaplan–Meier survival curves** for the best-trained models:
- Splits patients into **high-risk** and **low-risk** groups based on model predictions
- Compares survival trajectories between the two groups
- Also plots KM curves stratified by individual clinical features (age, gender, smoking, diabetes, hypertension, angiography results, thyroid status, etc.)

Figures are stored in `figures/kaplan_meier_comparision/`.

---

## Models Used

| Model | Abbreviation | Hyperparameter Search Space |
|-------|-------------|---------------------------|
| **Logistic Regression** | `lr` | Penalty type, solver, regularization strength (`C`), `max_iter` |
| **Support Vector Classifier** | `svc` | Kernel type, `C`, `gamma`, `degree` |
| **k-Nearest Neighbors** | `knn` | `n_neighbors`, weight function, algorithm, `leaf_size` |
| **Random Forest** | `rf` | `n_estimators`, criterion, `min_samples_split/leaf`, `max_features` |
| **AdaBoost** | `adaboost` | `n_estimators`, `learning_rate` |
| **Neural Network (MLP)** | `nn` | `hidden_layer_sizes`, solver, `learning_rate`, `alpha`, `max_iter` |
| **Gradient Boosting** | `gb` | `learning_rate`, `n_estimators`, `max_depth`, `subsample` |
| **XGBoost** | `xgb` | Booster type, `eta`, `gamma`, `max_depth`, regularization, `scale_pos_weight` |

---

## Project Structure

```
.
├── 1_data_process.ipynb          # Data cleaning & feature subset creation
├── 2_classifiers.ipynb           # Base classifier training & evaluation
├── 3.1_data_sampling.ipynb       # Oversampling with SMOTE variants
├── 3.2_find_best_ensemble.ipynb  # Search for optimal ensemble combinations
├── 3_ensemble_ablations.ipynb    # Ensemble calibration & analysis
├── 4_feature_cluster.ipynb       # Feature correlation, clustering & ablation
├── 5_survival_analysis.ipynb     # Kaplan-Meier survival analysis
│
├── train.py                      # Training pipeline (RandomizedSearchCV + evaluation)
├── ensemble.py                   # Ensemble building, averaging, best-combination search
├── utils.py                      # Preprocessing (StandardScaler), resampling utilities, DebuggablePipeLine
├── hyperparameters.py            # Hyperparameter search spaces for all models
│
├── data/
│   ├── raw/                      # Raw Excel data files
│   ├── 18features/               # Train/valid/test splits (18 features)
│   ├── 23features/               # 23 features (with lipids, no thyroid)
│   ├── 27features/               # 27 features (with thyroid, no lipids)
│   └── 32features/               # 32 features (all)
│
├── models/                       # Serialized .joblib model files
│   ├── 18features/
│   ├── 23features/
│   ├── 27features/
│   └── 32features/
│
├── models_output/                # Training logs & evaluation reports (.txt)
│   ├── 18features/
│   ├── 23features/
│   ├── 27features/
│   └── 32features/
│
├── figures/                      # Generated plots & visualizations
│   ├── 18features/               # Calibration plots
│   ├── 27features/               # Calibration + cluster plots
│   └── kaplan_meier_comparision/ # Kaplan-Meier survival curves
│
└── README.md
```

---

## How to Run

### Prerequisites

```
Python 3.8+
pandas, numpy, scipy, matplotlib, scikit-learn
imbalanced-learn
xgboost
joblib
openpyxl (for Excel files)
lifelines (for survival analysis)
```

Install with:

```bash
pip install pandas numpy scipy matplotlib scikit-learn imbalanced-learn xgboost joblib openpyxl lifelines
```

### Execution order (recommended)

```bash
# 1. Data preprocessing
jupyter notebook 1_data_process.ipynb

# 2. Train base classifiers
jupyter notebook 2_classifiers.ipynb

# 3. Retrain with oversampling
jupyter notebook 3.1_data_sampling.ipynb

# 4. Find best ensembles
jupyter notebook 3.2_find_best_ensemble.ipynb

# 5. Ensemble calibration & ablation
jupyter notebook 3_ensemble_ablations.ipynb

# 6. Feature clustering analysis
jupyter notebook 4_feature_cluster.ipynb

# 7. Survival analysis
jupyter notebook 5_survival_analysis.ipynb
```

Each notebook is self-contained and reads/writes from the `data/`, `models/`, `models_output/`, and `figures/` directories.

---

## Results Summary

- **Best single classifier** varies by feature subset and sampling strategy; XGBoost and Gradient Boosting consistently rank among the top.
- **Ensembles of 3–4 diverse models** outperform any single classifier, particularly when combining tree-based models (RF, GB, XGB) with a linear model (LR) or neural network.
- **Oversampling with SVM-SMOTE** tends to produce the best-calibrated models (lowest Brier score).
- **Thyroid panel features** (27-feature set) contribute meaningful predictive power in ablation studies.
- **Kaplan–Meier curves** confirm that the model's high-risk group has significantly lower survival probability, validating the clinical utility of the predictions.
