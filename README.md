# Subgroup-Disaggregated-Faithfulness

Disaggregated faithfulness evaluation for SHAP and LIME explanations — exposing demographic disparities hidden by aggregate XAI metrics on the Adult Income Dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Runs%20on-Google%20Colab-F9AB00.svg)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-f7931e.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-1c7c54.svg)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-LimeTabular-ff6600.svg)](https://github.com/marcotcr/lime)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c.svg)](https://matplotlib.org/)
[![UCI Dataset](https://img.shields.io/badge/Dataset-UCI%20Adult%20Income-4b5563.svg)](https://archive.ics.uci.edu/dataset/2/adult)

**XAI Fairness Research — June 2026**

Standard explainability metrics evaluate explanation quality by averaging scores across the entire dataset. The problem: that average can look acceptable globally while completely failing specific demographic groups — and nobody notices. This project quantifies exactly that failure by introducing three new **disaggregated faithfulness metrics** that break down explanation quality by subgroup (sex, race, age), revealing which groups are silently underserved by SHAP and LIME.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Research Contribution](#2-research-contribution)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Faithfulness Metrics & Formulations](#4-faithfulness-metrics--formulations)
5. [Dataset](#5-dataset)
6. [Model](#6-model)
7. [Subgroup Definitions](#7-subgroup-definitions)
8. [Notebooks Overview](#8-notebooks-overview)
9. [Artifacts Produced](#9-artifacts-produced)
10. [Installation & Setup](#10-installation--setup)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Project Structure](#12-project-structure)
13. [References](#13-references)

---

## 1. Background & Motivation

### The Aggregation Problem in XAI

Post-hoc explanation methods such as SHAP and LIME are routinely evaluated using global faithfulness scores averaged across a dataset. Two widely used faithfulness metrics are:

- **PGI (Prediction Gap on Important features)** — How much does the prediction drop when you mask the features the explanation says are most important? A faithful explanation should cause a large drop.
- **PGU (Prediction Gap on Unimportant features)** — How much does the prediction drop when you mask the features the explanation says are least important? A faithful explanation should cause a small drop.

Global averages of PGI and PGU can look healthy while concealing that a specific demographic group receives systematically worse explanations. A model certified as "well-explained" globally may provide explanations that are unreliable specifically for women, Black individuals, or younger people — and this failure is invisible without disaggregated reporting.

### Why This Matters

| Risk | Consequence |
|---|---|
| Globally adequate, locally failing explanations | Auditors miss subgroup-level XAI failures during model review |
| No disaggregated reporting standard | Certification bodies accept aggregate scores as sufficient |
| Selection bias in evaluation samples | Minority subgroups are statistically underrepresented in benchmark sets |
| Disparate reliance on flawed explanations | Decision-makers over-trust explanations for groups where they are least faithful |

This research directly addresses the gap between global XAI evaluation and the fairness requirements of real-world deployment.

### Literature Basis

| Technique | Reference |
|---|---|
| SHAP (SHapley Additive exPlanations) | Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS |
| LIME (Local Interpretable Model-Agnostic Explanations) | Ribeiro et al. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* KDD |
| Prediction Gap faithfulness metrics (PGI / PGU) | Dasgupta et al. (2022). *Framework for evaluating faithfulness of local explanations.* ICML |
| Disaggregated evaluation in ML fairness | Barocas et al. (2019). *Fairness and Machine Learning.* fairmlbook.org |
| Subgroup disparity reporting | Biderman & Scheirer (2020). *Pitfalls in machine learning research.* ICLR |
| Adult Income Dataset bias analysis | Bao et al. (2021). *How do fairness definitions fare?* AIES |

---

## 2. Research Contribution

### Three New Disaggregated Faithfulness Metrics

**Worst Subgroup Faithfulness (WSF)**

Identifies the demographic subgroup that receives the worst explanation quality. Unlike global metrics, WSF surfaces the most disadvantaged group explicitly:

```
WSF(metric, attribute) = min_{g ∈ groups(attribute)} score(metric, g)
```

A low WSF flags that at least one group is being "explanation-abandoned" regardless of how well the global score looks.

**Faithfulness Disparity (FD)**

Measures the gap between the best and worst subgroup. High disparity indicates unequal explanation quality across demographics:

```
FD(metric, attribute) = max_{g} score(metric, g) − min_{g} score(metric, g)
```

FD = 0 would mean perfect equality of explanation quality. Values above 0.1 (on a 0–1 scale) are considered substantively concerning.

**Explanation Desert Index (EDI)**

Identifies groups that are most "abandoned" relative to the global average — i.e., groups for which explanation quality falls furthest below the dataset-wide baseline:

```
EDI(g, metric) = global_score(metric) − score(metric, g)

EDI(attribute) = max_{g ∈ groups(attribute)} EDI(g, metric)
```

A group with EDI > 0 is below average; the group with the highest EDI is the most underserved. Comparing EDI across SHAP and LIME reveals whether one method systematically disadvantages certain demographics more than the other.

---

## 3. Pipeline Architecture

The project runs as four sequential notebooks, designed to execute in order on Google Colab with Google Drive as shared storage.

```
START
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  NOTEBOOK 01 — PREPROCESSING & MODEL TRAINING               │
│  Notebook: 01_preprocessing.ipynb                           │
│  • Load Adult Income Dataset from UCI API                    │
│  • Clean: strip whitespace, replace '?', drop duplicates/NaN│
│  • Save sensitive columns before encoding (sex, race, age)  │
│  • Label-encode categoricals, binary-encode target          │
│  • Train/test split (80/20, stratified by income)           │
│  • Train Random Forest (100 trees, accuracy = 0.852)        │
│  • Save: adult_test.csv, adult_train.csv,                   │
│           model_rf.pkl, feature_names.pkl                   │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  NOTEBOOK 02a — SHAP EXPLANATIONS                           │
│  Notebook: 02a_shap_explanations.ipynb                      │
│  • Sample 500 test instances (stratified by income class)   │
│  • Compute exact SHAP values via TreeExplainer              │
│  • Extract positive-class slice → shape (500, 14)           │
│  • Save: sample_indices.npy, X_sample.csv,                  │
│           y_sample.csv, sens_sample.csv, shap_values.npy    │
│  • Visualise: beeswarm, bar chart, waterfall example        │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  NOTEBOOK 02b — LIME EXPLANATIONS          (~15–30 min)     │
│  Notebook: 02b_lime_explanations.ipynb                      │
│  • Reload same 500 instances via sample_indices.npy         │
│  • Fit LimeTabularExplainer on training distribution        │
│  • Generate per-instance local linear models                │
│  • Extract feature weights by index → shape (500, 14)       │
│  • Save: lime_values.npy                                    │
│  • Visualise: importance chart, SHAP vs LIME ranking        │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  NOTEBOOK 03 — FAITHFULNESS METRICS (baseline + novel)      │
│  Notebook: 03_metrics.ipynb                                 │
│  • Load all artifacts from 02a/02b                          │
│  • Compute PGI and PGU (global baseline)                    │
│  • Disaggregate PGI/PGU by sex, race, age_group            │
│  • Compute WSF, FD, EDI for SHAP and LIME                   │
│  • Compare SHAP vs LIME across subgroups                    │
│  • Produce final tables and figures                         │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Notebook 01  ──►  adult_test.csv       ──►  Notebooks 02a, 02b, 03
             ──►  adult_train.csv      ──►  Notebook 02b (LIME background)
             ──►  model_rf.pkl         ──►  Notebooks 02a, 02b, 03
             ──►  feature_names.pkl    ──►  Notebooks 02a, 02b, 03

Notebook 02a ──►  sample_indices.npy   ──►  Notebook 02b (shared sample)
             ──►  X_sample.csv         ──►  Notebook 03
             ──►  y_sample.csv         ──►  Notebook 03
             ──►  sens_sample.csv      ──►  Notebook 03
             ──►  shap_values.npy      ──►  Notebooks 02b (comparison), 03

Notebook 02b ──►  lime_values.npy      ──►  Notebook 03
```

---

## 4. Faithfulness Metrics & Formulations

All metrics are computed in Notebook 03 on the shared 500-instance sample.

### 4.1 Prediction Gap on Important Features (PGI)

PGI measures how much the model's prediction changes when the features deemed **most important** by the explanation are masked (replaced with a baseline, e.g. the feature mean). A faithful explanation should assign high weights to features that genuinely drive the prediction — masking them should cause a large prediction drop:

```
PGI(x, φ) = f(x) − f(x̃_top-k)

where x̃_top-k is x with the top-k features (by |φ|) replaced by their mean values.
```

Higher PGI → more faithful (masking important features actually hurts predictions).

### 4.2 Prediction Gap on Unimportant Features (PGU)

PGU measures the prediction change when the features deemed **least important** are masked. A faithful explanation should assign low weights to features that don't affect the prediction — masking them should cause a small prediction drop:

```
PGU(x, φ) = f(x) − f(x̃_bottom-k)

where x̃_bottom-k is x with the bottom-k features (by |φ|) replaced by their mean values.
```

Lower PGU → more faithful (masking unimportant features barely changes predictions).

Both PGI and PGU use `k = 5` (top/bottom 5 of 14 features) by default.

### 4.3 Subgroup Disaggregation

For each metric `m ∈ {PGI, PGU}`, explainer `e ∈ {SHAP, LIME}`, and sensitive attribute `a ∈ {sex, race, age_group}`:

```
score(m, e, g)     = mean of m across all instances in subgroup g

WSF(m, e, a)       = min_{g ∈ groups(a)}  score(m, e, g)

FD(m, e, a)        = max_{g ∈ groups(a)}  score(m, e, g)
                   − min_{g ∈ groups(a)}  score(m, e, g)

EDI(g, m, e)       = score(m, e, all)  −  score(m, e, g)

EDI(m, e, a)       = max_{g ∈ groups(a)}  EDI(g, m, e)
```

### 4.4 Top-k Selection

Feature importance ranking uses absolute attribution values `|φᵢ|`, consistent between SHAP and LIME. For SHAP, `φ` is the exact Shapley value for the positive class. For LIME, `φ` is the local linear weight extracted via `local_exp[1]` (class 1 coefficient).

---

## 5. Dataset

**UCI Adult Income Dataset** (also known as the Census Income Dataset)

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository (ID=2)](https://archive.ics.uci.edu/dataset/2/adult) |
| Original size | 48,842 instances |
| After cleaning | 45,194 instances |
| Features | 14 |
| Target | Binary: income > \$50K/year |
| Class distribution | 75.2% ≤50K / 24.8% >50K |
| Missing values | Encoded as `'?'` in workclass, occupation, native-country |

### Feature Schema

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age in years |
| `workclass` | categorical | Employment type (Private, Self-emp, Gov, etc.) |
| `fnlwgt` | int | Census sampling weight |
| `education` | categorical | Highest education level |
| `education-num` | int | Education level as integer |
| `marital-status` | categorical | Marital status |
| `occupation` | categorical | Job category |
| `relationship` | categorical | Household relationship |
| `race` | categorical | Race (5 categories) |
| `sex` | categorical | Sex (Male / Female) |
| `capital-gain` | int | Capital gain income |
| `capital-loss` | int | Capital loss |
| `hours-per-week` | int | Weekly work hours |
| `native-country` | categorical | Country of origin (41 categories) |

### Cleaning Steps

1. Strip whitespace from all string columns (UCI Adult has trailing spaces)
2. Replace `'?'` with `NaN`
3. Remove duplicate rows (29 removed)
4. Drop rows with any `NaN` (3,619 removed)
5. Reset index

### Encoding

- **Categorical features**: `sklearn.LabelEncoder` per column
- **Target**: `str.contains('>50K')` → 1; otherwise 0 (handles both `>50K` and `>50K.` variants)

---

## 6. Model

A **Random Forest classifier** trained on the 80% training split.

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `random_state` | 42 |
| `n_jobs` | -1 (all CPU cores) |
| **Test accuracy** | **0.852** |
| Test precision (>50K) | 0.74 |
| Test recall (>50K) | 0.62 |
| Test F1 (>50K) | 0.67 |

Random Forest is used because:
- It is a non-linear black-box model — exactly the type for which post-hoc explanations (SHAP, LIME) are necessary
- `shap.TreeExplainer` computes **exact** Shapley values for tree ensembles in polynomial time
- Strong tabular performance without heavy hyperparameter tuning

---

## 7. Subgroup Definitions

Sensitive attributes are saved **before** label encoding (the raw string values), so subgroup labels remain human-readable throughout all analysis notebooks.

### Sex

| Group | Count in test set |
|---|---|
| Male | ~6,100 |
| Female | ~2,940 |

### Race

| Group | Count in test set |
|---|---|
| White | ~7,780 |
| Black | ~845 |
| Asian-Pac-Islander | ~260 |
| Amer-Indian-Eskimo | ~87 |
| Other | ~70 |

### Age Group

| Group | Bin | Count in test set |
|---|---|---|
| `<25` | [0, 25) | ~1,690 |
| `25-35` | [25, 35) | ~2,410 |
| `35-50` | [35, 50) | ~3,200 |
| `50-65` | [50, 65) | ~1,470 |
| `65+` | [65, 100] | ~270 |

> **Note on minority subgroups**: Amer-Indian-Eskimo and Other race categories, and the 65+ age group, have small sample counts in the 500-instance sample. Metrics for these groups should be interpreted with appropriate uncertainty.

---

## 8. Notebooks Overview

All notebooks are designed to run on **Google Colab** with outputs persisted to **Google Drive** at `MyDrive/XAIP/data/`.

### Notebook 01 — Preprocessing & Model Training

**Path**: [Notebooks/preprocessing_and_model/01_preprocessing.ipynb](Notebooks/preprocessing_and_model/01_preprocessing.ipynb)

**Owner**: Teammate

**What it does**:
- Downloads the UCI Adult Income dataset via `ucimlrepo`
- Cleans and encodes the data
- Saves sensitive columns (sex, race, age_group) before encoding
- Trains a Random Forest and evaluates on the test set
- Saves all four artifacts to Drive

---

### Notebook 02a — SHAP Explanations

**Path**: [Notebooks/explainers/02a_shap_explanations.ipynb](Notebooks/explainers/02a_shap_explanations.ipynb)

**Must run before**: 02b, 03

**Steps**:
1. Load `adult_test.csv`, `model_rf.pkl`, `feature_names.pkl`
2. Sample 500 instances stratified by income class (`random_state=42`)
3. Save `sample_indices.npy` for 02b to reuse
4. Save `X_sample.csv`, `y_sample.csv`, `sens_sample.csv`
5. Run `shap.TreeExplainer` with `check_additivity=False`
6. Extract positive-class slice → `shap_values.npy` shape `(500, 14)`
7. Visualise: beeswarm summary plot, mean |SHAP| bar chart, waterfall for one instance

**Runtime**: ~2–5 minutes on Colab

---

### Notebook 02b — LIME Explanations

**Path**: [Notebooks/explainers/02b_lime_explanations.ipynb](Notebooks/explainers/02b_lime_explanations.ipynb)

**Must run after**: 02a (depends on `sample_indices.npy`)

**Steps**:
1. Load all artifacts from 01 and `sample_indices.npy` from 02a
2. Reconstruct the exact same 500-instance sample
3. Fit `LimeTabularExplainer` on the full training set as background distribution
4. Loop over 500 instances with `tqdm`, calling `explain_instance` per row
5. Extract weights via `local_exp[1]` (feature index → weight) → `lime_values.npy` shape `(500, 14)`
6. Visualise: mean |LIME weight| bar chart, one example explanation, SHAP vs LIME ranking comparison

**Runtime**: ~15–30 minutes on Colab

---

### Notebook 03 — Faithfulness Metrics

**Path**: [Notebooks/metrics/03_metrics.ipynb](Notebooks/metrics/03_metrics.ipynb)

**Must run after**: 02a, 02b

**Steps**:
1. Load `X_sample.csv`, `y_sample.csv`, `sens_sample.csv`, `shap_values.npy`, `lime_values.npy`
2. Compute global PGI and PGU for SHAP and LIME (baseline reference)
3. Disaggregate PGI and PGU by sex, race, and age_group
4. Compute WSF, FD, and EDI for each explainer and attribute combination
5. Compare SHAP vs LIME: which method is more equitable across subgroups?
6. Produce result tables and figures

---

## 9. Artifacts Produced

All files are saved to `MyDrive/XAIP/data/` on Google Drive.

| File | Shape / Type | Produced by | Consumed by |
|---|---|---|---|
| `adult_train.csv` | (36155, 15) | 01 | 02b |
| `adult_test.csv` | (9039, 18) | 01 | 02a, 02b, 03 |
| `model_rf.pkl` | RandomForest | 01 | 02a, 02b, 03 |
| `feature_names.pkl` | list[str] × 14 | 01 | 02a, 02b, 03 |
| `sample_indices.npy` | (500,) int | 02a | 02b |
| `X_sample.csv` | (500, 14) float | 02a | 03 |
| `y_sample.csv` | (500,) int | 02a | 03 |
| `sens_sample.csv` | (500, 3) str | 02a | 03 |
| `shap_values.npy` | (500, 14) float | 02a | 02b, 03 |
| `lime_values.npy` | (500, 14) float | 02b | 03 |

---

## 10. Installation & Setup

The notebooks run on **Google Colab**. All required packages are installed inline at the start of each notebook.

### Packages installed per notebook

| Notebook | Additional install |
|---|---|
| 01 | `ucimlrepo` |
| 02a | `shap` |
| 02b | `lime` |
| 03 | *(all standard: numpy, pandas, matplotlib, scikit-learn)* |

### Local development (optional)

If you want to run notebooks locally:

```bash
# Clone the repository
git clone https://github.com/your-username/Subgroup-Disaggregated-Faithfulness.git
cd Subgroup-Disaggregated-Faithfulness

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install ucimlrepo shap lime scikit-learn pandas numpy matplotlib tqdm jupyter
```

Remove or comment out the `drive.mount` and `google.colab` cells when running locally, and set `BASE` to a local directory.

---

## 11. Running the Pipeline

### On Google Colab (recommended)

1. Upload notebooks to your Google Drive or open directly from GitHub
2. Run in order: **01 → 02a → 02b → 03**
3. Each notebook mounts Drive at `/content/drive` and reads/writes to `MyDrive/XAIP/data/`

### Execution order matters

```
01_preprocessing.ipynb          ← must run first (creates model + test set)
    │
    └──► 02a_shap_explanations.ipynb    ← must run before 02b and 03
              │
              └──► 02b_lime_explanations.ipynb   ← must run before 03
                        │
                        └──► 03_metrics.ipynb    ← final output
```

### Smoke testing 02b (LIME is slow)

LIME generates 1,000 perturbations per instance across 500 instances. To test the pipeline quickly, reduce the loop range in 02b:

```python
# Change this in the loop cell of 02b:
for i in tqdm(range(20), desc='LIME explanations'):   # 20 instead of 500
```

---

## 12. Project Structure

```
Subgroup-Disaggregated-Faithfulness/
│
├── Notebooks/
│   ├── preprocessing_and_model/
│   │   └── 01_preprocessing.ipynb      # Data cleaning, encoding, RF training
│   │
│   ├── explainers/
│   │   ├── 02a_shap_explanations.ipynb # SHAP values (500, 14) + shared sample
│   │   └── 02b_lime_explanations.ipynb # LIME weights (500, 14)
│   │
│   └── metrics/
│       └── 03_metrics.ipynb            # PGI, PGU, WSF, FD, EDI
│
├── data/                               # Local copies of Drive artifacts
│   ├── adult_train.csv
│   ├── adult_test.csv
│   ├── model_rf.pkl
│   └── feature_names.pkl
│
├── LICENSE
└── README.md
```

---

## 13. References

| Reference | Role in this project |
|---|---|
| Lundberg, S. & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS. | SHAP method and TreeExplainer |
| Ribeiro, M. T. et al. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* KDD. | LIME method |
| Dasgupta, S. et al. (2022). *Framework for evaluating faithfulness of local explanations.* ICML. | PGI and PGU metric definitions |
| Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning.* fairmlbook.org. | Disaggregated evaluation framework |
| Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository.* University of California, Irvine. | Adult Income Dataset |
| Bao, M. et al. (2021). *How do fairness definitions fare? Examining public attitudes towards algorithmic definitions of fairness.* AIES. | Fairness analysis on Adult dataset |
| Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32. | Random Forest classifier |

---

## Team

**Amirhesam Torkashvand** — XAI metrics, SHAP explanations, LIME explanations

**Jacopo** — Data preprocessing and model training 

**Stefano** — PGI, PGU, F_global computation and first plots

---

*Built with [scikit-learn](https://scikit-learn.org/) · [SHAP](https://shap.readthedocs.io/) · [LIME](https://github.com/marcotcr/lime) · [Google Colab](https://colab.research.google.com/)*
