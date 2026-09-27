
## Musanze Cooperative Harvest and Dispatch Decision Lab

**Project type:** Scenario-based machine learning group assignment 
**Course:** SWE 3513 — Artificial Intelligence 
**Institution:** Institut d'Enseignement Supérieur de Ruhengeri (INES-Ruhengeri) 
**Group code:** `AI-G05` 
**Model version:** `1.0`

## 1. Project Overview

This project implements a reproducible command-line machine learning pipeline for the fictional **Musanze HarvestLink Cooperative**.

The pipeline supports three operational decisions:

1. **Regression** — estimate expected potato harvest weight (`actual_yield_kg`).
2. **Classification** — predict whether a consignment requires dispatch attention (`dispatch_attention`).
3. **Clustering** — group records with similar operating profiles using only input features.

## 2. Group Information

| Role | Student | Registration Number | Main Responsibility |

| Member 1 — Data & UX Lead | `[NAME]` | `[REG NO.]` | Schema validation, vectorization, data report, UI/UX coordination |
| Member 2 — Regression Engineer | `[KAWAYA GUEYLORD]` | `[25/27953]` | NumPy regression, gradient descent, scaling, loss curve, regression metrics |
| Member 3 — Classification Engineer | `[NAME]` | `[REG NO.]` | Classification split, model, metrics, confusion matrix, error-cost interpretation |
| Member 4 — Clustering & QA Engineer | `[NAME]` | `[REG NO.]` | Clustering, silhouette comparison, labels, cross-pipeline QA |
| Member 5 — Reproducibility & Release Lead | `[NAME / N/A]` | `[REG NO. / N/A]` | CLI, prediction command, requirements, README, release evidence |

> For a four-member group, Member 4 also owns reproducibility and release responsibilities.

**GitHub repository:** `[INSERT REPOSITORY URL]` 

## 3. Dataset Schema

The lecturer-issued CSV must keep the following column names unchanged.

| Field | Type | Purpose |
|---|---|---|
| `record_id` | text | Unique identifier; never used as a model feature |
| `plot_area_ha` | numeric | Farm area |
| `rainfall_mm` | numeric | Recent rainfall estimate |
| `soil_ph` | numeric | Soil acidity measure |
| `seed_kg` | numeric | Seed quantity |
| `distance_km` | numeric | Distance to collection point |
| `arrival_hour` | numeric | Planned arrival hour |
| `actual_yield_kg` | numeric target | Regression target |
| `dispatch_attention` | binary target | Classification target (`0` or `1`) |

### Model input features

The same six input features are used for regression and classification:

```text
plot_area_ha
rainfall_mm
soil_ph
seed_kg
distance_km
arrival_hour
```

For clustering, only these six input features are used.

The following are **never clustering inputs**:

```text
record_id
actual_yield_kg
dispatch_attention
```

This prevents identifier/target leakage into clustering.

---

## 4. Project Structure

```text
AI_A1_G05/
├── README.md
├── requirements.txt
├── run_all.py
├── predict.py
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── regression.py
│   ├── classification.py
│   └── clustering.py
│
├── data/
│   └── AI_A1_G05.csv
│
├── artifacts/
│   ├── data_report.json
│   ├── regression_metrics.json
│   ├── regression_loss.png
│   ├── classification_metrics.json
│   ├── confusion_matrix.png
│   ├── clustering_metrics.json
│   ├── clusters.csv
│   └── cluster_plot.png
│
├── models/
│   └── saved model and preprocessing objects
│
└── evidence/
    ├── AI_A1_G05_DEMO.mp4
    ├── AI_USE.md
    └── TEST_LOG.pdf
```

Do not include virtual environments, `__pycache__`, notebook checkpoints, API keys, passwords, duplicated datasets, or unrelated files in the final ZIP.

---

## 5. Requirements

### Tested Python version

```text
[INSERT THE EXACT PYTHON VERSION USED FOR FINAL TESTING]
```

Check it with:

```bash
python --version
```

### External dependencies

All required third-party packages are listed in `requirements.txt`.

Typical project dependencies are:

- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 6. Environment Setup

Create a clean virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependency integrity check:

```bash
python -m pip check
```

The virtual environment itself must **not** be included in the submission ZIP.

---

## 7. Run the Complete Pipeline

From the project root, run:

```bash
python run_all.py --data data/AI_A1_GXX.csv --output artifacts/ --group AI-GXX
```

Replace `GXX` with the assigned group number.

Example for Group 04:

```bash
python run_all.py --data data/AI_A1_G04.csv --output artifacts/ --group AI-G04
```

### The command performs the following workflow

```text
CSV input
   ↓
schema and type validation
   ↓
missing/duplicate checks
   ↓
SHA-256 fingerprint
   ↓
NumPy feature matrix
   ↓
regression training/evaluation
   ↓
classification training/evaluation
   ↓
clustering evaluation and final labels
   ↓
model/preprocessing persistence
   ↓
JSON / CSV / PNG artifacts
```


