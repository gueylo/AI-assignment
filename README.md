# Musanze Cooperative Harvest and Dispatch Decision Lab

**Course:** SWE 3513 – Artificial Intelligence  
**Assignment:** Assignment 1  
**Group code:** AI-G05  
**Group members:** Replace this line with the real names and registration numbers.  
**Assigned roles:** Replace this line with the real role allocation.

> **Demo-data warning:** `data/AI_A1_G05.csv` currently contains synthetic demonstration data created only to verify that the software runs. It is not lecturer-issued data and must be replaced before submission or hidden-dataset testing.

## Project purpose

This project analyses a fictional potato cooperative dataset and produces three outputs:

1. A regression prediction for expected potato harvest weight (`actual_yield_kg`).
2. A classification prediction for dispatch attention (`dispatch_attention`, either 0 or 1).
3. Mathematical operating-profile clusters using only the six input features.

The model supports human decisions. It does not guarantee an outcome and must not replace cooperative staff review.

## Dataset schema

The lecturer-issued CSV must contain exactly these required columns:

```text
record_id, plot_area_ha, rainfall_mm, soil_ph, seed_kg,
distance_km, arrival_hour, actual_yield_kg, dispatch_attention
```

The six model inputs are `plot_area_ha`, `rainfall_mm`, `soil_ph`, `seed_kg`, `distance_km`, and `arrival_hour`. `record_id` is an identifier. The two target columns are never used as clustering inputs.

## Tested Python version

Python 3.10 or newer is recommended. Record the exact version used by the group after clean-environment testing.

## Setup

From the project folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the original lecturer-issued file in `data/`. Do not rename or edit its contents. The current `AI_A1_G05.csv` is only synthetic demo data and must be replaced with the real file before submission.

## Run the complete pipeline

```bash
python run_all.py --data data/AI_A1_G05.csv --output artifacts/ --group AI-G05
```

The optional models-folder argument can also be supplied explicitly:

```bash
python run_all.py --data data/AI_A1_G05.csv --output artifacts/ --models models/ --group AI-G05
```

The pipeline dynamically validates the data, calculates the original CSV SHA-256, trains the three analyses, saves models, and creates all required artifacts.

## Make one prediction

Run this only after the pipeline has created the model files:

```bash
python predict.py --models models/ --record "{\"plot_area_ha\":1.2,\"rainfall_mm\":81,\"soil_ph\":5.7,\"seed_kg\":210,\"distance_km\":14,\"arrival_hour\":9}"
```

The output is calculated from the saved preprocessing objects and trained models. An incomplete input should produce a clear validation message:

```bash
python predict.py --models models/ --record "{\"plot_area_ha\":1.2}"
```

## Folder structure

```text
README.md
requirements.txt
run_all.py
predict.py
src/
  __init__.py
  data_pipeline.py
  regression.py
  classification.py
  clustering.py
data/
  AI_A1_G05.csv
artifacts/
models/
evidence/
  AI_USE.md
  TEST_LOG.pdf
```

## Analysis explanations

### Regression

The regression module predicts harvest weight. It standardizes input features using training data only, then learns weights and a bias using NumPy batch gradient descent. It reports mean absolute error (MAE), root mean squared error (RMSE), and R-squared.

### Classification

The classification module uses interpretable logistic regression to estimate whether dispatch attention is required. It reports accuracy, precision, recall, F1 score, and a confusion matrix. A false negative means the model predicted that attention was not required when it was actually required; that could cause a needed intervention to be missed.

### Clustering

The clustering module standardizes the six input features and evaluates KMeans for `k=2`, `k=3`, `k=4`, and `k=5` whenever mathematically valid. Silhouette score selects the best valid `k`. Clusters are mathematical groups of similar standardized operating characteristics, not good/bad farm labels. PCA is used only to draw the two-dimensional plot.

## Expected output files

After a successful run, `artifacts/` contains:

```text
data_report.json
regression_metrics.json
regression_loss.png
classification_metrics.json
confusion_matrix.png
clustering_metrics.json
clusters.csv
cluster_plot.png
```

`models/` contains the saved scalers, models, and metadata needed by `predict.py`.

## Reproducibility settings

The default random seed is `42`. The regression learning rate and epochs, classification threshold, and clustering `k` values are easy-to-find constants near the top of their source files. Changing them changes the trained result and requires rerunning the pipeline.

## Submission information

- GitHub repository URL: `PASTE_REAL_REPOSITORY_URL_HERE`
- Final commit hash: `PASTE_REAL_COMMIT_HASH_HERE`
- Dataset SHA-256: read from the generated `artifacts/data_report.json`
- Demonstration video link: `PASTE_REAL_LINK_HERE`

## Known limitations

- The project requires the lecturer-issued CSV and cannot run meaningful training without it.
- Very small datasets may not support a reliable train/test split, both classes, or every requested clustering value.
- Predictions are recommendations and require human review.
- The project does not claim that a cluster is operationally good or bad.
- The UI/UX PDF is a concept design, not a coded graphical interface.
