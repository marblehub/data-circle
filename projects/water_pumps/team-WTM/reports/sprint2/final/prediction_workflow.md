# Prediction workflow (Issue #17)

Code: `src/team_wtm/predict.py` · Tests: `tests/test_predict.py` ·
Check report: `reports/sprint2/final/prediction_check.json`

This workflow predicts the pump status of new records using the saved
final model. It **only predicts**. It never fits or retrains the model, and it
never reads or scores the labelled holdout. It does not use Streamlit, so the
command line, notebooks and the app can all call the same functions.

## What it uses

| Item | Value | Where it comes from |
|---|---|---|
| Model file | `models/final_pipeline.joblib` (~104 MB, git-ignored) | `final_model.MODEL_PATH` |
| Model | `random_forest_balanced` | `../final_model.json → settings.model` |
| Features | `full` (22 columns) | `../final_model.json → settings.features` |
| Cleaning + feature columns | `clean_data` → `add_feature_columns` | `model_comparison.add_model_columns` |
| Feature selection | `select_features(..., "full")` | `features.py` |
| Column order | `model.feature_names_in_` | saved pipeline |

Prediction calls the same shared functions as training,
so the records are prepared exactly the same way. For deployment, supply the trusted binary whose checksum matches the manifest.
Running `python -m team_wtm.final_model` fits and publishes a new run.

## Steps

1. **Load the model** (`load_model`). Verifies the approved binary checksum, estimator parameters and ordered columns against its manifest. Warns if pandas, numpy, scikit-learn or
   joblib differ from the versions saved in `final_model.json`.
2. **Check the records** (`validate_records`). Refuses records that:
   - are not a pandas DataFrame,
   - are empty,
   - contain the target column `status_group`,
   - have no `id` column, or missing or duplicate ids.
3. **Build the features** (`prepare_features`). Cleans the records, adds the
   feature columns, selects the 22 `full` columns and puts them in the order
   the model was trained with. Custom pipelines can use another registered feature option, and
   `check_feature_columns` refuses missing, extra or wrongly ordered columns.
4. **Predict** (`predict_records`). Calls `model.predict` only. Returns one
   row per record with columns `id, status_group`, in the original record
   order (`validate_predictions` checks this).
5. **Save** (`save_predictions`) as CSV or JSON.

Allowed labels: `functional`, `functional needs repair`, `non functional`.

## Usage

Run from the `team-WTM` folder (Windows path to Python shown; on macOS/Linux
use `.venv/bin/python` or `uv run python`).

```powershell
# Completion check on the DrivenData test records (default run)
.venv\Scripts\python.exe -m team_wtm.predict

# Your own records (CSV without status_group)
.venv\Scripts\python.exe -m team_wtm.predict --records path\to\records.csv --output path\to\predictions.csv

# JSON output and a different model file
.venv\Scripts\python.exe -m team_wtm.predict --records records.csv --model models\final_pipeline.joblib --format json --output predictions.json
```

| Option | Default | Meaning |
|---|---|---|
| `--records` | test records in `data/raw` | CSV of records without `status_group`. Without it, the completion check runs. |
| `--model` | `models/final_pipeline.joblib` | Saved model file. |
| `--model-report` | Approved manifest or matching custom `.json` sidecar | A manifest supplied explicitly must match the model. A trusted standalone custom pipeline without a manifest is identified by its estimator type and checksum, with a null run ID. |
| `--output` | `data/processed/test_predictions.csv` | Where predictions are written. |
| `--format` | `csv` | `csv` or `json` (a list of `{id, status_group}`). |

From Python (for example a notebook or the app):

```python
import pandas as pd
from team_wtm.predict import load_model, predict_records

model = load_model()                      # models/final_pipeline.joblib
records = pd.read_csv("data/raw/Test_set_values.csv")
predictions = predict_records(records, model)   # columns: id, status_group
```

## Completion check (default run)

Without `--records`, the command looks for the test records in `data/raw`
(first found of `test_features.csv.csv`, `Test_set_values.csv`,
`test_features.csv`; the file is not renamed) and checks against
`data/raw/Submission_format.csv`:

- same number of rows,
- same ids in the same order,
- columns exactly `id, status_group`,
- only allowed labels,
- no missing values,
- identical predictions after loading the model file a second time.

Predictions are written to `data/processed/test_predictions.csv`
(git-ignored, local only). A totals-only report (counts and PASS/FAIL, no
individual records) is written to `reports/sprint2/final/prediction_check.json`.
Custom-model completion checks use a separate totals file next to the predictions (the output filename plus `.check.json`). The command exits with an error if any check fails. See the [dashboard results index](../README.md) for run IDs and report verification.

Result of the last run:

| Check | Result |
|---|---|
| Records | 14,850 rows, 40 columns |
| Predictions | 14,850 rows |
| Rows, ids and order match SubmissionFormat | PASS |
| Columns exactly `id, status_group` | PASS |
| Labels valid, no missing values | PASS |
| Same predictions after reload | PASS |
| Version warnings | none |

Predicted label counts: functional 7,848 · functional needs repair 1,377 ·
non functional 5,625.

These are predictions only. The test records have no labels, so this run
gives no score; the model's score is in `holdout_result.json`.

## Automated checks

```powershell
.venv\Scripts\python.exe -m ruff check src/team_wtm/predict.py tests/test_predict.py
.venv\Scripts\python.exe -m ruff format --check src/team_wtm/predict.py tests/test_predict.py
.venv\Scripts\python.exe -m pytest tests/test_predict.py
```

`tests/test_predict.py` uses made-up records (no real data) and checks:
one prediction per record with ids in order, allowed labels, no fitting
(a wrapped model fails if `fit` is called), no holdout code in the module,
every rejected-record case, column order and refusals, reload gives the same
predictions, version warnings, all three test file names, CSV and JSON
output, the command line, the completion check (pass and fail) and that
Streamlit is not imported.
