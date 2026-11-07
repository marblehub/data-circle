# Sprint 2 evaluation plan

PM-1 tasks 01–04, issue [#14](https://github.com/marblehub/data-circle/issues/14).
Updated 29 October 2026. Review these working settings with PM-2 and PM-4
before using them for model comparisons.

## 01 — Prediction goal and features

Predict the condition recorded for a waterpoint: `functional`,
`functional needs repair`, or `non functional`. This is classification of the
recorded condition, not a forecast of when a pump will fail.

Use the supplied labelled training records to learn and compare models.
`Test_set_values.csv` is a separate unlabelled prediction set.
`Submission_format.csv` contains placeholders, not answers for scoring.

The majority baseline uses no predictors. For the first trained-model
comparison, use this working feature list, matching the PM-3 base proposal:

| Group | First comparison fields |
| --- | --- |
| Technology and source | extraction_type_class, waterpoint_type_group, source_type |
| Management and payment | management_group, payment_type, scheme_management, permit, public_meeting |
| Water quality and quantity | water_quality, quantity, quantity_group |
| Numeric fields | population, gps_height |

These 13 fields are a starting point, not a claim that other fields are useless.
PM-2 and PM-3 can propose changes, recording the exact feature list for each run.
Use PM-3's age, missing-value and location options for the planned comparisons.

Field rules:

- Keep `id` for joins and local split matching. Never use it as a predictor.
- Keep `status_group` as the target only. Exclude label-derived flags too.
- Follow the shared cleaner: zero population/construction year becomes missing,
  recording dates are parsed, and `num_private` is dropped.
- Derive pump age using the recording year. Do not calculate it from today's date.
- Keep original rows and labels, including repeated features and conflicting labels.
- Leave waterpoint/scheme names, funder/installer names, amount_tsh and other
  non-base fields out of the first comparison. Review them before adding them;
  do not rewrite the source table or silently merge names.
- `quantity` and `quantity_group` must be removed together in the quantity check.
- Retain uncertain elevation zeros in shared data. Their meaning remains open.

Availability assumption: candidate fields come from the same supplied survey.
We have not confirmed that each field would be available before a real-world
prediction. In particular, quantity and water quality describe current inspected
conditions. Their use here does not prove future prediction ability or leakage.
PM-4 should review the intended use with PM-1; PM-2 should record any unavailable
fields. The quantity comparison remains required before the final choice.

## 02 — Shared splits and cross-validation

Working settings:

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Labelled final holdout | First fold from a five-fold StratifiedGroupKFold split; about 20% of records |
| Development records | Remaining roughly 80% |
| Model comparisons | Five-fold StratifiedGroupKFold within development records |
| Duplicate groups | Exact feature matches after shared cleaning, excluding ID and target |
| Local split file | data/processed/sprint2_splits.csv |
| Final holdout fold value | cv_fold = -1; split = holdout |

The split code sorts by ID before grouping, so changing CSV row order does not
change assignments. Missing values stay in matching-feature groups. Conflicting
labels do not separate otherwise identical features. Stratification aims to keep
class proportions similar; group sizes mean exact percentages are not promised.

The local split file contains `id`, `group`, `split` and `cv_fold`. It is ignored
by Git. Share the code/settings, not individual records. Use the same assignments
for every model and feature comparison. Do not regenerate the plan from a reduced
feature subset. Keep the same training files and package versions for each run.
If the training files change, make new splits and rerun the baseline before
comparing models.

During cross-validation:

1. Fit missing-value replacements, category grouping/encoding, scaling and any
   learned feature selection on the fitting rows of that fold only.
2. Apply those fitted steps to the fold's validation rows.
3. Score those predictions using the common functions below.
4. Repeat for all five folds. Report the mean and sample standard deviation.

The grouping rule prevents exact cleaned-feature matches from crossing folds.
It does not establish that all pumps or nearby locations are independent.
Reducing features can also create further matches. PM-2 should review this;
PM-3's separate location-based development check addresses geographic transfer.

Do not tune against the labelled final holdout. After model/feature selection,
PM-4 evaluates the fixed setup on that holdout once. Only then refit the complete
pipeline on all labelled training records and predict the supplied test records.
No holdout model score is produced by the baseline command.

## 03 — Majority-class baseline

Run from the project directory using the existing environment:

```bash
.venv/bin/python -m team_wtm.baseline
```

Or, with the project environment managed by uv:

```bash
uv run python -m team_wtm.baseline
```

The command loads the two training CSVs, validates their ID join, makes the
shared plan, and fits a fresh `DummyClassifier(strategy="most_frequent")`
in each development fold. Its majority class comes from that fold's fitting
labels only. Numeric zeros supply row counts; no feature values are learned.

Outputs:

- `reports/sprint2/baseline_cv_scores.csv`: one aggregate row per validation fold.
- `reports/sprint2/baseline_summary.json`: mean/spread, labelled confusion matrices,
  settings and package versions. Matrix row and column labels
  are given by `class_order`; rows are actual and columns are predicted labels.
- `reports/sprint2/model_comparison_template.csv`: the shared CSV header. Existing
  template edits are not overwritten when rerunning the baseline.
- `data/processed/sprint2_splits.csv`: local ID-level assignments; do not commit.

Rerunning replaces the generated baseline report and local split file. Use the
same training files when comparing results. Raw CSVs are not edited.

The [baseline results](baseline_results.md) explain the completed run.

The two Python files have separate jobs:

- `baseline.py` loads training data, runs the baseline, and saves scores.
- `evaluation.py` makes the shared splits and calculates the common scores.

The comments in `baseline.py` follow the run in four steps. The splitter comes
from scikit-learn because it must balance classes while keeping matching records
together. No Git installation is needed to run the baseline.

## 04 — Common results and handoff

Use one row per development validation fold in the CSV template. Copy the
baseline rows or add Logistic Regression, Random Forest and feature experiments
with the same columns. Use a short feature-option name; record full settings and
fields in the accompanying experiment notes.

Required results:

- accuracy and macro F1 (each pump state has equal weight in macro F1);
- precision, recall, F1 and validation support for each state;
- missed repairs: actual repair records predicted as another state;
- false repair alarms: non-repair records predicted as needing repair;
- a three-by-three confusion matrix in the fixed class order;
- fold number, fitting/validation counts, seed and experiment notes.

Class column prefixes are `functional`, `repair` and `non_functional`.
A zero score is reported for an unpredicted class rather than hiding it.
The confusion matrices live in the JSON report, not in CSV cells. For the final
model comparison report, summarize fold scores with mean and sample standard
deviation (ddof=1). Keep final-holdout results separate and clearly labelled.

Working selection rule: compare mean macro F1 first. Review fold variation,
repair recall and repair precision before choosing a model. Missing a repair
and raising a false repair alarm have different costs; those costs have not
been quantified. When differences are small, prefer a simpler model unless
there is a documented repair-detection benefit. There is no promised target score.

PM-4 can reuse the split and score functions like this:

```python
import pandas as pd
from team_wtm.cleaning import PROJECT_DIR, load_data, merge_data
from team_wtm.evaluation import development_folds, score_predictions

values, labels = load_data()
training = merge_data(values, labels, keep_merge_indicator=True)
assert training["_merge"].eq("both").all()
training = training.drop(columns="_merge")
plan = pd.read_csv(PROJECT_DIR / "data/processed/sprint2_splits.csv")

for fold, fit_rows, validation_rows in development_folds(training, plan):
    fit_data = training.iloc[fit_rows]
    validation_data = training.iloc[validation_rows]
    # Select the same feature list for both tables; exclude ID and target.
    # Fit a fresh preparation-and-model pipeline using fit_data only.
    # Predict validation_data, then call:
    # scores = score_predictions(validation_data["status_group"], predictions)
```

PM-2 owns learned preparation. PM-3 owns feature comparisons and geographic
checks. PM-4 owns the model-running code. PM-1 supplies this plan, baseline and
results format. PM-4 reviews PM-1's evaluation code; PM-1 reviews PM-4's modelling.

Review the first feature list, field availability, split settings and selection
priorities with PM-2 and PM-4 before starting the model comparisons.
