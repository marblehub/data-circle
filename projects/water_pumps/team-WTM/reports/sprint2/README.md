# Sprint 2 reports

Start with the written findings, then open the supporting tables when you need
the numbers. The selected model is a balanced Random Forest using the 22 columns
in the `full` column set.

## Suggested reading order

1. [Model comparison and decision](comparisons/model_comparison.md) explains
   which model was selected, its results and its limitations.
2. [Feature and regional findings](feature_checks/pm3_feature_location_checks.md)
   explains the effect of added columns and performance in new regions.
3. [Final holdout result](final/holdout_result.json) records the reserved
   evaluation of the selected model.
4. [Prediction workflow](final/prediction_workflow.md) explains how to use the
   saved model and check delivery.

## Directory guide

| Location | Contents |
| --- | --- |
| [evaluation_plan.md](evaluation_plan.md) | How records are split and how models are evaluated. |
| [baseline/](baseline/) | Majority-class reference scores and [written explanation](baseline/baseline_results.md). |
| [comparisons/](comparisons/) | Candidate model, quantity and location comparisons, per-condition scores, the model decision and the CSV template. |
| [feature_checks/](feature_checks/) | Added-column experiments and results by region and available age. |
| [final/](final/) | Reserved holdout result, prediction delivery checks and prediction instructions. |
| [final_model.json](final_model.json) | Approved model identity, settings and checksummed links to its supporting results. |

Filenames describe the experiment and the level of detail.

- `*_comparison_table.csv` contains averages and variation across development folds.
- `*_cv_scores.csv` contains one row per model, column set and development fold.
- `*_comparison_summary.json` contains settings, package versions and confusion matrices.
- `per_class_scores.csv` contains average precision, recall and F1 for each pump condition.
- `model_comparison_template.csv` is an empty header for collecting results.
- In `feature_checks/`, the existing `pm3_` prefix identifies PM-3's work.
  `ladder` adds column groups step by step, `h6` compares individual columns with
  the combined column set, `groups` reports region and age breakdowns, and
  `region` evaluates transfer to regions excluded from fitting.

## Dashboard data

The keys below are entries in `final_model.json` under `reports`. They identify
the reviewed evidence for the approved model. Development scores come from
separate fold fits. The final saved model was fitted on all 59,400 labelled records.

| Dashboard question | Manifest key and source | How to read it |
| --- | --- | --- |
| How do the candidate models compare? | `model_comparison` · [model comparison](comparisons/model_comparison_table.csv) | One row per model and feature option on the initial comparison. `macro_f1_mean` gives equal weight to each condition. |
| How does adding age, missing-value indicators or location affect results? | `feature_groups` · [feature comparison](feature_checks/pm3_ladder_comparison_table.csv) | Compare the balanced Random Forest across column groups. The `full` row includes location. |
| What changes when water quantity is unavailable? | `quantity` · [quantity comparison](comparisons/quantity_comparison_table.csv) | Compare `full` with `full_no_quantity` for the same model. The supplied test records contain quantity. A pre-visit dashboard needs this value before it can use the approved model. |
| Which pump conditions are hardest to identify? | `per_condition` · [per-condition scores](comparisons/per_class_scores.csv) | Filter by model, feature option, condition and metric. `mean` and `std` summarize the five development folds. |
| How well does the model transfer to a new region? | `region_comparison` · [split comparison](feature_checks/pm3_region_comparison_table.csv) | Ordinary development folds give macro F1 of about 0.683. With entire regions excluded from fitting, it falls to about 0.489. Label the split in the chart. |
| How many repairs did the model miss in each region? | `region_transfer` · [regional transfer](feature_checks/pm3_region_transfer_by_region.csv) | One row per region. Compare `missed_repairs_main` with `missed_repairs_unseen`. The corresponding recall columns give the fraction of actual repairs found. |
| How does performance vary by region or available age? | `regional_groups` · [region](feature_checks/pm3_groups_by_region.csv), `age_groups` · [age](feature_checks/pm3_groups_by_age.csv), `region_age_groups` · [region and age](feature_checks/pm3_groups_by_region_age.csv) | Display `records` and `repair_records` alongside the scores. `too_small` flags groups with fewer than 30 actual repairs. Missing age is concentrated in a few regions. |
| What was the final reserved evaluation score? | `holdout` · [reserved holdout](final/holdout_result.json) | One evaluation of the selected setup on the reserved labelled records. Read `scores.macro_f1`, `scores.repair_recall`, or the confusion matrix with its recorded class order. |
| Did prediction delivery succeed? | [prediction check](final/prediction_check.json) | Check the run ID and model checksum against the manifest, then read `all_checks_passed` and `prediction_rows`. The 14,850 supplied records have no known target, so this report measures delivery only. |

Scores and recall use a 0–1 scale. Columns ending in `_std` use the sample
standard deviation across folds. Mean missed-repair and false-alarm counts in
the comparison tables are per fold. Counts in the regional tables cover the
records in that region. See the [model decision](comparisons/model_comparison.md) and
[PM-3's regional findings](feature_checks/pm3_feature_location_checks.md) for interpretation.

## Read verified results

A results dashboard needs the manifest and its report files. It can read
these without the model binary or the raw data.

```python
from io import BytesIO

import pandas as pd
from team_wtm.artifacts import read_prediction_check, verified_report
from team_wtm.config import SETTINGS

manifest = SETTINGS.report_dir / "final_model.json"
regional = pd.read_csv(BytesIO(verified_report(manifest, "region_transfer")))
delivery = read_prediction_check(manifest, SETTINGS.report_dir / "final/prediction_check.json")
```

`verified_report` compares the file with the recorded SHA-256 before returning
its contents. Report and input text checksums normalize CRLF to LF for Windows
and Linux checkouts. Model and prediction-file checksums use the original bytes.
For a dashboard built with another platform, publish the verified CSV files
together with the manifest and retain its run ID with the imported tables.

## How the existing release was linked

Run `sprint2-rf-419c9f7abddd` was assigned to the existing saved pipeline
during the refactor. Its training settings and columns were checked against
the loaded pipeline, the split plan was reproduced, and prediction delivery
was rerun. The input and report checksums record the files present at that
verification. The historical holdout scores were preserved.

Future successful final fits create a unique run directory under the
configured `paths.runs_dir`, with `model.joblib`, `model.json`, the linked
reports, `config_snapshot.json` and `run.log`. The archive contains local
model bytes and stays outside Git. A fresh setup links its own holdout.
A repeated fit can carry the earlier evaluation tables forward when the
settings, inputs and report contents match. Each fit gets a new run ID;
rerun the prediction completion check to publish delivery totals for it.

If the saved holdout settings, input fingerprints or evidence disagree, restore
the matching files before publishing. A legacy holdout without fingerprints
needs its previously verified manifest. If a final fit is forcibly terminated,
confirm the process has stopped before clearing a leftover `.final_model.lock`
directory in the report folder.

## Generate reports

Run these commands from the project directory. Each command writes to the
configured `paths.report_dir` using the folders above.

| Command | Output folder |
| --- | --- |
| `uv run python -m team_wtm.baseline` | `baseline/` and the comparison template in `comparisons/` |
| `uv run team-wtm compare`, followed by `compare quantity` and `compare location` | `comparisons/` |
| `uv run python -m team_wtm.per_class_scores` | `comparisons/`, using the saved fold scores |
| `uv run python -m team_wtm.feature_checks ladder` and `h6` | `feature_checks/`, using earlier comparisons |
| `uv run python -m team_wtm.feature_checks groups`, followed by `regions` | `feature_checks/` |
| `uv run python -m team_wtm.feature_checks charts` | Charts in `reports/figures/sprint2/` from saved tables |
| `uv run python -m team_wtm.final_model` | Manifest at the report root, holdout in `final/`, model and run archive at configured paths |
| `uv run team-wtm predict` | Delivery check in `final/`, prediction CSV at the configured path |

Comparison commands use development records. The reserved holdout is evaluated
after model selection, with reuse checks described in the evaluation plan.
Prediction delivery checks confirm the output format, record count and model
identity. The supplied test records have no known target, so their prediction
file supplies no accuracy estimate.

## Experiment history

The folders above hold the current published report files. Each new workflow
invocation also saves its own configuration, progress log, status and aggregate
outputs under `paths.workflow_runs_dir`, normally `artifacts/workflows/`.
Use the archived report copy to inspect a past experiment. Use the approved
manifest to select the reviewed evidence for the dashboard.

The CLI exposes `baseline`, `compare`, `features`, `scores`, `final-model`,
`predict` and `export-cleaned`. Their existing Python module commands use the
same workflows. See the [project usage guide](../../README.md#workflow-history-and-logs).

Project entry points also write a rotating log at `paths.log_file`, normally
`artifacts/logs/project.log`. Messages include UTC time, severity, module and
workflow/run IDs. See the [logging guide](../../README.md#standard-project-logging).

For canonical Python imports and the deployment checklist, see the
[source and deployment guide](../../src/team_wtm/README.md).
