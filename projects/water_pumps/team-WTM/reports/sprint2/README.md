# Sprint 2 results for the dashboard

The approved Sprint 2 model is a balanced Random Forest using the 22 columns in
the `full` feature option. Start with [final_model.json](final_model.json).
Its `run_id` identifies the approved fit, `model_sha256` identifies the exact
saved pipeline bytes, and `settings` records the model parameters and ordered
columns. The manifest also records training-library versions, checksums of the
training inputs and split plan, and links to the evaluation evidence.

## Which file should I use?

The keys below are entries in the manifest's `reports` table. They identify
the reviewed results for this release. The development results come from
separate fold fits using the recorded setup. The final pipeline was fitted
on all 59,400 labelled records.

| Dashboard question | Manifest key and source | How to read it |
| --- | --- | --- |
| How do the candidate models compare? | `model_comparison` · [model comparison](model_comparison_table.csv) | One row per model and feature option on the initial comparison. `macro_f1_mean` gives equal weight to each condition. |
| How does adding age, missing-value indicators or location affect results? | `feature_groups` · [feature comparison](pm3_ladder_comparison_table.csv) | Compare the balanced Random Forest across column groups. The `full` row includes location. |
| What changes when water quantity is unavailable? | `quantity` · [quantity comparison](quantity_comparison_table.csv) | Compare `full` with `full_no_quantity` for the same model. The supplied test records contain quantity. A pre-visit dashboard needs this value before it can use the approved model. |
| Which pump conditions are hardest to identify? | `per_condition` · [per-condition scores](per_class_scores.csv) | Filter by model, feature option, condition and metric. `mean` and `std` summarize the five development folds. |
| How well does the model transfer to a new region? | `region_comparison` · [split comparison](pm3_region_comparison_table.csv) | Ordinary development folds give macro F1 of about 0.683. With entire regions excluded from fitting, it falls to about 0.489. Label the split in the chart. |
| How many repairs did the model miss in each region? | `region_transfer` · [regional transfer](pm3_region_transfer_by_region.csv) | One row per region. Compare `missed_repairs_main` with `missed_repairs_unseen`. The corresponding recall columns give the fraction of actual repairs found. |
| How does performance vary by region or available age? | `regional_groups` · [region](pm3_groups_by_region.csv), `age_groups` · [age](pm3_groups_by_age.csv), `region_age_groups` · [region and age](pm3_groups_by_region_age.csv) | Display `records` and `repair_records` alongside the scores. `too_small` flags groups with fewer than 30 actual repairs. Missing age is concentrated in a few regions. |
| What was the final reserved evaluation score? | `holdout` · [reserved holdout](holdout_result.json) | One evaluation of the selected setup on the reserved labelled records. Read `scores.macro_f1`, `scores.repair_recall`, or the confusion matrix with its recorded class order. |
| Did prediction delivery succeed? | [prediction check](prediction_check.json) | Check the run ID and model checksum against the manifest, then read `all_checks_passed` and `prediction_rows`. The 14,850 supplied records have no known target, so this report measures delivery only. |

Scores and recall use a 0–1 scale. Columns ending in `_std` use the sample
standard deviation across folds. Mean missed-repair and false-alarm counts in
the comparison tables are per fold. Counts in the regional tables cover the
records in that region. See the [model decision](model_comparison.md) and
[PM-3's regional findings](pm3_feature_location_checks.md) for interpretation.

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
delivery = read_prediction_check(manifest, SETTINGS.report_dir / "prediction_check.json")
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
