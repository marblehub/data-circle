# Model comparison and final setup

PM-4, issue [#17](https://github.com/marblehub/data-circle/issues/17). Run on
11 November 2026. Follows the [evaluation plan](evaluation_plan.md) and
compares against the [majority baseline](baseline_results.md).

## Result in short

- **Final setup:** Random Forest, `class_weight="balanced"`, PM-3 `full` features.
- **Development cross-validation:** macro F1 0.683 ± 0.005, accuracy 0.779.
- **Labelled holdout (scored once, after the choice):** macro F1 0.682,
  accuracy 0.780, repair recall 0.481, repair precision 0.380.
- **Plan B** if `quantity` is judged unavailable at prediction time:
  the same model on `full_no_quantity`, macro F1 0.658 ± 0.007
  (cross-validation only; not scored on the holdout).
- Baseline for comparison: macro F1 0.235, repair recall 0.

## Method

- Same split as the baseline: seed 42, 47,520 development records in five
  StratifiedGroupKFold folds, 11,880 records in the labelled holdout
  (`cv_fold = -1`), from `data/processed/sprint2_splits.csv`.
- Per fold: one scikit-learn Pipeline is fitted on the fitting rows only and
  predicts the validation rows. Steps:
  1. `categories_as_text`: True/False values become text, missing stays missing.
     Needed because the encoder cannot mix True/False with `"__MISSING__"`.
     Learns nothing.
  2. PM-2 `build_preprocessor`: median imputation with missing indicators,
     scaling, one-hot encoding (categories under 1% grouped). Learned per fold.
  3. The model, newly created for each fold.
- Before the folds, the shared `clean_data` and PM-3 `add_feature_columns`
  are applied. They are fixed row-by-row rules and learn nothing from the data.
- Scores come from PM-1 `score_predictions`. Summary: mean and sample
  standard deviation (ddof=1) over the five folds.
- Models: Logistic Regression (`max_iter=2000`) and Random Forest
  (300 trees), each with and without `class_weight="balanced"`.
  No hyperparameters were tuned.

Code: `src/team_wtm/model_comparison.py` and `src/team_wtm/final_model.py`.

## 1. Models on the `base` features (13 columns)

Mean ± sample std over five folds. Missed repairs and false alarms are counts
per validation fold (about 9,500 records, about 690 of them needing repair).

| Model | Macro F1 | Accuracy | Repair precision | Repair recall | Repair F1 | Missed repairs | False repair alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF balanced | 0.616 ± 0.008 | 0.710 ± 0.009 | 0.252 ± 0.012 | 0.510 ± 0.013 | 0.337 ± 0.011 | 339 ± 9 | 1,048 ± 74 |
| RF | 0.613 ± 0.005 | 0.756 ± 0.002 | 0.404 ± 0.014 | 0.219 ± 0.016 | 0.284 ± 0.016 | 539 ± 11 | 223 ± 12 |
| LR balanced | 0.521 ± 0.003 | 0.585 ± 0.002 | 0.151 ± 0.005 | 0.588 ± 0.020 | 0.240 ± 0.008 | 284 ± 14 | 2,283 ± 72 |
| LR | 0.488 ± 0.002 | 0.721 ± 0.001 | 0.350 ± 0.224 | 0.001 ± 0.001 | 0.003 ± 0.002 | 690 ± 1 | 2 ± 1 |
| Majority baseline | 0.235 ± 0.000 | 0.543 ± 0.000 | 0.000 | 0.000 | 0.000 | 691 ± 0 | 0 |

- Every model beats the baseline. Random Forest beats Logistic Regression by
  about 0.1 macro F1.
- Without class weights, Logistic Regression almost never predicts
  "needs repair". Balanced weights raise repair recall a lot for both models,
  at the cost of more false alarms and lower accuracy.

## 2. Quantity check (Random Forest)

`full` = base + age + missing-value flags + location (22 columns).
`full_no_quantity` = `full` without `quantity` and `quantity_group`.
Only this pair isolates the effect of quantity.

| Features | Model | Macro F1 | Accuracy | Repair precision | Repair recall | Missed repairs | False repair alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | RF balanced | **0.683 ± 0.005** | 0.779 ± 0.005 | 0.383 ± 0.012 | 0.489 ± 0.013 | 353 ± 9 | 545 ± 33 |
| full | RF | 0.678 ± 0.011 | 0.797 ± 0.006 | 0.494 ± 0.028 | 0.333 ± 0.027 | 461 ± 18 | 235 ± 20 |
| full_no_quantity | RF balanced | 0.658 ± 0.007 | 0.748 ± 0.005 | 0.376 ± 0.015 | 0.484 ± 0.016 | 356 ± 11 | 556 ± 33 |
| full_no_quantity | RF | 0.649 ± 0.010 | 0.764 ± 0.005 | 0.481 ± 0.024 | 0.316 ± 0.023 | 473 ± 16 | 235 ± 13 |

- Removing quantity lowers macro F1 by 0.025 (balanced) and 0.029 (plain RF),
  3–5 times the fold standard deviation. Quantity adds real information.
- `full_no_quantity` still beats `base` (+0.04), because age, missing-value
  flags and location add more than quantity.

## 3. Location check (RF balanced)

`base_age_missing` = `full` without the location group.

| Features | Macro F1 | Accuracy | Repair precision | Repair recall | False repair alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.616 ± 0.008 | 0.710 ± 0.009 | 0.252 ± 0.012 | 0.510 ± 0.013 | 1,048 ± 74 |
| base_age_missing | 0.638 ± 0.008 | 0.732 ± 0.009 | 0.278 ± 0.012 | 0.529 ± 0.018 | 949 ± 70 |
| full | 0.683 ± 0.005 | 0.779 ± 0.005 | 0.383 ± 0.012 | 0.489 ± 0.013 | 545 ± 33 |

- Of the +0.067 gain from `base` to `full`, age and missing-value flags give
  +0.022 and location gives +0.045.
- Location mainly reduces false repair alarms (949 → 545 per fold). Repair
  recall falls slightly (0.53 → 0.49).

## Choice (PM-1 selection rule)

1. **Mean macro F1:** `full` is clearly best. RF balanced (0.683) and RF
   (0.678) differ by less than one fold standard deviation, so they are close.
2. **Fold variation:** RF balanced is more stable (0.005 vs 0.011).
3. **Repair recall and precision:** RF balanced finds about 108 more
   real repairs per fold (recall 0.49 vs 0.33) and raises about 310 more
   false alarms (precision 0.38 vs 0.49).
4. **Simpler model:** both are the same model type. The documented gain in
   repair detection decides for RF balanced.

The costs of a missed repair and a false alarm have not been quantified. If
the team later values false alarms as more costly, plain RF on `full` is the
alternative with almost the same macro F1.

## Labelled holdout (scored once)

The fixed setup was fitted on all 47,520 development records and scored once
on the 11,880 holdout records. Nothing was changed after this score.

| Metric | Cross-validation | Holdout |
| --- | ---: | ---: |
| Macro F1 | 0.683 ± 0.005 | 0.682 |
| Accuracy | 0.779 ± 0.005 | 0.780 |
| Repair precision | 0.383 ± 0.012 | 0.380 |
| Repair recall | 0.489 ± 0.013 | 0.481 |
| Repair F1 | 0.429 ± 0.009 | 0.425 |

Holdout confusion matrix (rows: actual, columns: predicted):

| Actual \ Predicted | functional | needs repair | non functional |
| --- | ---: | ---: | ---: |
| functional (6,452) | 5,214 | 494 | 744 |
| needs repair (863) | 322 | 415 | 126 |
| non functional (4,565) | 746 | 183 | 3,636 |

448 repairs were missed and 677 false repair alarms were raised. The holdout scores are close to the cross-validation scores, so there is no sign that the model selection overfitted the development folds. This does not show how the model transfers to a new area.

## Final model

- Refitted on all 59,400 labelled records.
- Saved locally as `models/final_pipeline.joblib` (about 104 MB, ignored by Git).
- The file contains the whole pipeline: input = the 22 `full` columns after
  `clean_data` and `add_feature_columns`.
- Reload check: identical predictions and probabilities on all 59,400 records.
- The supplied test file was not loaded and no test predictions were made.

## Caveats and open points

- **Quantity availability (PM-1, PM-4):** quantity describes the inspected
  condition. It is not confirmed that it would be known before a real
  prediction. If not, use plan B.
- **Location:** the folds keep exact duplicate rows together, not nearby
  pumps. Part of the location gain may come from neighbouring pumps looking
  alike. PM-3's location-based validation should be read with this result
  before claiming the model transfers to new areas.
- **No tuning:** default settings and 300 trees. Tuning would use the same
  development folds; the holdout is already used and must not guide it.
- **Costs:** the trade-off between missed repairs and false alarms is a team
  decision; the numbers above support it but do not make it.

## Output files

All in `reports/sprint2/`; scores and counts only, no record-level data.

- `model_*`, `quantity_*`, `location_*`: `_cv_scores.csv` (one row per fold,
  shared template columns), `_comparison_table.csv` (mean and std),
  `_comparison_summary.json` (confusion matrices per fold, settings, versions).
- `holdout_result.json`: the one-time holdout score.
- `final_model.json`: final settings and the reload check.

## Reproduce

From `projects/water_pumps/team-WTM`, in this order:

```bash
uv run python -m team_wtm.baseline                     # split file + baseline
uv run python -m team_wtm.model_comparison             # four models on base
uv run python -m team_wtm.model_comparison quantity    # quantity check
uv run python -m team_wtm.model_comparison location    # location check
uv run python -m team_wtm.final_model                  # holdout (once) + final model
```

Later commands reuse files written by earlier ones. `final_model` does not
score the holdout again if `holdout_result.json` already exists.
