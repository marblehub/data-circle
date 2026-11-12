# Issue #17: train, compare and evaluate models

## Objective and scope

11–12 November 2026. Train Logistic Regression and Random Forest on the shared
Sprint 2 folds, compare them with the majority baseline, choose a final setup
with the PM-1 rule, score it once on the labelled holdout, then refit and save
it. XGBoost was not added (not part of Issue #17).

## Changed and created

- `src/team_wtm/model_comparison.py`: cross-validation loop, mean/std summary,
  comparison table, and the `models`, `quantity` and `location` experiments.
- `src/team_wtm/final_model.py`: one-time holdout score, refit on all labelled
  records, joblib save and reload check.
- `tests/test_model_comparison.py` (9 tests), `tests/test_final_model.py`
  (3 tests), `tests/conftest.py` (shared 120-row synthetic table).
- `reports/sprint2/`: fold scores, comparison tables and JSON summaries for the
  three experiments, `holdout_result.json`, `final_model.json`,
  `model_comparison.md`.
- This task log.

Merged without conflicts and without changes: PM-1 #14, PM-3 #16, PM-2 #15
(in that order). No teammate file was edited.

## Decisions and reasoning

- **Shared evaluation code.** My earlier local `evaluation.py` had the same
  name as PM-1's file. PM-1's `score_predictions` already covered its scores
  and matches the CSV template, so my file was dropped. Only the comparison
  table idea moved into `model_comparison.py`, plus new mean/std code.
- **Same folds for everything.** All runs read PM-1's split file. The plan was
  not rebuilt from any feature subset.
- **No leakage.** Learned preparation (PM-2 preprocessor) sits inside a
  Pipeline, fitted on each fold's fitting rows only. `clean_data` and PM-3
  `add_feature_columns` are fixed row-wise rules, applied before splitting.
- **True/False fix.** PM-2's encoder failed on `permit` / `public_meeting`
  (True/False mixed with `"__MISSING__"`). Added `categories_as_text` as the
  first pipeline step instead of editing PM-2's file.
- **Class weights.** "Needs repair" is about 7% of records, so every model
  was also run with `class_weight="balanced"`.
- **Quantity check.** `full` vs `full_no_quantity` isolates quantity. `base`
  vs `full_no_quantity` (as first asked) also adds age, flags and location, so
  both comparisons are reported.
- **Location check.** `base_age_missing` vs `full` isolates location.
- **Final choice.** RF balanced + `full` (see `model_comparison.md`). Plan B:
  RF balanced + `full_no_quantity`, cross-validation only.
- **Holdout once.** `final_model.py` writes `holdout_result.json` first and
  never scores the holdout again if the file exists.

## Checks and results

- PM-1 baseline rerun: accuracy 0.5431, macro F1 0.2346, repair recall 0 in
  every fold. Same scores as PM-1; only the Python patch version differed
  in `baseline_summary.json`, which was restored.
- Development CV, macro F1 (mean ± std): baseline 0.235, LR 0.488,
  LR balanced 0.521, RF 0.613, RF balanced 0.616 (base); RF balanced 0.638
  (base_age_missing), 0.658 (full_no_quantity), 0.683 (full).
- Holdout (once): macro F1 0.682, accuracy 0.780, repair recall 0.481,
  repair precision 0.380. Close to CV.
- Final pipeline refitted on 59,400 records; after joblib reload,
  predictions and probabilities identical on all 59,400 records.
- Tests: 12 new tests pass. They include a "spy" pipeline proving that fitting
  never sees validation or holdout rows, and a check that changing every
  holdout label leaves the CV scores unchanged.
- Full suite: 69 passed (57 existing + 12 new), 1 failed. The failure
  (`test_export_cleaned.py::test_snapshot_round_trip_and_metadata_preserve_all_training_records`)
  existed before this task: on Windows, `export_cleaned` stores paths with
  backslashes but the test expects forward slashes. Not changed here.

## Commands

```bash
uv run python -m team_wtm.baseline
uv run python -m team_wtm.model_comparison
uv run python -m team_wtm.model_comparison quantity
uv run python -m team_wtm.model_comparison location
uv run python -m team_wtm.final_model
uv run pytest -q
```

A full run takes about 15–20 minutes on a laptop (Random Forest runs dominate).

## Uncertainty and remaining work

- Confirm with PM-1 whether `quantity` is available at prediction time
  (decides final setup vs plan B).
- Read PM-3's location-based validation with the location gain.
- Optional: tune Random Forest settings on the development folds.
- Team decision on the cost of missed repairs vs false alarms.
- Report the Windows path issue in `test_export_cleaned.py` to its owner.

## Handover

Code, tests and aggregate reports ready for review by PM-1. The saved model
(`models/final_pipeline.joblib`) and the split file stay local and are not
committed. No individual records are committed.
