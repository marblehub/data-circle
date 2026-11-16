# Majority-class baseline results

PM-1 task 03, rerun on 29 October 2026. See the [evaluation plan](evaluation_plan.md)
for the scope, split settings and instructions.

## Data and method

- Labelled training records: 59,400.
- Development records: 47,520.
- Reserved labelled holdout: 11,880; no model score was calculated on it.
- Exact cleaned-feature groups: 59,363; 37 extra matching-feature rows retained.
- Five development folds, with identical cleaned-feature records kept together.
- A fresh majority-class classifier is fitted on each fold's fitting labels.
- All five classifiers predicted `functional` for every validation record.
- The supplied test file was not loaded and no test predictions were generated.

## Development cross-validation results

| Metric | Mean | Sample standard deviation |
| --- | ---: | ---: |
| accuracy | 0.543077 | 0.000058 |
| macro_f1 | 0.234629 | 0.000016 |
| repair_precision | 0.000000 | 0.000000 |
| repair_recall | 0.000000 | 0.000000 |
| repair_f1 | 0.000000 | 0.000000 |

The classifier missed all 3,454 repair cases across the development validation
folds and raised 0 repair alarms. Each development record was validated once.
It also detected no non-functional records. Accuracy is about 54.31% because
functional records are the largest class; this is not useful repair detection.
The low macro F1 shows why model selection must consider every class.

## Files for the team

- [Fold scores](baseline_cv_scores.csv): all required class scores and counts.
- [Summary and confusion matrices](baseline_summary.json): settings, averages,
  score spread and package versions.
- [Results template](model_comparison_template.csv): one row per validation fold.
- `data/processed/sprint2_splits.csv`: ignored local ID-level split assignments.

These scores are a reference for Logistic Regression, Random Forest and feature
comparisons. They are not a final-holdout estimate, a result for the unlabelled
test set, or evidence of transfer to new regions. Small variation across folds
reflects their similar class proportions; it does not establish model skill.

## Checks

- Full project test suite: 48 passed.
- Evaluation checks include fold isolation, correct scoring, and running without Git.
- Ruff lint and formatting passed for the new Python files.
- Real-data checks confirm split coverage, matching-feature groups and scores.

Review the first feature list, field availability and evaluation settings with
PM-2 and PM-4 before comparing trained models.
