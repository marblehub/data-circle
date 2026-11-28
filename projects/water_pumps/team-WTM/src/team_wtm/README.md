# Source and dashboard guide

The application and terminal commands share this package. Choose a function
by its responsibility; keep page layout, widgets and platform-specific code
in the application layer, such as a future `app/` directory.

## Module map

| Package or module | Responsibility | Main files |
| --- | --- | --- |
| `data/` | Read, validate, clean and prepare records; export a snapshot | `cleaning.py`, `preparation.py`, `export.py` |
| `modeling/` | Define column sets, learned preprocessing and estimators; fit and predict | `features.py`, `preprocessing.py`, `estimators.py`, `training.py`, `predict.py` |
| `evaluation/` | Define shared splits and metrics; compare candidates and summarize scores | `splits.py`, `baseline.py`, `comparison.py`, `per_condition.py` |
| `analysis/` | Investigate numerical/categorical relationships, added columns and regional performance | `numeric.py`, `categorical.py`, `feature_checks.py`, `charts.py` |
| `config.py` | Load and validate the central TOML settings | `TEAM_WTM_CONFIG` selects an alternative file |
| `cli.py` | Parse terminal arguments and display returned results | `team-wtm` command |
| `artifacts.py` | Verify model/report identity and read archived evidence | `verified_report()`, `read_prediction_check()` |
| `runtime.py` | Record workflow inputs, settings, source hashes, status and outputs | `track_workflow()` |
| `logging_config.py` | Configure project logs and route contextual workflow messages | `configure_logging()` |

A useful reading order is `data/preparation.py`, `modeling/features.py`,
`modeling/preprocessing.py`, `evaluation/splits.py`, then
`evaluation/comparison.py`, `modeling/training.py` and `modeling/predict.py`.
The [Sprint 2 index](../../reports/sprint2/README.md) explains the saved results.

Record preparation applies fixed cleaning and column rules. Learned imputation,
scaling and encoding live in the model pipeline and are fitted inside each
training fold. Prediction applies the saved preprocessing; the dashboard does
not fit encoders, retrain the model or read the labelled holdout.

## Predict from a dashboard

Set the configuration environment variable before importing configured modules,
then initialize logging once during application startup. For example, in the
deployment environment:

```bash
export TEAM_WTM_CONFIG=/srv/wtm/configs/project.toml
```

The application can call these ordinary Python functions:

```python
from pathlib import Path

from team_wtm.logging_config import configure_logging
from team_wtm.modeling.predict import load_model, predict_records

configure_logging(console=False)

model = load_model(
    Path("/srv/wtm/models/model.joblib"),
    report_path=Path("/srv/wtm/models/model.json"),
)
# records is a pandas DataFrame loaded from an upload or another data source.
predictions = predict_records(records, model)
```

Load the approved model at application startup or through the framework's
resource cache, and reuse it for prediction. When changing the model, invalidate
that cache using its manifest identity or checksum. Present input-validation
errors in the application. Keep record IDs and the returned row order when
displaying or downloading predictions.

`predict_records()` returns a DataFrame without creating batch output files.
For a saved batch and invocation history, use
`team_wtm.modeling.predict.run_prediction()`. Give concurrent users distinct
output paths and check-report paths. Reusable workflows accept an `on_progress`
callback when a progress display is needed.

## Display reviewed results

A results page needs the approved manifest and linked reports. It can read them
without loading the model, configuring training or accessing the training CSVs:

```python
from io import BytesIO
from pathlib import Path

import pandas as pd
from team_wtm.artifacts import verified_report

manifest = Path("/srv/wtm/reports/sprint2/final_model.json")
comparison = pd.read_csv(BytesIO(verified_report(manifest, "model_comparison")))
```

Keep the manifest and its relative report paths together. Display its run ID
alongside results so readers can identify the model being discussed. A prediction
delivery check confirms successful delivery; it supplies no accuracy estimate
because the supplied test records have no known target.

## Deployment checklist

1. Install the package and the versions locked in `uv.lock`. Deploy the complete
   package, including the compatibility modules below.
2. Supply `configs/project.toml` separately for a wheel installation. Point
   `TEAM_WTM_CONFIG` at its absolute path before configured imports. Set
   `paths.project_root` to the deployment's data and artifact directory;
   relative paths resolve from that root.
3. For predictions, supply the reviewed model and its matching manifest. A
   custom model defaults to a JSON sidecar; `load_model(..., report_path=...)`
   can select another manifest. Deploy the reviewed report bundle as well
   when the application displays results.
4. Configure writable `log_file` and `workflow_runs_dir` paths. Batch workflows
   also need writable output paths. Use persistent storage if histories must
   survive application restarts. Raw data and reviewed artifacts can remain
   read-only for an inference-only application.
5. Use `predict_records()` with uploaded records. The default batch completion
   check additionally needs the supplied test CSV and submission template.
   Retraining workflows need the labelled CSVs and matching split plan.
6. Verify the chosen deployment with representative input records and expected
   model identity. Keep training/comparison jobs outside ordinary page rendering;
   display their saved reports instead.

The package has no dependency on dashboard widgets. The same APIs can serve
Streamlit, a web service or another Python dashboard. Folder organization and
package compatibility have been checked; production hosting, application UI
and platform-specific caching still belong to the dashboard implementation.

## Existing imports and commands

Earlier root modules are small compatibility forwards, for example:

| Existing import | Canonical implementation |
| --- | --- |
| `team_wtm.cleaning` | `team_wtm.data.cleaning` |
| `team_wtm.model_data_preparation` | `team_wtm.data.preparation` and `team_wtm.modeling.preprocessing` |
| `team_wtm.features` | `team_wtm.modeling.features` |
| `team_wtm.model_comparison` | `team_wtm.evaluation.comparison` |
| `team_wtm.final_model` | `team_wtm.modeling.training` |
| `team_wtm.predict` | `team_wtm.modeling.predict` |
| `team_wtm.feature_checks` | `team_wtm.analysis.feature_checks` |

`team_wtm.evaluation` continues to export the shared split/scoring functions.
New code should use the canonical packages. Existing notebooks, commands such
as `python -m team_wtm.predict`, and serialized pipeline references remain
supported. The preferred terminal interface is still `team-wtm`.
