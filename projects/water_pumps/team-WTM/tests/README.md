# Test suite

Run commands from the Team WTM project directory.

| Folder | Purpose |
| --- | --- |
| `unit/` | Individual cleaning, configuration, column preparation, scoring and validation contracts. Small temporary files or synthetic fitted models support some checks. |
| `integration/` | Connected cleaning, training, prediction, export, logging, saved-model and CLI workflows. |
| `conftest.py` | Shared synthetic records, model fixtures and isolation of generated logs/history. |

Run the complete suite.

```bash
uv run pytest -q
```

Run a group or a focused workflow.

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest tests/integration/test_prediction_workflow.py -q
```

List collected cases without running them.

```bash
uv run pytest --collect-only -q
```

Fixtures create fresh records and temporary outputs for each test. Shared
factories provide the small preparation and EDA tables. The separate
`evaluation_training` fixture includes matching records for split-isolation
checks. Synthetic final-model fixtures support inference and reload checks.

Test module filenames are distinct across the two groups, which supports
pytest's default import mode. New checks belong beside the component or
workflow they exercise.
