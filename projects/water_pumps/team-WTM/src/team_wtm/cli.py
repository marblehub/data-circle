"""Terminal parsing and presentation for the reusable project workflows."""

import argparse
import json
from pathlib import Path

from team_wtm.config import SETTINGS


def print_comparison(table, report_dir):
    """Display the saved comparison in the terminal."""
    shown = [
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "repair_recall_mean",
        "repair_precision_mean",
        "missed_repairs_mean",
        "false_repair_alarms_mean",
    ]
    print(table[shown].round(4).to_string())
    print(f"Saved results to {report_dir}")
    print("The labelled holdout was not scored.")


def _compare(args):
    from team_wtm.model_comparison import run_comparison

    table = run_comparison(args.experiment, on_progress=print)
    print_comparison(table, SETTINGS.report_dir)


def _baseline(args):
    from team_wtm.baseline import REPORT_DIR, SPLIT_PATH, run_baseline_workflow

    summary = run_baseline_workflow(on_progress=print)
    print(f"Saved baseline results to {REPORT_DIR / 'baseline'}")
    print(f"Local split plan: {SPLIT_PATH}")
    print(f"Accuracy: {summary['cv_mean']['accuracy']:.4f}")
    print(f"Macro F1: {summary['cv_mean']['macro_f1']:.4f}")
    print("The labelled holdout was not scored.")


def _final_model(args):
    from team_wtm.final_model import run_final_model

    result = run_final_model(on_progress=print)
    scores = result["holdout"]
    print(f"Model run: {result['manifest']['run_id']}")
    print(
        f"Holdout macro F1: {scores['macro_f1']:.4f}  accuracy: {scores['accuracy']:.4f}"
    )
    print(
        f"Holdout repair recall: {scores['repair_recall']:.4f}  precision: {scores['repair_precision']:.4f}"
    )
    print(f"Reload check: {result['manifest']['reload_check']}")


def _features(args):
    from team_wtm.feature_checks import run_feature_checks

    result = run_feature_checks(args.experiment, on_progress=print)
    if args.experiment == "charts":
        for path in result:
            print(f"Saved {path}")
    elif isinstance(result, dict):
        print(result["age"].round(3).to_string(index=False))
    elif args.experiment == "regions":
        print(result.round(3).to_string(index=False))
    else:
        print_comparison(result, SETTINGS.report_dir / "feature_checks")


def _scores(args):
    from team_wtm.per_class_scores import (
        OUTPUT_PATH,
        markdown_table,
        run_per_class_report,
    )

    summary = run_per_class_report(on_progress=print)
    scores = summary.attrs["macro_f1"]
    print(f"Macro F1: {scores['mean']:.4f} ± {scores['std']:.4f}")
    print(markdown_table(summary))
    print(f"Saved {OUTPUT_PATH}")


def _export(args):
    from team_wtm.export_cleaned import export_cleaned_dataset

    metadata = export_cleaned_dataset(
        args.values,
        args.labels,
        args.output,
        overwrite=args.overwrite,
        on_progress=print,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "metadata": str(args.output.resolve().with_suffix(".metadata.json")),
                "rows": metadata["output"]["rows"],
                "columns": metadata["output"]["columns"],
                "flag_counts": metadata["quality"]["flag_counts"],
                "git_commit": metadata["code"]["git_commit"],
                "source_files_dirty": metadata["code"]["source_files_dirty"],
            },
            indent=2,
        )
    )


def _predict(args):
    from team_wtm.predict import relative, run_prediction

    result = run_prediction(
        args.records,
        args.model,
        args.output,
        args.format,
        model_report=args.model_report,
        check_report=args.check_report,
        on_progress=print,
    )
    rows = result["prediction_rows"]
    if "checks" not in result:
        print(f"Predicted {rows} records; saved to {args.output}")
        return
    print(f"Predicted {rows} records.")
    for name, passed in result["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Label counts: {result['label_counts']}")
    print(f"Saved predictions to {relative(args.output)}")
    print(f"Saved totals-only report to {result['report_file']}")
    if not result["all_checks_passed"]:
        raise SystemExit("Completion check failed.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="team-wtm", description="Water pump workflows"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "baseline", help="Create the split plan and majority-class reference scores"
    )
    commands.add_parser(
        "final-model",
        help="Check reserved evaluation reuse, refit and save the configured model",
    )
    commands.add_parser(
        "scores", help="Summarize each pump condition using saved fold scores"
    )
    compare = commands.add_parser("compare", help="Compare models on development folds")
    compare.add_argument(
        "experiment",
        nargs="?",
        default="models",
        choices=("models", "quantity", "location"),
        help="Compare configured models, water quantity, or location columns.",
    )
    features = commands.add_parser(
        "features", help="Evaluate column groups and transfer to new regions"
    )
    features.add_argument(
        "experiment", choices=("ladder", "h6", "groups", "regions", "charts")
    )
    export = commands.add_parser(
        "export-cleaned", help="Save a cleaned Parquet snapshot and provenance"
    )
    export.add_argument("--values", type=Path, default=SETTINGS.training_values)
    export.add_argument("--labels", type=Path, default=SETTINGS.training_labels)
    export.add_argument("--output", type=Path, default=SETTINGS.cleaned_snapshot)
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing snapshot and metadata.",
    )
    predict = commands.add_parser(
        "predict", help="Predict pump status using a saved model"
    )
    predict.add_argument(
        "--records",
        type=Path,
        help="CSV without status_group. Omit to check predictions on the supplied test records.",
    )
    predict.add_argument(
        "--model",
        type=Path,
        default=SETTINGS.model_file,
        help="Saved pipeline to load (default: %(default)s).",
    )
    predict.add_argument(
        "--model-report",
        type=Path,
        help="Matching model manifest; custom models can use a .json sidecar.",
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=SETTINGS.predictions_file,
        help="Destination for predictions (default: %(default)s).",
    )
    predict.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output file format (default: %(default)s).",
    )
    predict.add_argument(
        "--check-report",
        type=Path,
        help="Destination for supplied-test completion checks. Useful with a temporary prediction output.",
    )
    args = parser.parse_args(argv)
    handlers = {
        "compare": _compare,
        "predict": _predict,
        "baseline": _baseline,
        "final-model": _final_model,
        "features": _features,
        "scores": _scores,
        "export-cleaned": _export,
    }
    handlers[args.command](args)
