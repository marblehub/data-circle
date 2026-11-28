"""Command-line interface for the reusable comparison and prediction workflows."""

import argparse
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


def _predict(args):
    from team_wtm.predict import CHECK_REPORT_PATH, relative, run_prediction

    result = run_prediction(args.records, args.model, args.output, args.format)
    rows = result["prediction_rows"]
    if "checks" not in result:
        print(f"Predicted {rows} records; saved to {args.output}")
        return
    print(f"Predicted {rows} records.")
    for name, passed in result["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Label counts: {result['label_counts']}")
    print(f"Saved predictions to {relative(args.output)}")
    print(f"Saved totals-only report to {relative(CHECK_REPORT_PATH)}")
    if not result["all_checks_passed"]:
        raise SystemExit("Completion check failed.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="team-wtm", description="Water pump workflows"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare models on development folds")
    compare.add_argument(
        "experiment",
        nargs="?",
        default="models",
        choices=("models", "quantity", "location"),
        help="Compare configured models, water quantity, or location columns.",
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

    args = parser.parse_args(argv)
    if args.command == "compare":
        from team_wtm.model_comparison import run_comparison

        table = run_comparison(args.experiment, on_progress=print)
        print_comparison(table, SETTINGS.report_dir)
    else:
        _predict(args)
