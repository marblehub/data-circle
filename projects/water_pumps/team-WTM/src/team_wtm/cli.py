"""Terminal parsing and presentation for the reusable project workflows."""

import argparse
import json
import logging
from pathlib import Path

from team_wtm import logging_config
from team_wtm.config import SETTINGS

logger = logging.getLogger(__name__)


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


def _compare(args):
    from team_wtm.model_comparison import run_comparison

    table = run_comparison(args.experiment)
    print_comparison(table, SETTINGS.report_dir)


def _baseline(args):
    from team_wtm.baseline import run_baseline_workflow

    run_baseline_workflow()


def _final_model(args):
    from team_wtm.final_model import run_final_model

    run_final_model()


def _features(args):
    from team_wtm.feature_checks import run_feature_checks

    result = run_feature_checks(args.experiment)
    if args.experiment == "charts":
        return
    if isinstance(result, dict):
        print(result["age"].round(3).to_string(index=False))
    elif args.experiment == "regions":
        print(result.round(3).to_string(index=False))
    else:
        print_comparison(result, SETTINGS.report_dir / "feature_checks")


def _scores(args):
    from team_wtm.per_class_scores import markdown_table, run_per_class_report

    print(markdown_table(run_per_class_report()))


def _export(args):
    from team_wtm.export_cleaned import export_cleaned_dataset

    metadata = export_cleaned_dataset(
        args.values,
        args.labels,
        args.output,
        overwrite=args.overwrite,
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
    from team_wtm.predict import run_prediction

    result = run_prediction(
        args.records,
        args.model,
        args.output,
        args.format,
        model_report=args.model_report,
        check_report=args.check_report,
    )
    if "checks" in result and not result["all_checks_passed"]:
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
    logging_config.configure_logging(SETTINGS)
    logger.info("Running command %s", args.command)
    try:
        handlers[args.command](args)
    except Exception as error:
        logger.error("Command %s failed: %s", args.command, error)
        raise
    logger.info("Command %s completed", args.command)
