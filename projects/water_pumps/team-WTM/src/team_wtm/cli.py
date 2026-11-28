"""Command-line interface for the reusable comparison and prediction workflows."""

import argparse
from pathlib import Path

from team_wtm.config import SETTINGS


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

        run_comparison(args.experiment)
    else:
        from team_wtm.predict import run_prediction

        run_prediction(args.records, args.model, args.output, args.format)
