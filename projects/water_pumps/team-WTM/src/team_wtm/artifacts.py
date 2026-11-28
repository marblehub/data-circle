"""Identity and report checks shared by training, prediction and dashboard code."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4


def sha256_file(path: Path) -> str:
    """Hash file bytes, with bounded memory use."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def write_json(path, data):
    """Publish a complete JSON file, including when replacing a previous report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def text_digest(content):
    """Use LF for text checksums so Windows and Linux checkouts agree."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def report_ref(path):
    return {"file": path.name, "sha256": text_digest(path.read_bytes())}


def verified_report(manifest_path, name):
    """Read a linked table or JSON report, checking its contents before returning bytes."""
    manifest_path = Path(manifest_path)
    entry = read_json(manifest_path)["reports"][name]
    return _report_bytes(manifest_path.parent, entry)


def _report_bytes(directory, entry):
    if Path(entry["file"]).name != entry["file"]:
        raise ValueError("Expected a report filename in the manifest directory")
    content = (directory / entry["file"]).read_bytes()
    if text_digest(content) != entry["sha256"]:
        raise ValueError(f"Report checksum mismatch for {entry['file']}")
    return content


def evaluation_inputs(training_values, training_labels, split_plan):
    """Identify the inputs used for the reserved evaluation."""
    return {
        "training_values": text_digest(training_values.read_bytes()),
        "training_labels": text_digest(training_labels.read_bytes()),
        "split_plan": text_digest(split_plan.read_bytes()),
    }


def model_manifest(report, model_path, inputs, reports):
    return {
        **report,
        "run_id": uuid4().hex,
        "model_sha256": sha256_file(model_path),
        "evaluation_inputs": inputs,
        "reports": reports,
    }


def archive_run(manifest, model_path, report_dir, config, *, reused_holdout):
    """Keep a reloadable copy and the result for this successful final fit."""
    runs_dir = config.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=runs_dir) as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        shutil.copyfile(model_path, bundle / "model.joblib")
        write_json(bundle / "model.json", {**manifest, "model_file": "model.joblib"})
        for entry in manifest["reports"].values():
            (bundle / entry["file"]).write_bytes(_report_bytes(report_dir, entry))
        effective = {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()
        }
        write_json(bundle / "config_snapshot.json", effective)
        (bundle / "run.log").write_text(
            f"Final model run {manifest['run_id']}\n"
            f"Holdout result {'reused' if reused_holdout else 'scored'}\n"
            f"Fitted and verified reload on {manifest['fit_rows']} records\n"
            f"Model SHA-256 {manifest['model_sha256']}\n",
            encoding="utf-8",
        )
        bundle.rename(runs_dir / manifest["run_id"])


def read_prediction_check(manifest_path, check_path):
    """Return delivery totals only when they belong to the requested run and model."""
    manifest = read_json(manifest_path)
    check = read_json(check_path)
    for key in ("run_id", "model_sha256"):
        if not manifest.get(key) or check.get(key) != manifest[key]:
            raise ValueError(f"Prediction check has a different {key}.")
    return check
