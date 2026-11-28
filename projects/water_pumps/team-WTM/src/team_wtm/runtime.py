"""Persistent progress logs and reproducible records for workflow invocations."""

import platform
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from team_wtm.artifacts import sha256_file, write_json
from team_wtm.config import PACKAGE_DIR, SETTINGS
from team_wtm.logging_config import workflow_logging


class WorkflowRun:
    """Collect aggregate results and progress for one workflow invocation."""

    def __init__(self, name, config, on_progress):
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex
        self.directory = config.workflow_runs_dir / self.run_id
        self.directory.mkdir(parents=True)
        self.on_progress = on_progress
        self.details = {
            "schema_version": 1,
            "working_directory": str(Path.cwd()),
            "run_id": self.run_id,
            "workflow": name,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "versions": {
                name: version(name)
                for name in ("pandas", "numpy", "scikit-learn", "joblib", "pyarrow")
            },
            "code_sha256": {
                path.relative_to(PACKAGE_DIR).as_posix(): sha256_file(path)
                for path in sorted(PACKAGE_DIR.rglob("*.py"))
            },
            "inputs": {},
            "outputs": {},
        }
        settings = {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(config).items()
        }
        write_json(self.directory / "config_snapshot.json", settings)
        write_json(self.directory / "run.json", self.details)

    def progress(self, message):
        self.logger.info(message)
        if self.on_progress is not None:
            self.on_progress(message)

    def record_inputs(self, paths):
        """Record exact file identities, including any reused result files."""
        for name, path in paths.items():
            path = Path(path).resolve()
            self.details["inputs"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        write_json(self.directory / "run.json", self.details)

    def archive(self, paths, root):
        """Copy aggregate reports while retaining their relative folders."""
        root = Path(root).resolve()
        for path in paths:
            path = Path(path).resolve()
            relative = path.relative_to(root)
            target = self.directory / "reports" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(path, target)
            self.details["outputs"][relative.as_posix()] = {
                "file": target.relative_to(self.directory).as_posix(),
                "sha256": sha256_file(target),
            }


@contextmanager
def track_workflow(name, *, config=None, on_progress=None, report_dir=None):
    """Record successful and failed workflows independently, including concurrent calls."""
    config = config or SETTINGS
    run = WorkflowRun(name, config, on_progress)
    with workflow_logging(
        name, run.run_id, run.directory / "run.log", config.log_level
    ) as logger:
        run.logger = logger
        lock = None
        try:
            if report_dir is not None:
                report_dir.mkdir(parents=True, exist_ok=True)
                candidate = report_dir / ".workflow.lock"
                try:
                    candidate.mkdir()
                except FileExistsError:
                    raise RuntimeError(
                        f"Another workflow is publishing reports at {report_dir}"
                    ) from None
                lock = candidate
            run.progress(f"Started {name}")
            yield run
            for entry in run.details["inputs"].values():
                if sha256_file(Path(entry["path"])) != entry["sha256"]:
                    raise RuntimeError(
                        f"Input changed during workflow: {entry['path']}"
                    )
            run.progress(f"Finished {name}")
        except BaseException as error:
            run.details["status"] = "failed"
            run.details["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            run.logger.exception("Workflow failed")
            raise
        else:
            if run.details["status"] == "running":
                run.details["status"] = "completed"
        finally:
            run.details["finished_at"] = datetime.now(UTC).isoformat()
            try:
                write_json(run.directory / "run.json", run.details)
            finally:
                if lock is not None:
                    lock.rmdir()
