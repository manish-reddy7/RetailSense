"""MLflow helpers (blueprint 10.2): config hash, subset hash and git commit on every run."""

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow

from retailsense.config import ROOT, ExperimentCfg


def cfg_path() -> Path:
    return ROOT / os.environ.get("RETAILSENSE_CONFIG", "configs/experiment.yaml")


def file_hash(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "uncommitted"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: v})
    return out


def start_run(cfg: ExperimentCfg, experiment: str, run_name: str):
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(ROOT / uri.removeprefix('sqlite:///')).as_posix()}"
    mlflow.set_tracking_uri(uri)
    experiment = f"{cfg.dataset}_{experiment}"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)
    configs = ROOT / "configs"
    mlflow.set_tags(
        {
            "git_commit": git_commit(),
            "config_hash": file_hash(cfg_path(), configs / "inventory.yaml"),
            "subset_hash": file_hash(cfg.path("subset")),
            "dataset": cfg.dataset,
        }
    )
    mlflow.log_params(_flatten(cfg.model_dump(mode="json")))
    return run
