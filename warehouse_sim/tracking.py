"""Optional MLflow experiment tracking.

`--mlflow` on calibrate.py / train.py logs parameters, per-step metrics and artefacts to ./mlruns
(view with `mlflow ui`). Without the flag, or without mlflow installed, every call is a no-op,
so the core project has no hard dependency on it.
"""
from __future__ import annotations

import contextlib


class Tracker:
    def __init__(self, enabled: bool, experiment: str):
        self.mlflow = None
        if enabled:
            import mlflow                      # imported only when asked for
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow

    @contextlib.contextmanager
    def run(self, name: str, **params):
        if self.mlflow is None:
            yield self
            return
        with self.mlflow.start_run(run_name=name):
            self.mlflow.log_params(params)
            yield self

    @contextlib.contextmanager
    def child(self, name: str, **params):
        """Nested run, e.g. one per baseline policy under a training run."""
        if self.mlflow is None:
            yield self
            return
        with self.mlflow.start_run(run_name=name, nested=True):
            if params:
                self.mlflow.log_params(params)
            yield self

    def metrics(self, values: dict, step: int | None = None, prefix: str = ""):
        if self.mlflow is not None:
            self.mlflow.log_metrics({prefix + k: float(v) for k, v in values.items()}, step=step)

    def tags(self, **tags):
        if self.mlflow is not None:
            self.mlflow.set_tags(tags)

    def artifact(self, path: str):
        if self.mlflow is not None:
            self.mlflow.log_artifact(path)

    def dict(self, d: dict, name: str):
        if self.mlflow is not None:
            self.mlflow.log_dict(d, name)
