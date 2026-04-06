"""
experiment_tracking.py
======================
Optional wrapper around Comet ML.  When Comet is disabled the tracker
becomes a no-op so the rest of the code never has to check.
"""

from __future__ import annotations
from typing import Any


class _NoOpExperiment:
    """Drop-in replacement that silently ignores all Comet calls."""

    def __getattr__(self, _name: str):
        return lambda *a, **kw: None

    def get_key(self) -> str:
        return "noop"


class ExperimentTracker:
    """Thin facade over ``comet_ml``.

    Parameters
    ----------
    comet_cfg : dict
        The dict returned by ``attack_config.get_comet_config(cfg)``.
    """

    def __init__(self, comet_cfg: dict):
        self.enabled = comet_cfg.get("enabled", False)
        self._cfg = comet_cfg
        self.experiment: Any = _NoOpExperiment()
        self.experiment_key: str = "noop"

    # ------------------------------------------------------------------
    def start(self) -> "ExperimentTracker":
        """Start (or re-attach to) a Comet experiment."""
        if not self.enabled:
            print("[ExperimentTracker] Comet ML disabled – using no-op tracker.")
            return self
        try:
            from comet_ml import start
            self.experiment = start(
                api_key=self._cfg["api_key"],
                project_name=self._cfg["project_name"],
                workspace=self._cfg["workspace"],
            )
            self.experiment_key = self.experiment.get_key()
            print(f"[ExperimentTracker] Comet experiment key: {self.experiment_key}")
        except Exception as exc:
            print(f"[ExperimentTracker] Failed to start Comet: {exc} – falling back to no-op.")
            self.experiment = _NoOpExperiment()
        return self

    def reattach(self) -> "ExperimentTracker":
        """End the current experiment and re-attach (for code capture)."""
        if not self.enabled:
            return self
        try:
            self.experiment.end()
            from comet_ml import ExistingExperiment
            self.experiment = ExistingExperiment(
                api_key=self._cfg["api_key"],
                previous_experiment=self.experiment_key,
            )
            print("[ExperimentTracker] Re-attached to experiment.")
        except Exception as exc:
            print(f"[ExperimentTracker] Re-attach failed: {exc}")
        return self

    def end(self):
        if self.enabled:
            try:
                self.experiment.end()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------
    def log_parameters(self, params: dict):
        self.experiment.log_parameters(params)

    def log_metric(self, name: str, value, step: int | None = None):
        self.experiment.log_metric(name, value, step=step)

    def log_image(self, image, name: str = "", step: int | None = None):
        self.experiment.log_image(image, name=name, step=step)

    def log_asset(self, path: str):
        self.experiment.log_asset(path)

    def set_name(self, name: str):
        self.experiment.set_name(name)
