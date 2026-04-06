"""
classifier_loader.py
====================
Dynamically loads a classifier module based on the name from config.yaml.
Returns the same interface (predict_raw, orig_clases, model, weights) that
the rest of the pipeline expects.
"""

from __future__ import annotations
import importlib
from functools import partial
from typing import Callable


def load_classifier(module_name: str) -> dict:
    """Import *module_name* and return its public symbols as a dict.

    Returns
    -------
    dict with keys:
        - predict_raw : Callable
        - model        : the underlying torch model (or model name string)
        - weights      : weight enum (for ``weights.meta["categories"]``)
        - orig_clases  : tensor of forbidden class indices
    """
    mod = importlib.import_module(module_name)
    return {
        "predict_raw": mod.predict_raw,
        "model": mod.model,
        "weights": mod.weights,
        "orig_clases": getattr(mod, "orig_clases", None),
    }


def setup_classifiers(cfg: dict) -> dict:
    """Load train / dev / test classifiers according to config.

    Also applies ensemble weight overrides when relevant.

    Returns
    -------
    dict with keys:
        predict_raw, predict_raw_dev, predict_raw_test,
        model, model_dev, model_test,
        weights, orig_clases, model_name
    """
    train_clf = load_classifier(cfg["classifier_train"])
    dev_clf = load_classifier(cfg["classifier_dev"])
    test_clf = load_classifier(cfg["classifier_test"])

    predict_raw = train_clf["predict_raw"]
    model = train_clf["model"]
    weights = train_clf["weights"]
    orig_clases = train_clf["orig_clases"]

    model_name = model if isinstance(model, str) else model.__class__.__name__

    # If the train classifier exposes a weights_dict parameter, apply ensemble
    # weights from config.
    if "weights_dict" in predict_raw.__code__.co_varnames:
        ew = cfg.get("ensemble_weights", {})
        predict_raw = partial(predict_raw, weights_dict=ew)

    return {
        "predict_raw": predict_raw,
        "predict_raw_dev": dev_clf["predict_raw"],
        "predict_raw_test": test_clf["predict_raw"],
        "model": model,
        "model_dev": dev_clf["model"],
        "model_test": test_clf["model"],
        "weights": weights,
        "orig_clases": orig_clases,
        "model_name": model_name,
    }
