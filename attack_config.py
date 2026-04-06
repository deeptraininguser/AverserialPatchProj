"""
attack_config.py
================
Loads config.yaml and .env, and provides a single `cfg` dict plus helper
functions used across the pipeline.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (for Comet ML secrets)
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
_config_path = Path(__file__).resolve().parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load and return the YAML configuration dict."""
    path = Path(config_path) if config_path else _config_path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_on_remote(cfg: dict) -> bool:
    """Determine whether we are running on the remote machine.

    If ``cfg["on_remote"]`` is explicitly set, use that value.
    Otherwise auto-detect based on the current working directory.
    """
    val = cfg.get("on_remote")
    if val is not None:
        return bool(val)
    return os.getcwd() == os.environ.get("REMOTE_HOME", "/home/user")


def setup_environment(cfg: dict) -> bool:
    """``chdir`` to the remote project path when necessary.

    Returns the resolved *on_remote* flag.
    """
    on_remote = resolve_on_remote(cfg)
    if on_remote:
        remote_path = cfg.get("remote_project_path",
                              "/home/user/project/")
        os.chdir(remote_path)
    return on_remote


def get_comet_config(cfg: dict) -> dict:
    """Build a dict with Comet ML parameters (from config + .env)."""
    comet_cfg = cfg.get("comet", {})
    return {
        "enabled": comet_cfg.get("enabled", False),
        "api_key": comet_cfg.get("api_key") or os.getenv("COMET_API_KEY", ""),
        "project_name": comet_cfg.get("project_name") or os.getenv("COMET_PROJECT_NAME", ""),
        "workspace": comet_cfg.get("workspace") or os.getenv("COMET_WORKSPACE", ""),
    }
