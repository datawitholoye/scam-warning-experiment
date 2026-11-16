from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "simulation.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
