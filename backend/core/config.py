"""
Typed application settings and startup validation for backend services.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when backend settings are invalid."""


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_repo_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "backend").is_dir():
            return parent
    return Path.cwd()


def _resolve_path(raw_path: str | None, repo_root: Path, config_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    raw_path = raw_path.strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    candidates = [
        (repo_root / candidate).resolve(),
        (config_dir / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    ]
    for resolved in candidates:
        if resolved.exists():
            return resolved
    return candidates[0]


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(raw)
    cfg.setdefault("models", {})
    cfg.setdefault("datasets", {})

    scalar_overrides = {
        "device": os.getenv("AIVD_DEVICE"),
        "output_dir": os.getenv("AIVD_OUTPUT_DIR"),
        "temp_dir": os.getenv("AIVD_TEMP_DIR"),
    }
    for key, value in scalar_overrides.items():
        if value:
            cfg[key] = value

    batch_size = os.getenv("AIVD_BATCH_SIZE")
    if batch_size:
        cfg["batch_size"] = int(batch_size)

    frame_skip = os.getenv("AIVD_FRAME_SKIP")
    if frame_skip:
        cfg["frame_skip"] = int(frame_skip)

    cfg["models"].setdefault("efficientnet_b1", {})
    cfg["models"].setdefault("xceptionnet", {})
    cfg["models"].setdefault("mesonet", {})
    cfg["models"].setdefault("aasist", {})

    eff_weights = os.getenv("AIVD_EFFICIENTNET_WEIGHTS_PATH")
    if eff_weights:
        cfg["models"]["efficientnet_b1"]["weights_path"] = eff_weights

    xcp_weights = os.getenv("AIVD_XCEPTION_WEIGHTS_PATH")
    if xcp_weights:
        cfg["models"]["xceptionnet"]["weights_path"] = xcp_weights

    meso_weights = os.getenv("AIVD_MESONET_WEIGHTS_PATH")
    if meso_weights:
        cfg["models"]["mesonet"]["weights_path"] = meso_weights

    meso_env = os.getenv("AIVD_MESONET_ENV_PATH")
    if meso_env:
        cfg["models"]["mesonet"]["env_path"] = meso_env

    aasist_weights = os.getenv("AIVD_AASIST_WEIGHTS_PATH")
    if aasist_weights:
        cfg["models"]["aasist"]["weights_path"] = aasist_weights

    fakeav_metadata = os.getenv("AIVD_FAKEAVCELEB_METADATA_PATH")
    if fakeav_metadata:
        cfg["datasets"].setdefault("FakeAVCeleb", {})
        cfg["datasets"]["FakeAVCeleb"]["metadata"] = fakeav_metadata

    return cfg


def _resolve_dataset_paths(
    datasets: dict[str, Any], repo_root: Path, config_dir: Path
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for dataset_name, dataset_cfg in datasets.items():
        if not isinstance(dataset_cfg, dict):
            resolved[dataset_name] = {}
            continue
        normalized = deepcopy(dataset_cfg)
        for key, value in normalized.items():
            if isinstance(value, str) and value.strip():
                if key.endswith("_path") or key.endswith("_dir") or key == "metadata":
                    path = _resolve_path(value, repo_root, config_dir)
                    normalized[key] = str(path) if path is not None else value
        resolved[dataset_name] = normalized
    return resolved


def _validate_existing_file(path_value: str | None, field_name: str, errors: list[str]) -> None:
    if not path_value:
        errors.append(f"{field_name} is required")
        return
    path = Path(path_value)
    if not path.is_file():
        errors.append(f"{field_name} not found: {path}")


@dataclass(frozen=True)
class MTCNNSettings:
    margin: int
    min_face_size: int
    thresholds: list[float]
    factor: float
    post_process: bool
    select_largest: bool
    keep_all: bool


@dataclass(frozen=True)
class AppSettings:
    config_path: str
    repo_root: str
    device: str
    batch_size: int
    frame_skip: int
    output_dir: str
    temp_dir: str
    compare_baseline_accuracy: bool
    datasets: dict[str, dict[str, Any]]
    models: dict[str, dict[str, Any]]
    mtcnn: MTCNNSettings

    @classmethod
    def from_yaml(cls, config_path: str, validate_paths: bool = True) -> "AppSettings":
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            raise ConfigValidationError(f"Config file not found: {config_file}")

        with config_file.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        raw = _apply_env_overrides(raw)
        repo_root = _find_repo_root(config_file)
        config_dir = config_file.parent

        models = deepcopy(raw.get("models", {}))
        datasets = _resolve_dataset_paths(raw.get("datasets", {}), repo_root, config_dir)

        for model_name in ("efficientnet_b1", "xceptionnet", "mesonet", "aasist"):
            if model_name in models and isinstance(models[model_name], dict):
                weights_path = models[model_name].get("weights_path")
                resolved_weight = _resolve_path(weights_path, repo_root, config_dir)
                if resolved_weight is not None:
                    models[model_name]["weights_path"] = str(resolved_weight)

        if "mesonet" in models and isinstance(models["mesonet"], dict):
            mesonet_env = models["mesonet"].get("env_path")
            if mesonet_env:
                env_value = str(mesonet_env).strip()
                # If this is an executable token (e.g., "python3"), keep as-is
                # for PATH lookup at runtime/validation.
                if any(ch in env_value for ch in ["/", "\\"]) or env_value.startswith("."):
                    resolved_env = _resolve_path(env_value, repo_root, config_dir)
                    if resolved_env is not None:
                        models["mesonet"]["env_path"] = str(resolved_env)
                else:
                    models["mesonet"]["env_path"] = env_value

        output_dir = _resolve_path(raw.get("output_dir", "./outputs/ensemble_results/"), repo_root, config_dir)
        temp_dir = _resolve_path(raw.get("temp_dir", "./outputs/tmp/"), repo_root, config_dir)

        mtcnn_cfg = raw.get("mtcnn") or {}
        mtcnn = MTCNNSettings(
            margin=int(mtcnn_cfg.get("margin", 50)),
            min_face_size=int(mtcnn_cfg.get("min_face_size", 100)),
            thresholds=list(mtcnn_cfg.get("thresholds", [0.6, 0.7, 0.7])),
            factor=float(mtcnn_cfg.get("factor", 0.7)),
            post_process=bool(mtcnn_cfg.get("post_process", True)),
            select_largest=bool(mtcnn_cfg.get("select_largest", True)),
            keep_all=bool(mtcnn_cfg.get("keep_all", True)),
        )

        settings = cls(
            config_path=str(config_file),
            repo_root=str(repo_root),
            device=str(raw.get("device", "cpu")),
            batch_size=int(raw.get("batch_size", 60)),
            frame_skip=int(raw.get("frame_skip", 30)),
            output_dir=str(output_dir),
            temp_dir=str(temp_dir),
            compare_baseline_accuracy=bool(raw.get("compare_baseline_accuracy", False)),
            datasets=datasets,
            models=models,
            mtcnn=mtcnn,
        )
        settings.validate(validate_paths=validate_paths)
        return settings

    def validate(self, validate_paths: bool = True) -> None:
        errors: list[str] = []

        if self.device not in {"cpu", "cuda"}:
            errors.append(f"device must be 'cpu' or 'cuda', got '{self.device}'")
        if self.batch_size <= 0:
            errors.append("batch_size must be > 0")
        if self.frame_skip <= 0:
            errors.append("frame_skip must be > 0")

        if self.mtcnn.min_face_size <= 0:
            errors.append("mtcnn.min_face_size must be > 0")
        if len(self.mtcnn.thresholds) != 3:
            errors.append("mtcnn.thresholds must contain 3 values")

        for directory in (Path(self.output_dir), Path(self.temp_dir)):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                errors.append(f"unable to create directory '{directory}': {exc}")

        if validate_paths:
            eff_cfg = self.models.get("efficientnet_b1", {})
            if eff_cfg.get("active", True):
                _validate_existing_file(
                    eff_cfg.get("weights_path"),
                    "models.efficientnet_b1.weights_path",
                    errors,
                )

            xcp_cfg = self.models.get("xceptionnet", {})
            if xcp_cfg.get("active", True):
                _validate_existing_file(
                    xcp_cfg.get("weights_path"),
                    "models.xceptionnet.weights_path",
                    errors,
                )

            meso_cfg = self.models.get("mesonet", {})
            if meso_cfg.get("active", True):
                _validate_existing_file(
                    meso_cfg.get("weights_path"),
                    "models.mesonet.weights_path",
                    errors,
                )
                mesonet_env_path = meso_cfg.get("env_path")
                if not mesonet_env_path:
                    errors.append("models.mesonet.env_path is required")
                else:
                    env_value = str(mesonet_env_path)
                    # Support both absolute interpreter paths and executable names
                    # such as "python3" for portable setups.
                    if any(ch in env_value for ch in ["/", "\\"]) or env_value.startswith("."):
                        _validate_existing_file(
                            env_value,
                            "models.mesonet.env_path",
                            errors,
                        )
                    elif shutil.which(env_value) is None:
                        errors.append(
                            f"models.mesonet.env_path executable not found in PATH: {env_value}"
                        )

            if self.compare_baseline_accuracy:
                metadata_path = self.datasets.get("FakeAVCeleb", {}).get("metadata")
                _validate_existing_file(
                    metadata_path,
                    "datasets.FakeAVCeleb.metadata",
                    errors,
                )

        if errors:
            formatted = "\n".join(f"- {error}" for error in errors)
            raise ConfigValidationError(f"Invalid backend settings:\n{formatted}")

    def to_runtime_config(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "frame_skip": self.frame_skip,
            "output_dir": self.output_dir,
            "temp_dir": self.temp_dir,
            "compare_baseline_accuracy": self.compare_baseline_accuracy,
            "datasets": deepcopy(self.datasets),
            "models": deepcopy(self.models),
            "mtcnn": {
                "margin": self.mtcnn.margin,
                "min_face_size": self.mtcnn.min_face_size,
                "thresholds": list(self.mtcnn.thresholds),
                "factor": self.mtcnn.factor,
                "post_process": self.mtcnn.post_process,
                "select_largest": self.mtcnn.select_largest,
                "keep_all": self.mtcnn.keep_all,
            },
        }


def load_settings(config_path: str, validate_paths: bool = True) -> AppSettings:
    """Load typed app settings with environment overrides and validation."""
    env_validate = _parse_bool(os.getenv("AIVD_VALIDATE_PATHS"), validate_paths)
    return AppSettings.from_yaml(config_path=config_path, validate_paths=env_validate)
