"""Configuration — all settings from environment variables."""

import os
import torch


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _auto_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


HOST: str = os.environ.get("UASR_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("UASR_PORT", "9000"))

_device_env = os.environ.get("UASR_DEVICE", "auto")
DEVICE: str = _auto_device() if _device_env == "auto" else _device_env

_compute_env = os.environ.get("UASR_COMPUTE_TYPE", "auto")
COMPUTE_TYPE: str = _auto_compute_type(DEVICE) if _compute_env == "auto" else _compute_env

DEFAULT_MODEL: str = os.environ.get("UASR_DEFAULT_MODEL", "base")
DATA_DIR: str = os.environ.get("UASR_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
MODELS_DIR: str = os.path.join(DATA_DIR, "models")
