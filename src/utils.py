"""Small reusable helpers: dtype mapping, flattening configs, and reproducibility metadata."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import yaml


def weight_dtype_from(mixed_precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[mixed_precision]


def flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key + "."))
        else:
            out[key] = v
    return out


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _resolve_model_commit(name: str, revision: str | None) -> str | None:
    """The exact HF commit of the base model used (needs network; best-effort)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(name, revision=revision).sha
    except Exception:
        return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def save_run_metadata(cfg, run_dir: Path, num_images: int, data_fingerprint: str) -> dict:
    """Write everything needed to reproduce this run next to its outputs."""
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": cfg.model.name,
        "base_model_revision_requested": cfg.model.revision,
        "base_model_commit_resolved": _resolve_model_commit(cfg.model.name, cfg.model.revision),
        "seed": cfg.training.seed,
        "dataset_dir": str(Path(cfg.data.dir).resolve()),
        "dataset_num_images": num_images,
        "dataset_sha256": data_fingerprint,
        "lora": asdict(cfg.lora),
        "max_train_steps": cfg.training.max_train_steps,
        "project_git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {p: _version(p) for p in
                     ["torch", "diffusers", "transformers", "peft", "accelerate", "wandb", "bitsandbytes"]},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(info, indent=2))

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze.returncode == 0:
        (run_dir / "requirements.lock.txt").write_text(freeze.stdout)
    return info
