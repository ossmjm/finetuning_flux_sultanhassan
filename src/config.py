"""Configuration: one YAML file describes one experiment.

- Dataclasses define the schema, so typos in a YAML key fail loudly.
- A config may set `base: baseline.yaml` and only override what differs, which keeps
  experiment files tiny and makes the difference between two runs obvious.
- `--set lora.rank=32` style overrides are applied last (handy for quick sweeps).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class RunConfig:
    name: str = "baseline"          # output folder: <output_root>/<name>
    output_root: str = "outputs"


@dataclass
class ModelConfig:
    name: str = "black-forest-labs/FLUX.1-dev"
    revision: str | None = None      # pin a HF commit hash for strict reproducibility
    variant: str | None = None
    max_sequence_length: int = 512   # T5 token length (512 for dev)
    quantization: str = "none"       # none | nf4  (nf4 = 4-bit base weights, QLoRA; for GPUs < 40 GB)


@dataclass
class DataConfig:
    dir: str = "data"
    trigger_word: str = "sltnhsn"
    require_trigger_word: bool = True   # only *checks* captions, never edits them
    image_exts: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    caption_ext: str = ".txt"
    resolution: int = 1024              # pixel BUDGET (resolution^2): larger images are downscaled, smaller never upscaled
    random_flip: bool = False           # off by default: the building is not mirror-symmetric
    repeats: int = 1
    num_workers: int = 2
    cache_latents: bool = True          # VAE runs once up front, then is only used for validation
    # Aspect ratio is always preserved (no square crop); sides are snapped to multiples of 16.
    # Note: text embeddings are always precomputed once (captions are fixed, text encoders frozen).


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 1000          # optimizer steps (after accumulation)
    seed: int = 42
    mixed_precision: str = "bf16"        # bf16 | fp16 | no
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    optimizer: str = "adamw"             # adamw | adamw8bit (needs bitsandbytes)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class FlowMatchingConfig:
    # Defaults mirror the official Diffusers script.
    weighting_scheme: str = "none"       # none | sigma_sqrt | logit_normal | mode | cosmap
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    guidance_scale: float = 3.5          # FLUX.1-dev is guidance-distilled; value fed during training


@dataclass
class LoraConfigSection:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    init_lora_weights: str = "gaussian"
    target_modules: list[str] = field(default_factory=lambda: [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    ])


@dataclass
class CheckpointingConfig:
    every_n_steps: int = 250
    save_final: bool = True


@dataclass
class ValidationConfig:
    every_n_steps: int = 250
    at_start: bool = True                # step_0 = base model reference (LoRA is a no-op at init)
    seed: int = 1234
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    # [height, width] pairs, cycled over prompts (trigger prompts first, then control prompts).
    # Mix portrait and landscape because the training data does.
    sizes: list[list[int]] = field(default_factory=lambda: [[768, 1024], [1024, 768]])
    images_per_prompt: int = 1
    prompts: list[str] = field(default_factory=lambda: [
        "sltnhsn, a photograph of the Mosque-Madrasa of Sultan Hassan in Cairo, monumental stone facade, midday light",
        "sltnhsn, the towering minaret and the main portal in golden hour light",
        "sltnhsn, a wide view of the mosque rising above the surrounding Cairo streets at sunset",
        "sltnhsn, a detailed close-up of the carved stonework and muqarnas ornamentation",
    ])
    # Prompts WITHOUT the trigger word: watch these for drift / loss of base-model behaviour.
    control_prompts: list[str] = field(default_factory=lambda: [
        "a photograph of a historic mosque with a tall minaret in Istanbul",
        "a red vintage car parked on a quiet European street",
    ])


@dataclass
class LoggingConfig:
    report_to: str = "wandb"             # wandb | none
    project_name: str = "flux-sultan-hassan"
    run_name: str | None = None          # None -> auto name that encodes rank/alpha/lr/batch/seed
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    lora: LoraConfigSection = field(default_factory=LoraConfigSection)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.run.output_root) / self.run.name

    @property
    def wandb_run_name(self) -> str:
        if self.logging.run_name:
            return self.logging.run_name
        t, l = self.training, self.lora
        eff_bs = t.batch_size * t.gradient_accumulation_steps
        return f"{self.run.name}_r{l.rank}_a{l.alpha}_lr{t.learning_rate:g}_bs{eff_bs}_s{t.seed}"


# ----------------------------------------------------------------------------- loading


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)  # lists (e.g. target_modules) are replaced, not merged
    return out


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if base is not None:
        base_path = (path.parent / base).resolve()
        data = _deep_merge(_read_yaml(base_path), data)
    return data


def _apply_override(data: dict, override: str) -> None:
    key, sep, raw = override.partition("=")
    if not sep:
        raise ValueError(f"Override must look like section.key=value, got: {override!r}")
    parts = key.strip().split(".")
    node = data
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def _coerce(hint, value, where: str):
    """YAML 1.1 reads `1e-4` as a *string*; convert numbers so users don't have to write 1.0e-4."""
    if hint is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"'{where}' must be a number, got {value!r}")
    if hint is int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"'{where}' must be an integer, got {value!r}")
    return value


def _build(cls, data: dict, where: str = ""):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) in '{where or 'root'}': {sorted(unknown)}. Valid: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value or {}, f"{where}.{name}".lstrip("."))
        else:
            kwargs[name] = _coerce(hint, value, f"{where}.{name}".lstrip("."))
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    data = _read_yaml(Path(path).resolve())
    for o in overrides or []:
        _apply_override(data, o)
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    d, t, l, v = cfg.data, cfg.training, cfg.lora, cfg.validation
    if d.cache_latents and d.random_flip:
        raise ValueError("data.cache_latents=true needs deterministic preprocessing (random_flip=false). "
                         "Disable cache_latents to use flip augmentation.")
    if t.batch_size != 1:
        raise ValueError("training.batch_size must be 1: images keep their own aspect ratio, so they cannot be "
                         "stacked into one tensor. Use training.gradient_accumulation_steps for a larger effective batch.")
    if not v.sizes or any(len(x) != 2 or x[0] % 16 or x[1] % 16 for x in v.sizes):
        raise ValueError("validation.sizes must be a list of [height, width] pairs that are multiples of 16")
    if l.rank <= 0 or l.alpha <= 0:
        raise ValueError("lora.rank and lora.alpha must be > 0")
    if not l.target_modules:
        raise ValueError("lora.target_modules must not be empty")
    if cfg.model.quantization not in {"none", "nf4"}:
        raise ValueError("model.quantization must be none or nf4")
    if t.mixed_precision not in {"bf16", "fp16", "no"}:
        raise ValueError("training.mixed_precision must be bf16, fp16 or no")
    if t.optimizer not in {"adamw", "adamw8bit"}:
        raise ValueError("training.optimizer must be adamw or adamw8bit")
    if cfg.logging.report_to not in {"wandb", "none"}:
        raise ValueError("logging.report_to must be wandb or none")
    if cfg.checkpointing.every_n_steps <= 0 or v.every_n_steps <= 0:
        raise ValueError("checkpointing.every_n_steps and validation.every_n_steps must be > 0")
    if not v.prompts:
        raise ValueError("validation.prompts must contain at least one prompt")
    missing = [p for p in v.prompts if d.trigger_word not in p]
    if missing:
        raise ValueError(f"validation.prompts must contain the trigger word '{d.trigger_word}'. Offending: {missing}")
