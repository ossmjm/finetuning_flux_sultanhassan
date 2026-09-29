"""Validation: same prompts + same seeds for every experiment, saved per step.

Two prompt groups are generated:
  * "trigger"  - prompts containing the trigger word (does it learn the building? does it generalise?)
  * "control"  - prompts without it (has the base model drifted / been forgotten?)
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image

from .models import encode_prompts


@dataclass
class ValidationPrompts:
    prompts: list[str]
    kinds: list[str]                 # "trigger" | "control", aligned with prompts
    prompt_embeds: torch.Tensor      # [N, L, D] (CPU)
    pooled_embeds: torch.Tensor      # [N, 768]  (CPU)


def encode_validation_prompts(text_pipe: FluxPipeline, val_cfg, max_sequence_length: int, device) -> ValidationPrompts:
    prompts = list(val_cfg.prompts) + list(val_cfg.control_prompts)
    kinds = ["trigger"] * len(val_cfg.prompts) + ["control"] * len(val_cfg.control_prompts)
    embeds, pooled = encode_prompts(text_pipe, prompts, max_sequence_length, device)
    return ValidationPrompts(prompts, kinds, embeds, pooled)


def build_inference_pipeline(transformer, vae, scheduler) -> FluxPipeline:
    """Assemble a FluxPipeline from components already in memory (no text encoders needed
    because prompts are passed as precomputed embeddings)."""
    pipe = FluxPipeline(
        scheduler=copy.deepcopy(scheduler), vae=vae, transformer=transformer,
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_samples(pipeline: FluxPipeline, vp: ValidationPrompts, val_cfg, out_dir: str | Path,
                     device, autocast_dtype: torch.dtype | None = torch.bfloat16) -> list[dict]:
    """Generate every validation image and save it under `out_dir`.

    Seeds are `validation.seed + j` for the j-th image of every prompt, on a CPU generator, so
    the initial noise is identical across prompts, checkpoints, experiments and GPUs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    results = []
    for i, (prompt, kind) in enumerate(zip(vp.prompts, vp.kinds)):
        for j in range(val_cfg.images_per_prompt):
            seed = val_cfg.seed + j
            generator = torch.Generator("cpu").manual_seed(seed)
            ctx = (torch.autocast(device.type, dtype=autocast_dtype)
                   if autocast_dtype is not None else torch.autocast(device.type, enabled=False))
            with ctx:
                image = pipeline(
                    prompt_embeds=vp.prompt_embeds[i:i + 1].to(device),
                    pooled_prompt_embeds=vp.pooled_embeds[i:i + 1].to(device),
                    height=val_cfg.height, width=val_cfg.width,
                    num_inference_steps=val_cfg.num_inference_steps,
                    guidance_scale=val_cfg.guidance_scale,
                    generator=generator,
                ).images[0]
            path = out_dir / f"{i:02d}_{kind}_seed{seed}.png"
            image.save(path)
            results.append({"path": str(path), "prompt": prompt, "kind": kind, "seed": seed, "image": image})

    (out_dir / "prompts.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "image"} for r in results], indent=2))
    _save_grid([r["image"] for r in results], out_dir / "grid.jpg")
    return results


def _save_grid(images: list[Image.Image], path: Path, cols: int = 3, thumb: int = 384) -> None:
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for k, im in enumerate(images):
        grid.paste(im.resize((thumb, thumb)), ((k % cols) * thumb, (k // cols) * thumb))
    grid.save(path, quality=90)


def log_samples_to_wandb(accelerator, results: list[dict], step: int) -> None:
    import wandb

    run = accelerator.get_tracker("wandb", unwrap=True)
    payload = {}
    for kind in ("trigger", "control"):
        imgs = [wandb.Image(r["image"], caption=f"[step {step}] {r['prompt']} (seed {r['seed']})")
                for r in results if r["kind"] == kind]
        if imgs:
            payload[f"validation/{kind}"] = imgs
    run.log(payload, step=step)
