"""FLUX LoRA training, adapted from diffusers/examples/dreambooth/train_dreambooth_lora_flux.py.

Reused as-is from Diffusers: FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, FluxPipeline
latent helpers (_pack/_unpack/_prepare_latent_image_ids), timestep sampling + loss weighting from
`diffusers.training_utils`, PEFT LoRA, `FluxPipeline.save_lora_weights`.

The flow-matching step in `flow_matching_loss` follows the official script line by line.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import models, validation
from .dataset import CaptionedImageDataset, dataset_fingerprint, find_samples
from .utils import flatten_dict, save_run_metadata, weight_dtype_from

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- latents


def sample_latents(vae_params: torch.Tensor, vae) -> torch.Tensor:
    """Sample from the VAE posterior (mean/logvar concatenated on the channel dim) and
    apply FLUX's shift/scale. Identical maths to `latent_dist.sample()` in the official script."""
    mean, logvar = vae_params.chunk(2, dim=1)
    logvar = logvar.clamp(-30.0, 20.0)
    latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


@torch.no_grad()
def encode_pixels(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    return vae.encode(pixel_values.to(device=vae.device, dtype=vae.dtype)).latent_dist.parameters


@torch.no_grad()
def cache_vae_params(vae, dataset: CaptionedImageDataset) -> list[torch.Tensor]:
    """One [1, 2C, h, w] tensor per image (a list, because every image can have a different size)."""
    out = []
    for i in tqdm(range(dataset.num_unique), desc="Caching latents"):
        pixels = dataset.get_pixels(i)
        logger.info("[%02d/%d] %s -> %dx%d", i + 1, dataset.num_unique, dataset.samples[i][0].name,
                    pixels.shape[2], pixels.shape[1])
        out.append(encode_pixels(vae, pixels.unsqueeze(0)).cpu())
    return out


# ----------------------------------------------------------------------------- loss


def flow_matching_loss(transformer, latents, prompt_embeds, pooled_embeds, text_ids, scheduler,
                       fm_cfg, vae_scale_factor: int, device, dtype) -> torch.Tensor:
    """One rectified-flow training step: x_t = (1 - sigma) * x0 + sigma * noise, target = noise - x0."""
    bsz, _, h, w = latents.shape
    latents = latents.to(dtype)

    noise = torch.randn_like(latents)
    u = compute_density_for_timestep_sampling(
        weighting_scheme=fm_cfg.weighting_scheme, batch_size=bsz,
        logit_mean=fm_cfg.logit_mean, logit_std=fm_cfg.logit_std, mode_scale=fm_cfg.mode_scale,
    )
    indices = (u * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device)

    # sigma for the sampled timesteps
    schedule_timesteps = scheduler.timesteps.to(device)
    step_idx = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[step_idx].flatten()
    while sigmas.ndim < latents.ndim:
        sigmas = sigmas.unsqueeze(-1)

    noisy = (1.0 - sigmas) * latents + sigmas * noise
    packed = FluxPipeline._pack_latents(noisy, batch_size=bsz, num_channels_latents=latents.shape[1],
                                        height=h, width=w)
    img_ids = FluxPipeline._prepare_latent_image_ids(bsz, h // 2, w // 2, device, dtype)

    guidance = None
    if transformer.config.guidance_embeds:  # FLUX.1-dev is guidance-distilled
        guidance = torch.full((bsz,), fm_cfg.guidance_scale, device=device)

    pred = transformer(
        hidden_states=packed,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]
    pred = FluxPipeline._unpack_latents(pred, height=h * vae_scale_factor, width=w * vae_scale_factor,
                                        vae_scale_factor=vae_scale_factor)

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=fm_cfg.weighting_scheme, sigmas=sigmas)
    target = noise - latents
    loss = torch.mean((weighting.float() * (pred.float() - target.float()) ** 2).reshape(bsz, -1), 1)
    return loss.mean()


# ----------------------------------------------------------------------------- helpers


def build_optimizer(params, t_cfg):
    kwargs = dict(lr=t_cfg.learning_rate, betas=(t_cfg.adam_beta1, t_cfg.adam_beta2),
                  weight_decay=t_cfg.adam_weight_decay, eps=t_cfg.adam_epsilon)
    if t_cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, **kwargs)
    return torch.optim.AdamW(params, **kwargs)


def save_lora(accelerator, transformer, path: Path, cfg) -> None:
    """Save ONLY the LoRA adapter (never the 12B base model), with PEFT metadata so that
    alpha != rank is restored correctly when loading with `pipeline.load_lora_weights`."""
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    path.mkdir(parents=True, exist_ok=True)
    FluxPipeline.save_lora_weights(
        save_directory=path,
        transformer_lora_layers=get_peft_model_state_dict(unwrapped),
        transformer_lora_adapter_metadata=unwrapped.peft_config["default"].to_dict(),
    )
    (path / "training_info.json").write_text(json.dumps({
        "run": cfg.run.name, "lora": asdict(cfg.lora), "base_model": cfg.model.name,
        "trigger_word": cfg.data.trigger_word, "seed": cfg.training.seed,
    }, indent=2))


def run_validation(accelerator, transformer, vae, scheduler, vp, cfg, step: int, run_dir: Path) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    was_training = unwrapped.training
    unwrapped.eval()
    pipe = validation.build_inference_pipeline(unwrapped, vae, scheduler)
    results = validation.generate_samples(
        pipe, vp, cfg.validation, run_dir / "samples" / f"step_{step}", accelerator.device,
        autocast_dtype=None if cfg.training.mixed_precision == "no" else weight_dtype_from(cfg.training.mixed_precision),
    )
    if cfg.logging.report_to == "wandb":
        validation.log_samples_to_wandb(accelerator, results, step)
    del pipe, results
    free_memory()
    unwrapped.train(was_training)


# ----------------------------------------------------------------------------- main


def train(cfg, force: bool = False) -> None:
    run_dir = cfg.run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"{run_dir} already exists and is not empty. Use a new run.name or pass --force.")
    if force and run_dir.exists():
        shutil.rmtree(run_dir)

    t = cfg.training
    accelerator = Accelerator(
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        mixed_precision=t.mixed_precision,
        log_with=None if cfg.logging.report_to == "none" else cfg.logging.report_to,
        project_config=ProjectConfiguration(project_dir=str(run_dir)),
    )
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    set_seed(t.seed)
    device, dtype = accelerator.device, weight_dtype_from(t.mixed_precision)

    # ---- data -------------------------------------------------------------------------
    samples = find_samples(cfg.data.dir, cfg.data.image_exts, cfg.data.caption_ext,
                           cfg.data.trigger_word, cfg.data.require_trigger_word)
    dataset = CaptionedImageDataset(samples, cfg.data.resolution, cfg.data.random_flip, cfg.data.repeats)
    logger.info("Dataset: %d images x %d repeats", dataset.num_unique, cfg.data.repeats)
    if accelerator.is_main_process:
        save_run_metadata(cfg, run_dir, len(samples), dataset_fingerprint(samples))

    # ---- text embeddings (encoders are frozen and captions are fixed -> encode once, then free) ----
    text_pipe = models.load_text_pipeline(cfg, device, dtype)
    caption_embeds, caption_pooled = models.encode_prompts(
        text_pipe, dataset.captions, cfg.model.max_sequence_length, device)
    val_prompts = validation.encode_validation_prompts(text_pipe, cfg.validation, cfg.model.max_sequence_length, device)
    text_ids = torch.zeros(caption_embeds.shape[1], 3)  # FLUX uses all-zero ids for text tokens
    del text_pipe
    free_memory()

    # ---- models -----------------------------------------------------------------------
    scheduler = models.load_scheduler(cfg)
    vae = models.load_vae(cfg, device, dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_cache = None
    if cfg.data.cache_latents:
        latent_cache = cache_vae_params(vae, dataset)
        dataset.skip_pixels = True

    transformer = models.load_transformer(cfg, device, dtype)
    if t.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    lora_params = models.add_lora(transformer, cfg.lora)

    optimizer = build_optimizer(lora_params, t)
    dataloader = DataLoader(dataset, batch_size=t.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                            drop_last=len(dataset) >= t.batch_size)
    lr_scheduler = get_scheduler(
        t.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=t.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=t.max_train_steps * accelerator.num_processes,
    )
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler)

    # ---- tracking ---------------------------------------------------------------------
    if cfg.logging.report_to == "wandb":
        accelerator.init_trackers(
            cfg.logging.project_name,
            config=flatten_dict(asdict(cfg)),
            init_kwargs={"wandb": {"name": cfg.wandb_run_name, "tags": cfg.logging.tags,
                                   "dir": str(run_dir)}},
        )

    logger.info("***** Training: %d optimizer steps, effective batch %d *****", t.max_train_steps,
                t.batch_size * t.gradient_accumulation_steps * accelerator.num_processes)

    global_step = 0
    if cfg.validation.at_start:
        run_validation(accelerator, transformer, vae, scheduler, val_prompts, cfg, 0, run_dir)

    progress = tqdm(total=t.max_train_steps, disable=not accelerator.is_local_main_process, desc="steps")
    transformer.train()
    step_loss = 0.0
    while global_step < t.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                idx = batch["index"].cpu()
                if latent_cache is not None:
                    params = torch.cat([latent_cache[i] for i in idx.tolist()]).to(device, dtype)
                else:
                    params = encode_pixels(vae, batch["pixel_values"])
                latents = sample_latents(params, vae)

                loss = flow_matching_loss(
                    transformer, latents,
                    caption_embeds[idx].to(device, dtype), caption_pooled[idx].to(device, dtype),
                    text_ids.to(device, dtype), scheduler, cfg.flow_matching, vae_scale_factor, device, dtype,
                )
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(lora_params, t.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            step_loss += loss.detach().item() / t.gradient_accumulation_steps

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            logs = {"train/loss": step_loss, "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/samples_seen": global_step * t.batch_size * t.gradient_accumulation_steps}
            if grad_norm is not None:
                logs["train/grad_norm"] = float(grad_norm)
            accelerator.log(logs, step=global_step)
            progress.set_postfix(loss=f"{step_loss:.4f}")
            step_loss = 0.0

            if global_step % cfg.checkpointing.every_n_steps == 0 and global_step < t.max_train_steps:
                save_lora(accelerator, transformer, run_dir / "checkpoints" / f"checkpoint-{global_step}", cfg)
            if global_step % cfg.validation.every_n_steps == 0:
                run_validation(accelerator, transformer, vae, scheduler, val_prompts, cfg, global_step, run_dir)

            if global_step >= t.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if t.max_train_steps % cfg.validation.every_n_steps != 0:  # make sure the last step is always sampled
        run_validation(accelerator, transformer, vae, scheduler, val_prompts, cfg, t.max_train_steps, run_dir)
    if cfg.checkpointing.save_final:
        save_lora(accelerator, transformer, run_dir / "checkpoints" / "final", cfg)
    accelerator.end_training()
    logger.info("Done. Outputs in %s", run_dir)
