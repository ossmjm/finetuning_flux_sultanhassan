"""Model initialisation: everything that touches FLUX weights lives here."""
from __future__ import annotations

import logging

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig

logger = logging.getLogger(__name__)


def load_text_pipeline(cfg, device, dtype) -> FluxPipeline:
    """FluxPipeline holding ONLY tokenizers + CLIP + T5 (no transformer / VAE).

    We reuse `FluxPipeline.encode_prompt` instead of re-implementing the CLIP/T5 encoding
    the way the official script does.
    """
    pipe = FluxPipeline.from_pretrained(
        cfg.model.name, transformer=None, vae=None,
        revision=cfg.model.revision, torch_dtype=dtype,
    )
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    return pipe.to(device)


@torch.no_grad()
def encode_prompts(pipe: FluxPipeline, prompts: list[str], max_sequence_length: int, device, chunk: int = 8):
    """-> (prompt_embeds [N, L, D], pooled [N, 768]) on CPU, in the encoders' dtype."""
    embeds, pooled = [], []
    for i in range(0, len(prompts), chunk):
        e, p, _ = pipe.encode_prompt(
            prompt=prompts[i:i + chunk], prompt_2=None, device=device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        embeds.append(e.cpu())
        pooled.append(p.cpu())
    return torch.cat(embeds), torch.cat(pooled)


def load_scheduler(cfg) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model.name, subfolder="scheduler", revision=cfg.model.revision
    )


def load_vae(cfg, device, dtype) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        cfg.model.name, subfolder="vae", revision=cfg.model.revision, variant=cfg.model.variant
    )
    vae.requires_grad_(False)
    return vae.to(device, dtype=dtype)


def load_transformer(cfg, device, dtype) -> FluxTransformer2DModel:
    """Load the FLUX transformer. With model.quantization="nf4" the frozen base weights are stored in
    4-bit (QLoRA): ~7 GB instead of ~24 GB, so it fits on 16 GB GPUs. LoRA weights stay in higher precision."""
    kwargs = {}
    if cfg.model.quantization == "nf4":
        from diffusers import BitsAndBytesConfig  # needs `pip install bitsandbytes`
        kwargs = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
            device_map={"": device},  # quantize straight onto the GPU, never materialise 24 GB
        )
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.model.name, subfolder="transformer", revision=cfg.model.revision,
        variant=cfg.model.variant, torch_dtype=dtype, **kwargs,
    )
    transformer.requires_grad_(False)  # base model stays frozen; only LoRA params get grads
    return transformer if kwargs else transformer.to(device)


def add_lora(transformer: FluxTransformer2DModel, lora_cfg) -> list[torch.nn.Parameter]:
    """Inject LoRA adapters into the frozen transformer and return the trainable parameters."""
    transformer.add_adapter(LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        init_lora_weights=lora_cfg.init_lora_weights,
        target_modules=lora_cfg.target_modules,
    ))
    # Keep the (small) trainable LoRA weights in fp32 while the base weights stay in bf16.
    cast_training_params(transformer, dtype=torch.float32)

    params = [p for p in transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in transformer.parameters())
    if n_train == 0:
        raise ValueError(f"No LoRA layers were created: check lora.target_modules={lora_cfg.target_modules}")
    logger.info("LoRA: r=%d alpha=%d | trainable %.2fM / %.2fM params (%.3f%%)",
                lora_cfg.rank, lora_cfg.alpha, n_train / 1e6, n_total / 1e6, 100 * n_train / n_total)
    return params
