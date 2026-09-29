"""Generate the fixed validation set from a saved LoRA (or the plain base model).

    python scripts/generate_validation.py --config configs/baseline.yaml \
        --lora outputs/baseline/checkpoints/checkpoint-500 --out outputs/baseline/samples/manual_500
    python scripts/generate_validation.py --config configs/baseline.yaml --out outputs/base_model_samples

Uses the same prompts, seeds and sampler settings as training-time validation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from diffusers import FluxPipeline  # noqa: E402

from src.config import load_config  # noqa: E402
from src.validation import ValidationPrompts, generate_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora", default=None, help="checkpoint dir; omit to sample the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-scale", type=float, default=None, help="optional LoRA strength (default 1.0)")
    ap.add_argument("--cpu-offload", action="store_true", help="for GPUs with < ~40 GB")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = FluxPipeline.from_pretrained(cfg.model.name, revision=cfg.model.revision, torch_dtype=torch.bfloat16)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        if args.lora_scale is not None:
            pipe.fuse_lora(lora_scale=args.lora_scale)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    v = cfg.validation
    prompts = list(v.prompts) + list(v.control_prompts)
    kinds = ["trigger"] * len(v.prompts) + ["control"] * len(v.control_prompts)
    with torch.no_grad():
        embeds, pooled, _ = pipe.encode_prompt(prompt=prompts, prompt_2=None, device=device,
                                               max_sequence_length=cfg.model.max_sequence_length)
    generate_samples(pipe, ValidationPrompts(prompts, kinds, embeds.cpu(), pooled.cpu()), v, args.out, device)
    print(f"Saved samples to {args.out}")


if __name__ == "__main__":
    main()
