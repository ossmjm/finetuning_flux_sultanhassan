"""Train a FLUX LoRA from a YAML config.

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/experiment_rank32.yaml --set training.learning_rate=5e-5
    accelerate launch scripts/train.py --config configs/baseline.yaml      # multi-GPU
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.training import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set lora.rank=32 (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides), force=args.force)


if __name__ == "__main__":
    main()
