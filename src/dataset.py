"""Dataset handling: pairs every image with the .txt caption that shares its filename.

The official Diffusers script cannot do this for a plain folder (it opens *every* file in the
directory as an image and only supports captions via a HF dataset / metadata.jsonl), so this is
the one piece we replace. Captions are read verbatim and never modified.
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
from pathlib import Path

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose, mirror
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def find_samples(
    data_dir: str | Path,
    image_exts: list[str],
    caption_ext: str = ".txt",
    trigger_word: str | None = None,
    require_trigger: bool = True,
) -> list[tuple[Path, str]]:
    """Return [(image_path, caption)] sorted by filename. Raises on any inconsistency."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    exts = {e.lower() for e in image_exts}
    images = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not images:
        raise ValueError(f"No images with extensions {sorted(exts)} found in {data_dir}")

    stems = [p.stem for p in images]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        raise ValueError(f"Several images share the same filename stem (and would share a caption): {duplicates}")

    missing = [p.name for p in images if not p.with_suffix(caption_ext).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) have no {caption_ext} caption file, e.g. {missing[:5]}")

    orphans = sorted(c.name for c in data_dir.glob(f"*{caption_ext}") if c.stem not in set(stems))
    if orphans:
        logger.warning("Caption files without a matching image (ignored): %s", orphans[:10])

    samples, empty, no_trigger = [], [], []
    for img in images:
        caption = img.with_suffix(caption_ext).read_text(encoding="utf-8").strip()
        if not caption:
            empty.append(img.name)
        if trigger_word and trigger_word not in caption:
            no_trigger.append(img.name)
        samples.append((img, caption))

    if empty:
        raise ValueError(f"Empty caption for: {empty[:10]}")
    if no_trigger:
        msg = f"{len(no_trigger)} caption(s) do not contain trigger word '{trigger_word}': {no_trigger[:10]}"
        if require_trigger:
            raise ValueError(msg + " (set data.require_trigger_word=false to downgrade this to a warning)")
        logger.warning(msg)
    return samples


def dataset_fingerprint(samples: list[tuple[Path, str]]) -> str:
    """sha256 over image bytes + captions, so a run records exactly which data it saw."""
    h = hashlib.sha256()
    for path, caption in samples:
        h.update(path.name.encode())
        h.update(path.read_bytes())
        h.update(caption.encode("utf-8"))
    return h.hexdigest()


MULTIPLE = 16  # FLUX: VAE downsamples 8x, transformer patchifies 2x2 -> H and W must be multiples of 16


def plan_size(w: int, h: int, max_pixels: int, multiple: int = MULTIPLE) -> tuple[tuple[int, int], tuple[int, int]]:
    """Aspect-ratio-preserving size plan, no square crop.

    1. Images larger than `max_pixels` are downscaled (never upscaled - interpolated pixels add no detail).
    2. Whatever is left over after rounding DOWN to a multiple of 16 is trimmed evenly from the edges
       (at most 15 px in total per axis; zero for images whose sides are already multiples of 16).
    Returns ((resize_w, resize_h), (final_w, final_h)).
    """
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    rw, rh = (round(w * scale), round(h * scale)) if scale < 1.0 else (w, h)
    fw, fh = max(multiple, rw // multiple * multiple), max(multiple, rh // multiple * multiple)
    return (rw, rh), (fw, fh)


class CaptionedImageDataset(Dataset):
    """Yields {"index", "pixel_values"}; `index` points into `self.captions` / cached embeddings.

    Every image keeps its own aspect ratio (variable-size tensors), so use batch_size=1 and
    gradient accumulation. Images are decoded lazily, so flip augmentation is re-sampled every epoch.
    """

    def __init__(self, samples, resolution, random_flip=False, repeats=1):
        self.samples = samples
        self.captions = [c for _, c in samples]
        self.max_pixels = resolution * resolution
        self.random_flip = random_flip
        self.repeats = repeats
        self.skip_pixels = False  # set True once latents are cached
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    @property
    def num_unique(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def get_pixels(self, unique_index: int) -> torch.Tensor:
        image = exif_transpose(Image.open(self.samples[unique_index][0])).convert("RGB")
        (rw, rh), (fw, fh) = plan_size(*image.size, self.max_pixels)
        if (rw, rh) != image.size:
            image = image.resize((rw, rh), Image.LANCZOS)
        if (fw, fh) != (rw, rh):
            left, top = (rw - fw) // 2, (rh - fh) // 2
            image = image.crop((left, top, left + fw, top + fh))
        if self.random_flip and random.random() < 0.5:
            image = mirror(image)
        return self.to_tensor(image)

    def __getitem__(self, i: int) -> dict:
        idx = i % self.num_unique
        item = {"index": idx}
        if not self.skip_pixels:
            item["pixel_values"] = self.get_pixels(idx)
        return item
