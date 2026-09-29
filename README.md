# FLUX LoRA – Mosque-Madrasa of Sultan Hassan

LoRA fine-tuning of `black-forest-labs/FLUX.1-dev` with Hugging Face Diffusers. Trigger word: `sltnhsn`.
The training step is adapted from the official
[`train_dreambooth_lora_flux.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py);
everything else (config, dataset, validation, tracking, run metadata) is split into small modules.

## Layout

```
configs/      baseline.yaml + experiment_*.yaml (each experiment = `base: baseline.yaml` + overrides)
data/         image_N.jpg + image_N.txt pairs (you provide)
src/
  config.py       typed YAML schema, `base:` inheritance, --set overrides
  dataset.py      image <-> .txt pairing, checks, lazy preprocessing
  models.py       load FLUX components, encode prompts, inject LoRA
  training.py     flow-matching loss, optimizer, loop, checkpoints
  validation.py   fixed prompts/seeds, sample saving, W&B image logging
  utils.py        dtype helpers, run metadata (reproducibility)
scripts/
  train.py                 entry point
  generate_validation.py   re-generate the validation set from any checkpoint (or the base model)
outputs/<run.name>/        created per run (see below)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
# install a CUDA build of torch first, then:
pip install -r requirements.txt
huggingface-cli login      # FLUX.1-dev is gated: accept the license on the model page first
wandb login                # or set logging.report_to: none
```

Put your dataset in `data/` (`image_1.jpg` + `image_1.txt`, ...). On start-up the pipeline checks that
every image has a caption, no caption is empty, and every caption contains `sltnhsn`
(it only *checks* — captions are never edited).

**Hardware:** the 12B transformer alone is ~24 GB in bf16, so plan for a 40 GB+ GPU (A100/A6000/H100) for
1024 px training. Smaller cards would need quantisation or offloading, which this project does not implement.

## Run an experiment

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/experiment_rank32.yaml
python scripts/train.py --config configs/baseline.yaml --set lora.rank=64 --set lora.alpha=64 --set run.name=r64
accelerate launch scripts/train.py --config configs/baseline.yaml      # multi-GPU
```

New experiment = new YAML:

```yaml
base: baseline.yaml
run: {name: my_experiment}
lora:
  rank: 32
  alpha: 32
  target_modules: [attn.to_q, attn.to_k, attn.to_v, attn.to_out.0]   # lists replace, not merge
```

`configs/experiment_attn_ff.yaml` shows adding the feed-forward layers to the LoRA targets.
A run refuses to overwrite an existing `outputs/<run.name>/` unless you pass `--force`.

## Outputs

```
outputs/<run.name>/
  config_resolved.yaml       the fully merged config actually used
  run_metadata.json          base-model commit, seed, dataset path + sha256, LoRA config, package versions, GPU, git commit
  requirements.lock.txt      pip freeze of the environment
  checkpoints/checkpoint-250/, checkpoint-500/, ..., final/    LoRA adapter only (pytorch_lora_weights.safetensors)
  samples/step_0/, step_250/, ...   PNGs + grid.jpg + prompts.json
```

Load a result:

```python
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("outputs/baseline/checkpoints/final")
image = pipe("sltnhsn, ...", num_inference_steps=28, guidance_scale=3.5).images[0]
```

## Validation – how to choose a checkpoint

Every `validation.every_n_steps` (and at step 0 = base model) the same prompts with the same seeds
(`validation.seed + j`, CPU generator) are generated, so runs are comparable image-for-image.
Two groups are produced:

* **trigger** prompts (contain `sltnhsn`): different views, lighting, close-ups.
* **control** prompts (no trigger word): should look like the base model at every step.

Read the `step_*/grid.jpg` files (or the `validation/trigger` and `validation/control` panels in W&B) side by side:

| Observation | Interpretation |
|---|---|
| Trigger images look like generic mosques | underfitting: more steps / higher LR / more capacity |
| Facade, minaret and stonework become recognisably Sultan Hassan, new lighting/composition still works | learning the concept |
| Trigger images copy training compositions, ignore prompt changes (time of day, view) | starting to overfit |
| Control prompts change style/quality vs step 0, or show the mosque | drift / concept bleeding |

Pick the earliest checkpoint where the concept is clear and controls are still unchanged; loss alone won't tell you.
Re-generate at any time: `python scripts/generate_validation.py --config <cfg> --lora <checkpoint> --out <dir>`.

## Tracked in W&B

`train/loss`, `train/lr`, `train/grad_norm`, step, and the whole resolved config (rank, alpha, lr, batch size,
resolution, seed, target modules, ...). Run names encode the essentials: `<name>_r16_a16_lr0.0001_bs1_s42`.
Validation images are logged per step.

## Reproducing a run

1. Check out this repo at `project_git_commit` from `run_metadata.json`.
2. `pip install -r outputs/<run>/requirements.lock.txt` (or the requirements file).
3. Restore the dataset (`dataset_sha256` in the metadata lets you verify it).
4. Set `model.revision` to `base_model_commit_resolved` if the base weights may have changed.
5. `python scripts/train.py --config outputs/<run>/config_resolved.yaml --set run.name=<new-name>`

The seed fixes LoRA init, timestep/noise sampling and data order. GPU kernels are not bit-deterministic,
so expect close, not identical, curves.

## Design notes

* Text embeddings are computed once up front (encoders are frozen, captions fixed), then CLIP/T5 are freed;
  with `data.cache_latents: true` the VAE is also used only once for training (kept afterwards for validation decoding).
* Only the LoRA weights are trained and saved; the trainable LoRA parameters are kept in fp32 while the base model is bf16.
* `flow_matching.weighting_scheme` defaults to the official `none` (uniform timesteps); `logit_normal` is a common alternative to test.
* **No square crop.** Each image keeps its own aspect ratio: images above the `data.resolution`² pixel budget are downscaled,
  smaller ones are never upscaled, and sides are snapped down to multiples of 16 (FLUX needs this; at most 15 px trimmed,
  nothing at all if your files already comply). Because tensors differ in size, `training.batch_size` must be 1 -
  use `gradient_accumulation_steps` for a larger effective batch. Validation samples are generated at
  `validation.sizes` (landscape + portrait) to match the data.
* Not included on purpose: resume-from-checkpoint (optimizer state is not saved), text-encoder training, prior preservation.
