# P-Guide: Parameter-Efficient Prior Steering for Single-Pass CFG Inference

This repository is built upon the official implementation of *Flow Matching in Latent Space*.  
We extend it with **P-Guide**, a parameter-efficient framework for **single-pass classifier-free guidance (CFG)** via prior-space steering.

⚠️ Current status:  
We currently release only the **training code for P-Guide (ImageNet-1k, 256×256)**.  
More complete implementations (including sampling and evaluation pipelines) are under active preparation and will be released after paper acceptance.

---

## Installation

pip install -r requirements.txt

---

## Dataset

We use ImageNet-1k:

/path/to/imagenet/
    train/
    val/

---

## Training (Released Part)

We release the core training script:

train_imagenet256_DIT_PGuide.py

This implements the P-Guide prior steering mechanism in latent space:

z_cfg = μ_φ(∅) + w (μ_φ(y) - μ_φ(∅))

This enables single-pass CFG inference by shifting the initial latent distribution instead of modifying the velocity field during sampling.

---

### Run training

python train_imagenet256_DIT_PGuide.py \
  --lr 2e-5 \
  --ema_decay 0.9999 \
  --batch_size 128 \
  --total_steps 300001 \
  --save_step 20000

---

## Sampling (Coming Soon)

Sampling scripts are not included in the current release.

In P-Guide, inference avoids dual-pass CFG by applying guidance directly in the prior space:

# Standard CFG (dual-pass)
v = v_uncond + w * (v_cond - v_uncond)

# P-Guide (single-pass)
z = z_u + w * (z_c - z_u)
v = model(z)

This reduces inference cost by approximately 50%.

---

## Method Overview

P-Guide performs guidance by shifting the initial latent state rather than modifying the velocity field during ODE integration.

z_cfg = z_u + w (z_c - z_u)

This induces a trajectory-level perturbation that aligns with first-order CFG effects under flow linearization.

---

## Release Plan

- ✅ Training code (this repo)
- ⏳ Full sampling pipeline
- ⏳ Pretrained checkpoints
- ⏳ Evaluation scripts (FID / IS / sFID)
- ⏳ ImageNet-1k generation results
- ⏳ Clean unified codebase

All remaining components will be released after paper acceptance.

---

## Acknowledgments

This codebase is adapted from Flow Matching in Latent Space and related works.

---

## Contact

Please open an issue if you have questions or find bugs.