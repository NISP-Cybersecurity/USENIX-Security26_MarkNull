# MarkNull

This repository contains the implementation of **MarkNull: Model-Agnostic Watermark Removal in AI-Generated Images via On-Manifold Latent Manipulation**.

## 💡 Overview

MarkNull consists of two components:

- **MarkNull**: An optimization-based adversarial attack in the latent space.
- **MarkNull-A**: An amortized variant using a trained neural network for efficient watermark removal.


## Watermarked Image Preparation
To facilitate deployment of watermarking schemes, readers may refer to the original implementations of all baseline methods cited in our manuscript, or alternatively use the benchmark toolkit at https://github.com/THU-BPM/MarkDiffusion. After generating watermarked images in batch, please store them under ./Watermarked/ and then run our attack pipeline on this directory.

We provide Watermarked/SD2.1_GS/ as a concrete example, which contains watermarked images produced by the Gaussian Shading method under Stable Diffusion 2.1.

## Attack

### 1. Run MarkNull (Optimization-Based)
```bash
python MarkNull_attack.py
```

### 2 .Run MarkNull-A (Amortized Variant)

```bash
python train.py
```
```bash
python MarkNull_A_attack.py
```
