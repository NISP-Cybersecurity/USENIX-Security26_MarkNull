# MarkNull

This repository contains the implementation of **MarkNull: Model-Agnostic Watermark Removal in AI-Generated Images via On-Manifold Latent Manipulation**, accepted at USENIX Security 2026.

---

## 1. Overview

MarkNull consists of two complementary components:

- **MarkNull**: An optimization-based adversarial attack operating in the latent space.
- **MarkNull-A**: An amortized variant that performs efficient one-pass watermark suppression via a trained Watermark Removal Network (WRN).

---

## 2. Installation

We recommend using a fresh conda environment with Python 3.10.

```bash
conda create -n marknull python=3.10 -y
conda activate marknull
pip install -r requirements.txt
```

---

## 3. Watermarked Image Preparation

To reproduce experiments, watermarked images must first be generated using the target watermarking schemes. Readers may refer to the original implementations cited in our manuscript, or use the benchmark toolkit at https://github.com/THU-BPM/MarkDiffusion.

Once generated, store all watermarked images under `./Watermarked/` following the directory structure below. We provide `Watermarked/SD2.1_GS/` as a concrete example, containing images watermarked by Gaussian Shading under Stable Diffusion 2.1.

Watermark baselines: [DwtDctSvd](https://github.com/guofei9987/blind_watermark) and [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading) (CVPR 2024).

---

## 4. Data Layout

```text
Watermarked/               # Clean watermarked images (input)
  SD2.1_GS/
    0.png
    1.png

Attacked/                  # Watermark-suppressed images (output)
  MarkNull/
    SD2.1_GS/
      0.png
      1.png
  MarkNull_A/
    SD2.1_GS/
      0.png
      1.png
```

- `<SUBSET_NAME>` encodes the target model and watermarking scheme (e.g., `SD2.1_GS` = Gaussian Shading under SD2.1).
- `<ATTACK_NAME>` denotes the attack method (`MarkNull` or `MarkNull_A`).

---

## 5. Attack

### MarkNull (Optimization-Based)

```bash
python MarkNull_attack.py --input_dir ../Watermarked/SD2.1_GS

```

### MarkNull-A (Amortized Variant)

#### Option A — Train the WRN

```bash
cd MarkNull-A
python train.py
```

#### Option B — Use Pretrained Weights

```bash
curl -L -o trained_model.7z https://github.com/asdasewq/MarkNull/releases/download/v1.0.0/trained_model.7z
7z x trained_model.7z -o./MarkNull-A
```

#### Run MarkNull-A Attack

```bash
python MarkNull_A_attack.py --input_dir ../Watermarked/SD2.1_GS
```

---

## 6. Evaluation

### Watermark Bit Accuracy

Evaluate watermark bit accuracy before and after attack using the provided decode scripts.

**Before attack (original watermarked images):**

```bash
cd Decode
./gs.sh ../Watermarked/SD2.1_GS
```

**After attack:**

```bash
cd Decode
./gs.sh ../Attacked/MarkNull/SD2.1_GS
./gs.sh ../Attacked/MarkNull_A/SD2.1_GS
```

### Defense Evaluation

We additionally provide an implementation of the detection-based defense proposed in the paper:

```bash
python Attack_detection.py
```

---

## 7. Case Study: Breaking SynthID-Image

We demonstrate that MarkNull generalizes to the commercial SynthID-Image watermarking system embedded in Google Gemini.

1. Open [Gemini](https://gemini.google.com) and generate an image using a text prompt. Save the image to `../Watermarked/synthid_512/`.
2. Run MarkNull or MarkNull-A on the downloaded image (see Attack section above).
3. Verify watermark removal using Google's official SynthID detection tool:
   - Open [Gemini](https://gemini.google.com) on your Android device or via the web.
   - Upload the attacked image using the image upload button.
   - Follow the official detection guide: https://support.google.com/gemini/answer/16722517

<p align="center">
  <img src="IMGS/synthid_ori.png" alt="SynthID Detection Example (Watermarked)" width="45%"/>
  <img src="IMGS/synthid_marknull.png" alt="SynthID Detection Example (MarkNull Attacked)" width="45%"/>
  <br>
  <em>Left: SynthID detection on the original watermarked image.    Right: SynthID detection after MarkNull attack.</em>
</p>

---

## 8. Case Study: Breaking AI-Generated Video Watermarking

This case study demonstrates that MarkNull generalizes to AI-generated video watermarking without any modality-specific adaptation.

### Environment Setup

```bash
cd MarkDiffusion
conda create -n markdiffusion python=3.11 -y
conda activate markdiffusion
pip install -r requirements.txt
```

### Step 1: Generate Watermarked Videos

```bash
conda activate markdiffusion
python videomark.py --ver 0
```

Watermarked videos will be saved to `../Watermarked/SVD_VideoMark/`.

### Step 2: Verify Watermark Integrity (Before Attack)

```bash
python videomark.py --ver 1 --input_dir ../Watermarked/SVD_VideoMark
```

### Step 3: Run MarkNull Attack

```bash
cd ../
conda activate marknull
python MarkNull_attack.py --input_dir Watermarked/SVD_VideoMark
```

Attacked videos will be saved to `Attacked/SVD_VideoMark/`.

### Step 4: Verify Attack Effectiveness (After Attack)

```bash
cd MarkDiffusion
conda activate markdiffusion
python videomark.py --ver 1 --input_dir ../Attacked/SVD_VideoMark
```

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{marknull2026,
  title = {MarkNull: Model-Agnostic Watermark Removal in AI-Generated Images via On-Manifold Latent Manipulation},
  author = {Cao, Jie and Li, Qi and Zhang, Zelin and Wu, Xiaodong and Liu, Lingshuang and Li, Xiangman and Ni, Jianbing},  
  booktitle = {35nd USENIX Security Symposium (USENIX Security 26)},
  year = {2026}

}
```
