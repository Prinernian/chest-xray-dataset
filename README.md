# Unsupervised Pneumonia Detection with a Denoising Diffusion Model

Anomaly detection on chest X-rays using a DDPM trained **only on healthy (NORMAL) images**.
At test time, each image is partially noised and reconstructed; images the model cannot
reconstruct well are flagged as anomalous (pneumonia).

## Method

**Idea.** A diffusion model trained only on normal lungs learns the distribution of healthy chests.
When it denoises a pneumonia image, it tends to "heal" the abnormal region into healthy-looking
tissue, so the difference between the input and its reconstruction is larger.

**Training (one-class).**
- Forward process: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- Objective (simplified DDPM loss): $\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\,\|\epsilon - \epsilon_\theta(x_t, t)\|^2$
- Linear $\beta$ schedule from $10^{-4}$ to $0.02$, $T = 1000$ steps

**Anomaly scoring.** For each test image $X$:
1. Noise it to an intermediate step $t$ (here $t \in \{100, 200, 300\}$)
2. Reconstruct $X'$, either with a one-step estimate
   $\hat{x}_0 = \big(x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon\big)/\sqrt{\bar\alpha_t}$
   or with the full DDPM reverse chain $t \to 0$
3. Compute three scores, averaged over noise levels and noise draws:
   - **Noise-prediction MSE:** $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$
   - **Reconstruction mean:** mean of the error map $|X - X'|$
   - **Reconstruction q95:** 95th percentile of the error map $|X - X'|$

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle).

| Split | NORMAL | PNEUMONIA | Used for |
|---|---|---|---|
| train | 1,341 | — | training (NORMAL only) |
| test | 234 | 390 | evaluation |

The dataset itself is **not** included in this repository.

## Setup

| Component | Setting |
|---|---|
| Model | `deepinv.models.DiffUNet` (1 input / 1 output channel), trained from scratch |
| Input | 32×32 grayscale, range [0, 1] |
| Optimizer | Adam, lr = 1e-4, batch size 32 |
| Training | 100 epochs on NORMAL images only |
| Hardware | Kaggle GPU |

## Results (test set)

**AUROC**

| Score | One-step reconstruction | Full reverse chain + blurred, cropped error map |
|---|---|---|
| Noise-prediction MSE | 0.668 | 0.654 |
| Reconstruction mean | 0.661 | 0.638 |
| Reconstruction q95 | 0.650 | 0.655 |

**Operating point** (threshold = 95th percentile of NORMAL scores, i.e. ~5% false-positive rate; full-chain run)

| Score | Precision | Recall | Specificity | AP (baseline 0.625) |
|---|---|---|---|---|
| Noise-prediction MSE | 0.867 | 0.200 | 0.949 | 0.768 |
| Reconstruction mean | 0.846 | 0.169 | 0.949 | 0.755 |
| Reconstruction q95 | 0.840 | 0.162 | 0.949 | 0.765 |

**Interpretation.** The model detects a real but weak signal (AUROC ≈ 0.65). At a 5% false-alarm
rate it identifies about 1 in 5 pneumonia cases, mostly the most severe ones. Error maps show that
much of the reconstruction error comes from normal anatomical variation (body outline, diaphragm,
image borders) rather than from the disease itself.

## Limitations

- **Low resolution:** at 32×32, much of the fine opacity texture is lost.
- **Short training:** ~4,200 optimization steps, without EMA weights.
- **Tuning on the test set:** the noise level, scoring choices and threshold were selected on the
  test set (no held-out normals), so the reported numbers are slightly optimistic.
- **Possible acquisition bias:** the dataset is pediatric, and the classes differ in acquisition
  characteristics (exposure, contrast), which may contribute to the separation independently of
  pathology.

## Possible improvements

- Higher resolution (64×64 or 128×128)
- Longer training with EMA weights
- Inputs scaled to [-1, 1]
- Held-out NORMAL validation split for selecting $t$ and the threshold
- Simplex noise instead of Gaussian (AnoDDPM-style)
- Lung masking of the error map

## Usage

1. Download the dataset from Kaggle into `data/chest_xray/`.
2. Run the notebook `X-ray_chest_analysis.ipynb`:
   - **Training:** trains the DDPM on `train/NORMAL` and saves `trained_diffusion_model.pth`.
   - **Evaluation:** scores `test/NORMAL` and `test/PNEUMONIA`, and reports AUROC,
     precision/recall/AP, histograms and error heatmaps.

**Requirements:** `torch`, `torchvision`, `deepinv`, `scikit-learn`, `numpy`, `matplotlib`

## References

- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Wyatt et al. *AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise.* CVPR Workshops 2022.
- [DeepInverse](https://deepinv.github.io/) library.
