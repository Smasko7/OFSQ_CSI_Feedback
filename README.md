# 📡 Multi-Length CSI Feedback With Ordered Finite Scalar Quantization (OFSQ)

> Multi-Length CSI Feedback With Ordered Finite Scalar Quantization. This is a DL-based research work for next generation (6G) wireless communication networks, with focus on optimization of the physical layer (made with PyTorch).

This repository contains the official PyTorch implementation of the paper:

> 📄 **Multi-Length CSI Feedback With Ordered Finite Scalar Quantization**
> Kosmas Liotopoulos, Nikos A. Mitsiou, Panagiotis G. Sarigiannidis, and George K. Karagiannidis
> *IEEE Communications Letters*, vol. 29, no. 8, pp. 1973–1977, August 2025
> [[IEEE Xplore]](https://doi.org/10.1109/LCOMM.2025.3581951)

---

## 🔍 Overview

OFSQ is a lightweight, deep-learning-based architecture for **multi-length channel state information (CSI) feedback** in massive MIMO systems. It combines **Finite Scalar Quantization (FSQ)** with **ordered representation learning** (via nested dropout) to enable flexible-rate CSI feedback from a single trained autoencoder — without the need for exhaustive codebook search.

### ✨ Key Features

| | |
|---|---|
| 🔌 **Plug-in architecture** | Can be paired with any autoencoder (e.g., CsiNet, CRNet) |
| 📶 **Multi-length feedback** | A single model supports arbitrary feedback bitstream lengths by transmitting only the first *n\** out of *K* quantized sub-vectors |
| ⚡ **Low complexity** | Eliminates the codebook search of VQ, reducing quantization FLOPs by over **two orders of magnitude** vs. OVQ |
| 🎯 **No auxiliary losses** | Needs only a simple MSE reconstruction loss; no commitment or codebook losses required |
| 📈 **Improved reconstruction** | Achieves better or comparable NMSE and cosine similarity vs. OVQ and μ-law baselines across indoor and outdoor COST 2100 scenarios |

---

## ⚙️ Method

The OFSQ pipeline works as follows:

1. **Encoder** — compresses the angular-delay CSI matrix **H** into a latent vector **z** ∈ ℝ^M.
2. **Reshape** — **z** is reshaped into *K* sub-vectors of dimension *m*, such that *m × K = M*.
3. **FSQ Quantization** — each scalar entry is bounded via `⌊L/2⌋ · tanh(·)` and rounded to the nearest integer, producing quantized sub-vectors without any learned codebook. The hyperparameter vector **L** = [L₁, …, Lₘ] defines the implicit codebook of size ∏Lⱼ.
4. **Nested Dropout (Ordered Quantization)** — during fine-tuning, a random prefix of *n* sub-vectors is kept (the rest are zeroed), enforcing an importance ordering so that earlier sub-vectors carry the most information.
5. **Decoder** — reconstructs the CSI matrix from the (possibly truncated) quantized codeword.

### 🏋️ Training Phases

| Phase | Description |
|---|---|
| **Pre-training** | Full quantized codeword is used; model minimizes MSE reconstruction loss with the straight-through estimator (STE) for gradient flow through the rounding operation |
| **Fine-tuning** | Nested dropout is activated; a random truncation index *n* ~ U{1, …, K} is sampled per batch, enabling ordered representation learning |

At deployment, the UE selects *n\** sub-vectors based on its available resources, producing a feedback bitstream of *n\** × ⌈log₂(∏Lⱼ)⌉ bits.

---

## 🚀 Getting Started

### 1. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch**: must be installed separately to match your CUDA version.
> See the comment at the top of `requirements.txt` or visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### 2. 🗄️ Obtain the COST 2100 dataset

Download the COST 2100 channel matrices and place all four `.mat` files inside the `data/` directory:

```
data/
├── DATA_Htrainin.mat    # indoor  — training set (100,000 samples)
├── DATA_Htestin.mat     # indoor  — test set      (20,000 samples)
├── DATA_Htrainout.mat   # outdoor — training set  (100,000 samples)
└── DATA_Htestout.mat    # outdoor — test set      (20,000 samples)
```

### 3. 🔬 Train a model

> ⚠️ Scripts must be run **from their own directory** so that relative paths to `data/` and `outputs/` resolve correctly.

**Proposed method (OFSQ):**
```bash
cd proposed_method/train
python OFSQ_CRNet.py
```

**Baseline** (example — OVQ with CRNet):
```bash
cd baselines/train
python OVQ_CRNet.py
```

Trained model weights are saved to `outputs/models/`, loss curves to `outputs/plots/`, and loss logs to `outputs/logs/`.

### 4. 🧪 Evaluate a model

Run the corresponding `_TEST` script from its directory. It loads the model saved by the training script.

```bash
cd proposed_method/test
python OFSQ_CRNet_TEST.py
```

```bash
cd baselines/test
python OVQ_CRNet_TEST.py
```

### 5. 📊 Reproduce the paper plots

```bash
cd plots
python plot_CRNet.py     # NMSE and cosine similarity curves for CRNet-based models
python plot_CsiNet.py    # NMSE and cosine similarity curves for CsiNet-based models
```

Figures are saved to `outputs/plots/`.

---

## 🗂️ Repository Structure

```
OFSQ_CSI_Feedback/
├── proposed_method/    # OFSQ model implementation (encoder, decoder, FSQ, nested dropout)
│   ├── train/
│   └── test/
├── baselines/          # Baseline models (OVQ, VQ, FSQ without ordering)
│   ├── train/
│   └── test/
├── plots/              # Scripts for reproducing paper plots
├── data/               # COST 2100 channel data (.mat files — not tracked in git)
├── outputs/            # Saved model checkpoints, plots, and logs
│   ├── models/
│   ├── plots/
│   └── logs/
├── requirements.txt
└── README.md
```

---

## 🧮 Simulation Setup

The experiments use the **COST 2100** channel model with the following configuration:

| Parameter | Value |
|---|---|
| Transmit antennas (Nₜ) | 32 |
| Truncated subcarriers (Nc) | 32 |
| CSI matrix size | 32 × 32 (real & imag separated → 2048 parameters) |
| Compression ratio (γ) | 1/4 (M = 512) |
| Sub-vector dimension (m) | 4 |
| Number of sub-vectors (K) | 128 |
| Hyperparameter vector (L) | [8, 5, 5, 5] → implicit codebook size 1000 |
| Feedback bits (B) | {160, 320, 640, 960, 1280} |
| Pre-training epochs | 200 |
| Fine-tuning epochs | 100 |
| Optimizer | Adam (lr = 10⁻³) |
| Batch size | 200 |

Two scenarios are evaluated: **indoor picocell** (5.3 GHz DL) and **outdoor rural** (300 MHz DL).

---

## 📉 Results

OFSQ achieves competitive or superior NMSE and cosine similarity compared to OVQ, while being over **100× cheaper** in quantization FLOPs:

| Scheme | Quantization FLOPs | Auxiliary Losses |
|---|---|---|
| OVQ | ~1.44M | Yes (commitment + codebook) |
| μ-law | ~13K | No |
| ⭐ **OFSQ** | **~6.1K** | **No** |

For detailed NMSE and cosine similarity curves across feedback lengths and scenarios, see the figures in `outputs/plots/` or the paper.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{liotopoulos2025ofsq,
  author    = {Liotopoulos, Kosmas and Mitsiou, Nikos A. and Sarigiannidis, Panagiotis G. and Karagiannidis, George K.},
  title     = {Multi-Length {CSI} Feedback With Ordered Finite Scalar Quantization},
  journal   = {IEEE Communications Letters},
  volume    = {29},
  number    = {8},
  pages     = {1973--1977},
  year      = {2025},
  doi       = {10.1109/LCOMM.2025.3581951}
}
```

---

## 🙏 Acknowledgements

This work was funded by the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union's Horizon Europe research and innovation programme (Grant Agreement No. 101096456-NANCY).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — © 2025 Kosmas Liotopoulos.
