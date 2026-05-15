# DS677 — Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![VideoMAE](https://img.shields.io/badge/VideoMAE-Base%2086M-blueviolet)
![Kinetics](https://img.shields.io/badge/Dataset-Kinetics--mini%2064%20clips-orange)
![Gradio](https://img.shields.io/badge/Gradio-Live%20Demo-FF6B6B)
![NJIT](https://img.shields.io/badge/NJIT-M.S.%20AI-CC0000)

Temporal pattern learning for human behavior recognition and anomaly detection from video — comparing a large pre-trained transformer (VideoMAE-Base, 86M params) against a compact inductive-bias model (CNN+LSTM, 2.5M params) in the low-data regime.

---

## Project Overview

This project investigates a fundamental question in video understanding: **when does inductive bias beat scale?** Using a 64-clip subset of Kinetics-mini, seven experiments systematically measure accuracy, efficiency, and generalization for two architectures at opposite ends of the parameter spectrum. The core finding — that a 34× smaller model outperforms a pre-trained transformer by nearly 19 percentage points when data is scarce — has direct implications for deploying video models in resource-constrained settings.

The work also includes per-class anomaly detection via centroid distance, data crossover analysis at 16/32/48/64 clips, k-fold cross-validation for honest generalization estimates, and a live Gradio demo tested on out-of-distribution theft footage.

---

## Key Results

### Model Comparison (64-clip Kinetics-mini)

| Model | Params | Val Accuracy | Weighted F1 | Inference Latency |
|-------|--------|-------------|-------------|------------------|
| VideoMAE-Base | 86M | 56.25% | 58.44% | 68.1 ms |
| **CNN+LSTM** | **2.5M** | **75.00%** | **74.31%** | **2.9 ms** |
| **Gap** | | **+18.75 pp** | **+15.87 pp** | **23× faster** |

**Core finding:** CNN+LSTM's inductive bias (spatial locality via conv, temporal order via LSTM) outperforms VideoMAE's global attention in the low-data regime. At 64 clips, there is insufficient data to overcome VideoMAE's need for large-scale fine-tuning.

### Cross-Validation

| Metric | Value |
|--------|-------|
| K-Fold accuracy (CNN+LSTM, k=5) | 35.0% ± 6.1% |

> The honest k-fold estimate is lower than the single-split result — a known effect of small datasets where validation-set variance dominates. Reported for full transparency.

### Anomaly Detection (Experiment 5)

| Method | AUROC |
|--------|-------|
| Per-class centroid distance | 0.5312 |

Null result: centroid-based anomaly detection in embedding space did not exceed random chance on this dataset. Reported as a negative finding.

---

## Experiments

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | Core Ablation — accuracy gap | 18.75 pp advantage for CNN+LSTM |
| 2 | Per-Class F1 Analysis | Class-level performance breakdown |
| 3 | Fine-Tuning Depth Sensitivity | 4 strategies (frozen/partial/full/head-only) |
| 4 | K-Fold Cross-Validation | 35.0% ± 6.1% honest generalization estimate |
| 5 | Per-Class Centroid Anomaly Detection | Null result (AUROC 0.5312) |
| 6 | Data Crossover at 16/32/48/64 clips | No crossover found — CNN+LSTM leads at all sizes |
| 7 | Gradio Live Demo + OOD Theft Video Test | Live inference on out-of-distribution footage |

---

## Files

| File | Description |
|------|-------------|
| `DS677_Project.ipynb` | Complete notebook — all 7 experiments with full outputs |
| `DL_REPORT_FINAL.pdf` | 19-page research report with analysis and references |
| `DS677_Presentation.mp4` | ~30-minute video walkthrough of results |

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision transformers datasets gradio scikit-learn matplotlib
```

### Run the notebook

```bash
# Recommended: Google Colab with T4 GPU
jupyter notebook DS677_Project.ipynb
```

Run all cells top to bottom. The notebook is self-contained — it downloads the dataset, trains both models, runs all 7 experiments, and launches the Gradio demo in the final cell.

### Expected runtime

| Stage | Estimated Time (T4 GPU) |
|-------|------------------------|
| Dataset download | ~5 min |
| VideoMAE fine-tuning | ~20 min |
| CNN+LSTM training | ~5 min |
| All 7 experiments | ~45 min total |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training and evaluation |
| Transformers (VideoMAE-Base) | Pre-trained video transformer backbone |
| CNN + LSTM (custom) | Compact inductive-bias baseline |
| Kinetics-mini | 64-clip video classification dataset |
| scikit-learn | K-fold CV, F1, AUROC |
| Gradio | Live inference demo |
| Google Colab T4 | Training environment |

---

## Author

**Ayush Yadav** | M.S. AI @ NJIT | [ary22@njit.edu](mailto:ary22@njit.edu)
