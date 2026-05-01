# DS677 Deep Learning — Milestone 2
## Temporal Pattern Learning for Human Behavior Recognition and Anomaly Detection from Video

**Student:** Ayush Yadav | ary22@njit.edu  
**Instructor:** Prof. Khalid Bakhshaliyev | NJIT Spring 2026

**Date Of Submission:** 1st May 2026
### Project Summary
Seven experiments comparing VideoMAE-Base (86M params) vs CNN+LSTM (2.5M params) on a 64-clip subset of Kinetics-mini.

**Core finding:** CNN+LSTM outperforms VideoMAE by 18.75pp accuracy in the low-data regime. Inductive bias wins at 64 clips.

| Model | Val Accuracy | Weighted F1 | Latency |
|---|---|---|---|
| VideoMAE-Base | 56.25% | 58.44% | 68.1ms |
| CNN+LSTM | **75.00%** | **74.31%** | **2.9ms** |

### Files
- `DL_REPORT_FINAL.pdf` — Full 19-page research report
- `DS677_Final_Notebook_SUBMISSION.ipynb` — Complete notebook (7 experiments, all outputs)
- `DS677_Presentation.mp4` — Video presentation (~30 min)

### Experiments
1. Core Ablation — 18.75pp accuracy gap
2. Per-Class F1 Analysis
3. Fine-Tuning Depth Sensitivity (4 strategies)
4. K-Fold Cross-Validation — 35.0% ± 6.1% honest estimate
5. Per-Class Centroid Anomaly Detection — null result (AUROC 0.5312)
6. Data Crossover at 16/32/48/64 clips — no crossover found
7. Gradio Live Demo + OOD Theft Video Test
