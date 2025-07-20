
# Weakly-Supervised Video Anomaly Detection

This project implements a **weakly-supervised anomaly detection** framework for surveillance videos using a **Multiple Instance Learning (MIL)** strategy. It integrates features from **I3D** and **TimeSformer**, demonstrating high temporal precision and robustness on the **UCF-Crime** dataset.

## 📚 Overview

Traditional video anomaly detection often relies on fully-supervised training or weakly expressive features. In contrast, this project fuses *spatial-temporal motion (I3D)* and *semantic transformer features (TimeSformer)*, and optimizes anomaly prediction using a **top-k MIL loss** under weak supervision (only video-level labels).

Key components:
- Dual feature extraction (I3D + TimeSformer).
- Segment-based MIL classifier with top-k loss.
- Confidence-based anomaly scoring and t-SNE/UMAP visualization.
- Evaluation via ROC, PR curves, and confusion matrix.

## 🗂 Project Structure

```
.
├── main.py              # Training and evaluation loop
├── dataset.py           # Dataset loader for feature-based videos
├── learner.py           # Learner class with MIL classifier
├── loss.py              # Top-k MIL loss function
├── vis.py               # Visualization utilities (ROC, PR, confusion)
├── tsne_vis.py          # t-SNE and UMAP embeddings visualization
├── checkpoints/         # Trained models (optional)
├── plots/               # Output visualizations (auto-created)
└── data/
    ├── Training/
    └── Testing/
```

## 🚀 Running the Code

### 1. Prerequisites

Install Python packages:

```bash
pip install torch torchvision matplotlib scikit-learn umap-learn plotly
```

### 2. Prepare Data

Extract features using I3D and TimeSformer and organize them:

```
data/
├── Training/
│   ├── Normal_I3D_RGB_NPY/
│   └── Anomaly_I3D_RGB_NPY/
├── Testing/
│   ├── Normal_I3D_RGB_NPY/
│   └── Anomaly_I3D_RGB_NPY/
```

Same for TimeSformer features in parallel folders.

### 3. Train the Model

```bash
python main.py --data_root path/to/data --feat both --batch 16 --seg 64
```

Model checkpoints are saved in `checkpoints/`.

### 4. Visualize Embeddings (t-SNE)

```bash
python tsne_vis.py --data_root path/to/data --feat_type both --checkpoint checkpoints/model.pth
```

Output images and interactive plots will be saved in `plots/`.

## 📈 Evaluation & Visualization

- **ROC / PR curves** and **confusion matrix** are saved per epoch under `plots/`.
- The following example shows segment-level anomaly scores across time, overlaid with keyframe thumbnails and ground-truth regions.

## 🧠 Key Results

| Method (Weak Supervision)                     | Backbone / Key Idea           | UCF-Crime AUC (%) |
|----------------------------------------------|-------------------------------|-------------------|
| **Ours: I3D + TimeSformer, Top-k MIL**        | Feature fusion + Top-k MIL    | **90.70**         |
| MSTAgent-VAD (VideoSwin, multi-scale, RTFM)   | Transformer + RTFM            | 89.27             |
| MSTAgent-VAD (I3D variant)                    | I3D + RTFM                     | 85.52             |
| Chen et al. 2023 (VST-RGB)                    | Vision Swin-Transformer       | 86.67             |
| Tian et al. 2021                              | I3D + smoothness, sparsity    | 84.30             |
| Sultani et al. 2018 (original MIL ranking)    | I3D                           | 75.41             |

## 📍 Citation

If this work helps your research, please cite:

```bibtex
@article{your_name2024anomaly,
  title={Weakly-Supervised Anomaly Detection via Feature Fusion and Top-k MIL},
  author={Your Name},
  journal={Your Institution Report},
  year={2024}
}
```

## 🛠 Acknowledgements

- UCF-Crime Dataset: https://webpages.uncc.edu/cchen62/dataset.html
- MIL Loss based on: Sultani et al., 2018
- Transformer features via TimeSformer
