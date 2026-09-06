# Deepfake Image Detection using Deep Learning and Transfer Learning

A binary image classifier that distinguishes real face photographs from AI-generated (deepfake) faces, built using Transfer Learning with a ResNet18 Convolutional Neural Network.

## Overview

This project fine-tunes a ResNet18 model, pretrained on ImageNet, to classify face images as **real** or **fake**. Training was done in two stages — first training only the final layer (baseline), then unfreezing the entire network for deeper fine-tuning — to capture the subtle artifacts left behind by GAN-based image generation.

## Dataset

**[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** (Kaggle)

- 70,000 real faces from [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- 70,000 fake faces generated using [StyleGAN](https://arxiv.org/abs/1812.04948)
- Pre-split: 100,000 train / 20,000 validation / 20,000 test

> The dataset is not included in this repository due to its size. Download it from the Kaggle link above.

## Approach

1. **Baseline (Stage 1)** — Only the final fully-connected layer trained, rest of ResNet18 frozen.
2. **Full fine-tuning (Stage 2)** — Entire network unfrozen, with layer-wise learning rates (smaller for early layers, larger for the final layer).

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 99.03% |
| Precision (Fake class) | 98.26% |
| Recall (Fake class) | 99.84% |
| F1-score (Fake class) | 99.05% |
| ROC-AUC | 0.9998 |

Out of 20,000 test images, only 193 were misclassified.

**Confusion Matrix (Test Set):**

|  | Predicted: Fake | Predicted: Real |
|---|---|---|
| **Actually Fake** | 9,984 | 16 |
| **Actually Real** | 177 | 9,823 |

## Limitation: Generalization Gap

The model performs excellently on StyleGAN-generated faces (its training distribution) but shows reduced accuracy on deepfakes produced by other generation methods (e.g., different GANs, diffusion models). This is a known challenge in deepfake detection research. Future work includes training on a broader mix of generation techniques.

## Tech Stack

- Python 3, PyTorch, torchvision
- scikit-learn, Matplotlib
- Trained on Google Colab / Kaggle Notebooks (GPU: NVIDIA T4)

## How to Run

1. Download the dataset from Kaggle (link above) and update the dataset path in the notebook.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Deepfake_Detection.ipynb` and run cells in order.

## Repository Structure

```
├── Deepfake_Detection.ipynb   # Main notebook: data loading, training, evaluation
├── requirements.txt           # Python dependencies
└── README.md
```

## Author

Shubhankar Kumar — M.Sc. Computer Science, Central University of Tamil Nadu
Internship at National Institute of Technology Puducherry, under Dr. Srinivasan A
