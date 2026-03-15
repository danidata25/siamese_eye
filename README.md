# SiameseEye

A deep learning model for **face verification** using a Siamese Neural Network, trained and evaluated on the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset.

---

## Overview

SiameseEye learns to determine whether two face images belong to the same person. Instead of classifying identities, it learns a similarity metric — a hallmark of **one-shot learning**.

The architecture is based on the original [Siamese Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) paper by Koch et al. (2015), adapted for the LFW face verification benchmark.

---

## Architecture

- **Twin CNN branches** sharing weights, each processing one input image
- **Convolutional layers**: 4 conv blocks with BatchNorm + ReLU + MaxPool
- **Embedding layer**: 4096-dimensional fully connected layer
- **Scoring head**: Learns a weighted L1 distance between embeddings
- **Loss**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)

```
Input A ──► CNN ──► FC(4096) ──┐
                                ├──► |diff| ──► Linear(1) ──► similarity score
Input B ──► CNN ──► FC(4096) ──┘
```

---

## Dataset

**LFW (Labeled Faces in the Wild)** — a benchmark dataset for face verification:
- 13,000+ face images of ~5,700 public figures
- Official train/test pair splits from [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/)

Images are:
- Resized to **105×105**
- Converted to **grayscale**
- Augmented with random affine transforms (rotation, translation, scaling, shear)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the [LFW dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and extract it to a directory (e.g., `data/lfw2/`).

The pair split files are downloaded automatically.

### 3. Run
```bash
python siamese_eye.py
```
Update `data_directory` in the script to point to your LFW folder.

---

## Training

Uses **PyTorch Lightning**:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| LR | 1e-3 |
| LR Scheduler | ExponentialLR (γ=0.99) |
| Epochs | 100 |
| Batch Size | 128 |
| Loss | BCEWithLogitsLoss |

---

## Results

Model is evaluated on accuracy (threshold = 0.5 on sigmoid output) per epoch on both training and validation splits.

---

## Project Structure

```
SiameseEye/
├── siamese_eye.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-Shot Image Recognition.* ICML Deep Learning Workshop.
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)
- [Siamese Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
