# MAI - IDL 2026 — Final Project Assignment

**Team Members:**
1. Harshavardini Gowrisankar — 10012956
2. Anshita Ramayanam — 10013473

---

## Overview

This repository contains a stabilized, benchmarked, and extended version of a deep learning pipeline for histology image classification. The original codebase was recovered in a broken state (see `AUDIT_LOG.md` for the full bug-by-bug changelog) and was fixed, then used to complete three exercises:

1. **Core Benchmark** — AlexNet, VGG16, and ResNet18 evaluated across four histology datasets, identifying the best-performing model per dataset against assignment-defined minimum accuracy targets.
2. **Green Initiative** — a lightweight custom architecture (`GreenNet`) designed to cut training time, memory footprint, and inference latency, benchmarked against the three baseline models.
3. **Data-Scarcity Transfer Learning** — using a ResNet18 model pretrained on the larger `orgs.pt` dataset as a donor, adapted to the small `organs.pt` dataset via frozen-backbone and fine-tuned transfer learning, compared against training from scratch.

## Repository Structure

```
.
├── Code/
│   ├── data.py                
│   ├── models.py              
│   ├── fit.py                   
│   ├── train.py                 
│   ├── transfer_learning.py     
│   └── config.json              
├── data/
│   ├── cells.pt
│   ├── chest.pt
│   ├── lesions.pt
│   ├── orgs.pt                  
│   └── organs.pt               
├── Audit.md                 
├── report_task1.md / .pdf      
├── report_task2.md / .pdf     
├── report_task3.md / .pdf       
└── README.md                    
```

## Prerequisites

- Python 3.10+
- PyTorch with CUDA support (recommended; CPU also works but is significantly slower)
- `scikit-learn` (for precision/recall/F1 metrics)

Install dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib
```

## Dataset Setup

Download the datasets and place them in the `data/` folder

| File | Samples | Classes | Channels | Notes |
|---|---|---|---|---|
| `cells.pt` | 13,671 train / 3,421 test | 8 | 3 (RGB) | Histopathology tissue images |
| `chest.pt` | 5,232 train / 624 test | 2 | 1 (grayscale) | Chest X-ray|
| `lesions.pt` | 8,010 train / 2,005 test | 7 | 3 (RGB) | Dermatoscopic skin lesion images |
| `orgs.pt` | ~15,000 train / ~8,200 test | 11 | 1 (grayscale) | Large organ dataset |
| `organs.pt` | 500 train / 200 test | 11 | 1 (grayscale) | Small organ dataset |

## How to Run

### Exercise 1 & 2 — Training a single model/dataset combination

Edit `Code/config.json` to select the dataset and model, then run:

```bash
cd Code
python train.py
```

Example `config.json`:
```json
{
    "DATA": "cells",
    "DATA_PATH": "data",
    "MODEL": "VGG16",
    "CHANNELS": 3,
    "NUM_CLASSES": 8,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.001,
    "ACTIVATION": "ReLU",
    "DROP": 0.5
}
```

`MODEL` accepts `AlexNet`, `VGG16`, `ResNet18`, or `GreenNet`. `CHANNELS` and `NUM_CLASSES` must match the selected `DATA` file (see table above).

### Exercise 3 — Transfer learning on organs.pt

First train a ResNet18 model on `orgs.pt` (the large dataset) and save its weights, then run:

```bash
cd Code
python transfer_learning.py
```

This loads the saved donor weights (`ResNet18_orgs_weights.pth`) into a fresh ResNet18, freezes all layers except the final classifier (toggle this block in `transfer_learning.py` to switch between frozen and fine-tuned modes), and trains on `organs.pt`.
