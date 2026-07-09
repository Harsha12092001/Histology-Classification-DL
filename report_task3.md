# Report — Task 3: Transfer Learning Report
**Team Members:**  
1. Harshavardini Gowrisankar -10012956
2. Anshita Ramayanam - 10013473 
---

## Why ResNet18 Trained on orgs.pt Was Chosen as the Donor

- **Same domain, not a cross-domain transfer.** `orgs.pt` and `organs.pt` are both grayscale organ images across the same 11 classes — the features learned on the larger dataset are directly relevant, since it's the same task with fewer examples.
- **ResNet18 was the strongest performer on orgs.pt in Exercise 1** — 92.04–92.67% accuracy, ahead of VGG16 (~90%) and AlexNet (~89%). Its weights represent the most reliable feature extractor available to reuse.
- Donor weights trained on `orgs.pt` were saved to `ResNet18_orgs_weights.pth`, then loaded into a freshly built ResNet18 (`in_channels=1, num_classes=11`) before being adapted to `organs.pt`.

## Benchmark Matrix

| Approach | Epochs | Test Accuracy |
|---|---|---|
| From scratch | 15 | 47.00% |
| Transfer (frozen) | 15 | 57.00% |
| Transfer (fine-tuned) | 15 | 61.00% |


## From Scratch (15 epochs) — 47.00% Test Accuracy

```
Epoch [01/15] | Train Loss: 2.0203 - Train Acc: 34.67% | Val Loss: 2.9668 - Val Acc: 8.00%
Epoch [05/15] | Train Loss: 1.1367 - Train Acc: 59.56% | Val Loss: 3.0993 - Val Acc: 18.00%
Epoch [10/15] | Train Loss: 0.9924 - Train Acc: 61.78% | Val Loss: 1.4763 - Val Acc: 56.00%
Epoch [15/15] | Train Loss: 0.9181 - Train Acc: 65.11% | Val Loss: 1.6065 - Val Acc: 50.00%
--------------------------------------------------
Test Loss: 2.0441 - Test Acc: 47.00%
```

![ResNet18 from scratch on organs.pt](charts/scratch_15.png)

## Transfer Learning — Frozen Backbone — 57.00% Test Accuracy

```
Epoch [01/15] | Train Loss: 1.8716 - Train Acc: 63.78% | Val Loss: 1.6114 - Val Acc: 64.00%
Epoch [05/15] | Train Loss: 1.2097 - Train Acc: 71.56% | Val Loss: 1.4038 - Val Acc: 68.00%
Epoch [10/15] | Train Loss: 0.9087 - Train Acc: 72.22% | Val Loss: 1.2110 - Val Acc: 70.00%
Epoch [15/15] | Train Loss: 0.7268 - Train Acc: 74.44% | Val Loss: 0.9257 - Val Acc: 70.00%
--------------------------------------------------
Transfer Learning (Frozen) - Test Loss: 1.5762 - Test Acc: 57.00%
```

![Transfer learning frozen backbone on organs.pt](charts/frozen.png)

## Transfer Learning — Fine-Tuned — 61.00% Test Accuracy

```
Epoch [01/15] | Train Loss: 1.7112 - Train Acc: 66.44% | Val Loss: 1.5081 - Val Acc: 72.00%
Epoch [05/15] | Train Loss: 0.6565 - Train Acc: 81.78% | Val Loss: 0.7883 - Val Acc: 80.00%
Epoch [10/15] | Train Loss: 0.3887 - Train Acc: 87.33% | Val Loss: 0.6298 - Val Acc: 78.00%
Epoch [15/15] | Train Loss: 0.2415 - Train Acc: 91.78% | Val Loss: 0.4497 - Val Acc: 86.00%
--------------------------------------------------
Transfer Learning (Fine-Tuning) - Test Loss: 1.7667 - Test Acc: 61.00%
```

![Transfer learning fine-tuned on organs.pt](charts/finetune.png)

## Data-Scarcity Post-Mortem

- **From-scratch training is highly unstable.** Validation loss standard deviation across the 15-epoch scratch run is **2.79**, vs. **0.24** (frozen) and **0.30** (fine-tuned) — roughly **10x more volatile**. Validation loss spikes as high as 9.07 in epoch 3, drops to 1.48 by epoch 9, then back up to 8.72 in epoch 13.
- **Validation accuracy never settles for the scratch model** — it bounces between 8%, 60%, 28%, and 50% across the run, with no clear stable trend.
- **Both transfer approaches are far more stable.** Frozen-backbone validation accuracy stays in a tight 64–74% band. Fine-tuned validation accuracy trends steadily upward with only minor noise.
- **Fine-tuning beats frozen transfer (61% vs. 57%), and both beat from-scratch (47%).** A frozen backbone can only adapt the final classifier to the new class distribution; fine-tuning lets the earlier feature-extraction layers adjust too, at a low learning rate so they refine rather than overwrite what was already learned.

## Recommendation

**Fine-tuned transfer learning is the clear winner** — highest accuracy (61%, well above the 40% target), fastest-converging, and most stable validation behavior of the three viable approaches. It should be the recommended strategy for `organs.pt`.
