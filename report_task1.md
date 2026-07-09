# Report — Task 1: Best Model Per Dataset Report
**Team Members:**  
1. Harshavardini Gowrisankar -10012956
2. Anshita Ramayanam - 10013473 
---

## cells.pt — Target: 90%

| Model | Test Accuracy |
|---|---|
| AlexNet | 94.91% |
| VGG16 | 96.14% |
| ResNet18 | 95.32% |

**Best model: VGG16, at 96.14% — crosses the 90% target.**

**Configuration Used:**

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

**Training Loss and Accuracy flow against Epochs — VGG16 on cells.pt :**

![VGG16 on cells.pt](charts/VGG16_cells_run1.png)

---

## chest.pt — Target: 87%

| Model | Test Accuracy |
|---|---|
| AlexNet | 84.46% |
| VGG16 | 91.99% |
| ResNet18 | 81.25% |

**Best model: VGG16, at 91.99% — crosses the 87% target.**

**Configuration Used:**

```json
{
    "DATA": "chest",
    "DATA_PATH": "data",
    "MODEL": "VGG16",
    "CHANNELS": 1,
    "NUM_CLASSES": 2,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.001,
    "ACTIVATION": "ReLU",
    "DROP": 0.5
}
```

**Training Loss and Accuracy flow against Epochs — VGG16 on chest.pt :**

![VGG16 on chest.pt](charts/VGG16_chest_run2.png)

---

## lesions.pt — Target: 67%

| Model | Test Accuracy |
|---|---|
| AlexNet | 74.56% |
| VGG16 | 71.42% |
| ResNet18 | 73.97% |

**Best model: AlexNet, at 74.56% — crosses the 67% target.**

**Configuration Used:**

```json
{
    "DATA": "lesions",
    "DATA_PATH": "data",
    "MODEL": "AlexNet",
    "CHANNELS": 3,
    "NUM_CLASSES": 7,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.001,
    "ACTIVATION": "ReLU",
    "DROP": 0.5
}
```

**Training Loss and Accuracy flow against Epochs — AlexNet on lesions.pt :**

![AlexNet on lesions.pt](charts/AlexNet_lesions_run2.png)

---

## orgs.pt — Target: 83%

| Model | Test Accuracy |
|---|---|
| AlexNet | 89.14% |
| VGG16 | 90.21% |
| ResNet18 | 92.04% |

**Best model: ResNet18, at 92.04% — crosses the 83% target.**

**Configuration Used:**

```json
{
    "DATA": "orgs",
    "DATA_PATH": "data",
    "MODEL": "ResNet18",
    "CHANNELS": 1,
    "NUM_CLASSES": 11,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.001,
    "ACTIVATION": "ReLU",
    "DROP": 0.5
}
```

**Training Loss and Accuracy flow against Epochs — ResNet18 on orgs.pt :**

![ResNet18 on orgs.pt](charts/ResNet18_orgs_run1.png)

---
