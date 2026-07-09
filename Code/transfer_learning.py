import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Build a ResNet18 shaped like the donor (orgs.pt), then load its trained weights
model = models.ResNet18(in_channels=1, num_classes=11, activation_str="ReLU", drop_rate=0.5)
model.load_state_dict(torch.load("ResNet18_orgs_weights.pth"))

# Step 2: Freeze everything except the last layer (classifier), so only that layer learns
#for name, param in model.named_parameters():
 #   if "classifier" not in name:
  #      param.requires_grad = False

model = model.to(device)

# Step 3: Load the new small dataset
train_loader, val_loader, test_loader = get_loaders(data="organs", data_path="data", batch_size=32)

# Step 4: Train normally, but only unfrozen parameters actually update
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
trainer = Trainer(model, criterion, optimizer, device)
trainer.fit(train_loader, val_loader, epochs=15)

# Step 5: Test it
test_loss, test_acc = trainer.evaluate(test_loader)
print(f"Transfer Learning (Fine-Tuning) - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")