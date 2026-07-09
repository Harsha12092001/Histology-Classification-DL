"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer
import time

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader   = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])  #Fix variable assignment for test_loader in data loading

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=config["DROP"], activation_str=config["ACTIVATION"]).to(device)    #added drop_rate and activation as configurable one
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Start training and measure time
    train_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    
    # End training and measure time
    train_end = time.perf_counter()
    training_time = train_end - train_start
    print(f"Total Training Time: {training_time:.2f} seconds")

    # Measure peak memory usage during training
    if torch.cuda.is_available():
        peak_train_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak Training Memory: {peak_train_mem_mb:.1f} MB")

    #logging of test loss and accuracy after training
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")

    # Save the model weights after training
    torch.save(model.state_dict(), f"{config['MODEL']}_{config['DATA']}_weights.pth")
    print(f"Saved weights to {config['MODEL']}_{config['DATA']}_weights.pth")

    # Evaluate on the test set and compute metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:  # Use test_loader, not val_loader
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
        
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n=== Test Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # ---- Inference latency + inference-phase peak memory ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # reset so this only measures inference

    sample = next(iter(test_loader))[0][0:1].to(device)

    with torch.no_grad():
        _ = model(sample)  # warm-up, not timed

    num_runs = 50
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample)
    end = time.perf_counter()

    latency_ms = (end - start) / num_runs * 1000
    print(f"Inference Latency: {latency_ms:.3f} ms per sample")

    if torch.cuda.is_available():
        peak_infer_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak Inference Memory: {peak_infer_mem_mb:.1f} MB")

if __name__ == "__main__":
    main()