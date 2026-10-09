# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.transforms as T

from dataset import LogarithmicQCNNDataset
from model import HybridLogarithmicQCNN

def train_pipeline():
    # 1. Classical device selection targeting Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">>> Classical Acceleration Hardware: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(">>> Classical Acceleration Hardware: Central Processing Unit (CPU)")

    # 2. Data Setup
    print("Configuring Phase 2 Datasets...")
    raw_train_data = FakeData(size=10, image_size=(1, 28, 28), num_classes=2, transform=T.ToTensor())
    train_dataset = LogarithmicQCNNDataset(raw_train_data, patch_size=4, stride=2)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 3. Model & Optimization Initialization
    # Initialize the model structure
    model = HybridLogarithmicQCNN(num_classes=2)
    
    # CRITICAL: Only push classical sub-modules to Apple Silicon MPS.
    # This prevents PennyLane from trying to call CUDA operators on quantum simulators.
    model.classifier = model.classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Balanced Training Loop
    print("Beginning execution of validation training epoch...")
    model.train()
    
    for batch_idx, (patches, labels) in enumerate(train_loader):
        # Keep features on CPU so PennyLane can run the simulation seamlessly
        patches = patches.to(torch.device("cpu"))
        
        # Keep labels on the active acceleration target device
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        # We need to adapt model.py to handle the cross-device step cleanly
        logits = model(patches)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print("\n--- UNIFIED PYTORCH-MPS OPTIMIZATION PIPELINE PASSED ---")

if __name__ == "__main__":
    train_pipeline()

