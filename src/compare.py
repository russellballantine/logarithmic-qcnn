# src/compare.py
print(">>> DEBUG: Launching stabilized, NaN-protected QCNN Benchmark Suite...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from dataset import LogarithmicQCNNDataset

# --- NATIVE PENNYLANE CIRCUITS DEFINED INLINE FOR ISOLATED SAFETY ---
import pennylane as qml

dev_quantum = qml.device("default.qubit", wires=4)

@qml.qnode(dev_quantum, interface="torch")
def angle_conv_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
    for i in range(4):
        qml.RY(weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

@qml.qnode(dev_quantum, interface="torch")
def amp_conv_circuit(inputs, weights):
    # Enforce normalize=False because our forward pass handles the ground-state logic explicitly
    qml.AmplitudeEmbedding(inputs, wires=range(4), normalize=False)
    for i in range(4):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[i + 4], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    for i in range(4):
        qml.RY(weights[i + 8], wires=i)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]


# --- PHASE 1: ANGLE ENCODING BASELINE ---
class Phase1AngleQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(4))
        self.classifier = nn.Linear(36 * 4, 2)

    def forward(self, x):
        x_cpu = x.to("cpu")
        batch_size, num_patches, _ = x_cpu.shape
        x_4 = x_cpu[:, :, :4]
        
        batch_features = []
        for b in range(batch_size):
            image_features = []
            for p in range(num_patches):
                q_out = angle_conv_circuit(x_4[b, p], self.q_weights.to("cpu"))
                image_features.append(torch.stack(q_out))
            batch_features.append(torch.cat(image_features))
            
        q_features_cpu = torch.stack(batch_features)
        target_device = next(self.classifier.parameters()).device
        return self.classifier(q_features_cpu.float().to(target_device))


# --- PHASE 2: AMPLITUDE ENCODING WITH NAN-SAFETY PROTECTION ---
class Phase2AmplitudeQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(12))
        self.classifier = nn.Linear(36 * 4, 2)

    def forward(self, x):
        batch_size, num_patches, num_features = x.shape
        x_flat = x.view(-1, num_features).to("cpu")
        
        # --- THE QUANTUM VACUUM SAFETY INTERCEPT ---
        # If an MNIST patch is entirely black, its norm is 0.0.
        # We replace it with a valid ground state vector [1, 0, 0, ..., 0] (Norm = 1.0)
        # This completely prevents any division-by-zero NaN generation loops.
        norms = torch.norm(x_flat, p=2, dim=1)
        zero_mask = (norms == 0.0)
        if zero_mask.any():
            x_flat = x_flat.clone()
            x_flat[zero_mask] = 0.0
            x_flat[zero_mask, 0] = 1.0  # Force ground state injection profile
            
        # Process patches using native call arrays
        batch_features = []
        for i in range(x_flat.shape[0]):
            q_out = amp_conv_circuit(x_flat[i], self.q_weights.to("cpu"))
            batch_features.append(torch.stack(q_out))
            
        q_features_flat = torch.stack(batch_features)
        q_features_spatial = q_features_flat.view(batch_size, num_patches * 4)
        
        target_device = next(self.classifier.parameters()).device
        return self.classifier(q_features_spatial.float().to(target_device))


# --- SYSTEM MANAGEMENT EXECUTION PARADIGM ---
def run_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> Active Benchmark Target Hardware: {device}")
    
    transform = T.Compose([T.Grayscale(num_output_channels=1), T.Resize((14, 14)), T.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # STEP A: Filter for digits 3 and 8, then properly extract the index values using .squeeze()
    complex_indices = ((mnist_train.targets == 3) | (mnist_train.targets == 8)).nonzero().squeeze()
    subset_idx = complex_indices[:200]  # Take a safe slice of 200 samples
    train_subset = Subset(mnist_train, subset_idx)
    
    # Dynamically compute training/validation lengths to prevent split mismatch exceptions
    train_len = int(len(train_subset) * 0.75)
    val_len = len(train_subset) - train_len
    train_set, val_set = torch.utils.data.random_split(train_subset, [train_len, val_len])
    
    class TargetMapDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset): self.base_dataset = base_dataset
        def __len__(self): return len(self.base_dataset)
        def __getitem__(self, idx):
            patches, label = self.base_dataset[idx]
            return patches, (0 if label == 3 else 1)

    qcnn_train_set = TargetMapDataset(LogarithmicQCNNDataset(train_set, patch_size=4, stride=2))
    qcnn_val_set = TargetMapDataset(LogarithmicQCNNDataset(val_set, patch_size=4, stride=2))
    
    train_loader = DataLoader(qcnn_train_set, batch_size=10, shuffle=True)
    val_loader = DataLoader(qcnn_val_set, batch_size=10, shuffle=False)

    epochs = 5  
    metrics = {"angle_acc": [], "amp_acc": []}
    criterion = nn.CrossEntropyLoss()

    # --- TRAIN ANGLE ---
    print("\nTraining Phase 1: Angle Encoding Baseline...")
    model_angle = Phase1AngleQCNN().to(device)
    opt_angle = optim.Adam(model_angle.parameters(), lr=0.005)

    for epoch in range(epochs):
        model_angle.train()
        loss_accum = 0
        for patches, labels in train_loader:
            labels = labels.to(device)
            opt_angle.zero_grad()
            loss = criterion(model_angle(patches), labels)
            loss.backward()
            opt_angle.step()
            loss_accum += loss.item()
        
        model_angle.eval()
        correct = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                correct += (model_angle(patches).argmax(dim=1) == labels.to(device)).sum().item()
        acc = correct / len(val_set)
        metrics["angle_acc"].append(acc)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss_accum/len(train_loader):.4f} | Val Accuracy: {acc:.4f}")

    # --- TRAIN AMPLITUDE ---
    print("\nTraining Phase 2: Logarithmic Amplitude Encoding...")
    model_amp = Phase2AmplitudeQCNN().to(device)
    opt_amp = optim.Adam(model_amp.parameters(), lr=0.005)

    for epoch in range(epochs):
        model_amp.train()
        loss_accum = 0
        for patches, labels in train_loader:
            labels = labels.to(device)
            opt_amp.zero_grad()
            loss = criterion(model_amp(patches), labels)
            loss.backward()
            opt_amp.step()
            loss_accum += loss.item()
        
        model_amp.eval()
        correct = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                correct += (model_amp(patches).argmax(dim=1) == labels.to(device)).sum().item()
        acc = correct / len(val_set)
        metrics["amp_acc"].append(acc)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss_accum/len(train_loader):.4f} | Val Accuracy: {acc:.4f}")

    # --- PLOT GENERATION ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics["angle_acc"], 'o-', color='orange', linewidth=2, label='Phase 1: Angle Encoding (2x2 Patches)')
    plt.plot(range(1, epochs + 1), metrics["amp_acc"], 's-', color='teal', linewidth=2, label='Phase 2: Logarithmic Amplitude (4x4 Patches)')
    plt.title('Quantum Crossover Analysis: MNIST Digits 3 vs 8 (NaN Protected)', fontsize=12, fontweight='bold')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Accuracy %')
    plt.ylim(0.40, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig("encoding_comparison_chart.png", dpi=300)
    print("\n>>> Stabilized, NaN-protected plot saved as: encoding_comparison_chart.png")

if __name__ == "__main__":
    run_comparison()



