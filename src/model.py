# src/model.py
print(">>> DEBUG: Script execution started successfully!")
import torch
import torch.nn as nn
import pennylane as qml
from quantum_filter import quantum_conv_circuit

class HybridLogarithmicQCNN(nn.Module):
    """
    Hybrid Classical-Quantum Convolutional Neural Network
    utilising Logarithmic Amplitude Encoding.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Define the weight shapes expected by your quantum variational ansatz
        # Our quantum filter circuit uses exactly 12 parameter weights
        weight_shapes = {"weights": (12,)}
        
        # Wrap the QNode into a native PyTorch nn.Module layer
        # Output shapes from this layer will match the number of wire measurements: (4,)
        self.quantum_filter = qml.qnn.TorchLayer(quantum_conv_circuit, weight_shapes)
        
        # Classical fully connected head
        # Input size: 36 patches * 4 quantum features per patch = 144 features
        self.classifier = nn.Linear(36 * 4, num_classes)

    def forward(self, x):
        """
        Args:
            x (tensor): Shape [Batch, 36, 16] representing the image patches.
        Returns:
            tensor: Class logits of shape [Batch, num_classes].
        """
        # Explicitly extract the individual scalar dimensions from the input tensor
        batch_size, num_patches, num_features = x.shape  # Expected: [2, 36, 16]
        
        # 1. Flatten patch and batch dimensions for the quantum filter [Batch * 36, 16]
        x_flat = x.view(-1, num_features).to(torch.device("cpu"))
        
        # 2. Process quantum simulation on the CPU
        # Output shape from quantum_filter: [Batch * 36, 4]
        q_features_flat = self.quantum_filter(x_flat)
        
        # 3. Reshape features back to separate out the batch dimension cleanly
        # New shape: [Batch, 36 * 4] -> [2, 144]
        q_features = q_features_flat.view(batch_size, num_patches * 4)
        
        # 4. Dynamically push features to match whichever device the classical head is using (MPS)
        target_device = next(self.classifier.parameters()).device
        q_features = q_features.to(target_device)
        
        # 5. Execute the classical forward pass on your GPU
        logits = self.classifier(q_features)
        
        return logits



# Quick Sanity Check Execution Block
if __name__ == "__main__":
    print("Initializing Hybrid Logarithmic QCNN Architecture...")
    model = HybridLogarithmicQCNN(num_classes=2)
    print(model)
    
    # Simulate a batch of 2 images prepped by our dataset pipeline
    # Shape: [Batch size, Num Patches, Features per patch]
    dummy_batch = torch.randn(2, 36, 16)
    dummy_batch /= torch.norm(dummy_batch, p=2, dim=2, keepdim=True)
    
    print("\nExecuting forward pass...")
    out_logits = model(dummy_batch)
    
    print("\n--- MODEL ARCHITECTURE PASSED ---")
    print(f"Input batch shape : {dummy_batch.shape}")
    print(f"Output logits shape: {out_logits.shape} (Expected: [2, 2])")
