# src/dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

class LogarithmicQCNNDataset(Dataset):
    """
    Custom Dataset for Logarithmic Amplitude Encoding QCNN.
    Downsamples classical images to 14x14 and extracts L2-normalized 4x4 patches.
    """
    def __init__(self, base_dataset, patch_size=4, stride=2):
        """
        Args:
            base_dataset (Dataset): A standard PyTorch dataset (e.g., MNIST) 
                                   returning (image_tensor, label).
            patch_size (int): Spatial dimension of the square patch (Phase 2: 4).
            stride (int): Step size for the sliding window (default: 2).
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        
        # Phase 2 spatial constraint: Downsample to 14x14
        self.resize_transform = T.Compose([
            T.Grayscale(num_output_channels=1),  # Ensure single channel
            T.Resize((14, 14)),
            T.ToTensor() if not hasattr(base_dataset, 'transform') else lambda x: x
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Apply downsampling if not already handled by raw tensor operations
        if not isinstance(img, torch.Tensor):
            img = self.resize_transform(img)
        elif img.shape[-2:] != (14, 14):
            img = T.functional.resize(img, (14, 14))
            if img.shape[0] > 1:
                img = T.functional.rgb_to_grayscale(img)

        # Shape adjustment: (1, 14, 14) -> (1, 1, 14, 14) for unfold operation
        img_unsqueezed = img.unsqueeze(0)
        
        # Extract 4x4 spatial patches using a sliding window
        # Output shape: (1, patch_size * patch_size, num_patches)
        patches = F.unfold(
            img_unsqueezed, 
            kernel_size=self.patch_size, 
            stride=self.stride
        )
        
        # Squeeze batch dimension and transpose to get (num_patches, 16)
        patches = patches.squeeze(0).t()
        
        # Safe L2 Normalization to prepare for qml.AmplitudeEmbedding
        norms = torch.norm(patches, p=2, dim=1, keepdim=True)
        
        # Avoid division by zero for completely flat/black background patches
        norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
        normalized_patches = patches / norms
        
        return normalized_patches, label

# Quick Sanity Check Execution Block
if __name__ == "__main__":
    from torchvision.datasets import FakeData
    
    print("Initializing dummy dataset for verification...")
    raw_data = FakeData(size=5, image_size=(1, 28, 28), num_classes=2, transform=T.ToTensor())
    qcnn_data = LogarithmicQCNNDataset(raw_data, patch_size=4, stride=2)
    
    patches, label = qcnn_data[0]
    print("\n--- SANITY CHECK PASSED ---")
    print(f"Output patches tensor shape: {patches.shape} (Expected: [36, 16])")
    print(f"Sample patch vector L2 norm: {torch.norm(patches[0], p=2).item():.4f} (Expected: 1.0000)")
