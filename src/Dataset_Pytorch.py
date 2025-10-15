# src/dataset.py
"""
PyTorch Dataset for acoustic spectrogram classification.
Integrates augmentation pipeline and normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, cast

from augmentation import get_augmentation_pipeline


class AcousticSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading and augmenting acoustic spectrograms.
    
    Features:
    - Loads pre-computed log-mel spectrograms (.npy files)
    - Applies normalization using training set statistics
    - Applies on-the-fly augmentation (train mode only)
    - Handles class label encoding
    """
    
    def __init__(
        self,
        metadata_path: str,
        norm_stats_path: str,
        mode: str = 'train',
        transform=None
    ):
        """
        Args:
            metadata_path: Path to CSV with columns [FeaturePath, VesselType, ...]
            norm_stats_path: Path to JSON with normalization statistics
            mode: 'train', 'val', or 'test'
            transform: Optional additional transforms (use None for default augmentation)
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)
        
        # Load normalization statistics
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = stats['mean']
        self.std = stats['std']
        
        # Setup augmentation
        if transform is None:
            self.transform = get_augmentation_pipeline(mode)
        else:
            self.transform = transform
        
        # Create label encoding
        self.classes = sorted(self.metadata['VesselType'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logging.info(f"Initialized {mode} dataset:")
        logging.info(f"  Samples: {len(self.metadata)}")
        logging.info(f"  Classes: {self.classes}")
        logging.info(f"  Class distribution:")
        for cls, count in self.metadata['VesselType'].value_counts().items():
            logging.info(f"    {cls}: {count} ({count/len(self.metadata)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            spectrogram: Tensor of shape (1, H, W) - normalized and augmented
            label: Integer class label
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        spec_path = row['FeaturePath']
        vessel_type = row['VesselType']
        label = self.class_to_idx[vessel_type]
        
        # Load spectrogram
        try:
            spectrogram = np.load(spec_path).astype(np.float32)
        except FileNotFoundError:
            logging.error(f"File not found: {spec_path}")
            raise
        
        # Normalize using training set statistics
        spectrogram = (spectrogram - self.mean) / (self.std + 1e-8)
        
        # Apply augmentation (only for training)
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        # Convert to tensor and add channel dimension
        # Shape: (H, W) -> (1, H, W)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
        
        return spectrogram, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        Useful for weighted loss functions.
        
        Returns:
            Tensor of shape (n_classes,) with inverse frequency weights
        """
        class_counts = self.metadata['VesselType'].value_counts()
        weights = []
        
        for cls in self.classes:
            count = class_counts.get(cls, 1)
            weight = len(self.metadata) / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample (for debugging)."""
        row = self.metadata.iloc[idx]
        return {
            'index': idx,
            'path': row['FeaturePath'],
            'vessel_type': row['VesselType'],
            'label': self.class_to_idx[row['VesselType']],
            'vessel_group_key': row.get('VesselGroupKey', 'N/A')
        }


def create_dataloaders(
    train_meta_path: str,
    val_meta_path: str,
    test_meta_path: str,
    norm_stats_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train/val/test dataloaders.
    
    Args:
        train_meta_path: Path to train metadata CSV
        val_meta_path: Path to validation metadata CSV
        test_meta_path: Path to test metadata CSV
        norm_stats_path: Path to normalization stats JSON
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = AcousticSpectrogramDataset(
        train_meta_path,
        norm_stats_path,
        mode='train'
    )
    
    val_dataset = AcousticSpectrogramDataset(
        val_meta_path,
        norm_stats_path,
        mode='val'
    )
    
    test_dataset = AcousticSpectrogramDataset(
        test_meta_path,
        norm_stats_path,
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logging.info(f"Created dataloaders with batch_size={batch_size}")
    logging.info(f"  Train batches: {len(train_loader)}")
    logging.info(f"  Val batches: {len(val_loader)}")
    logging.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_dataset():
    """Test dataset functionality with dummy data."""
    import tempfile
    import os
    
    logging.info("Testing dataset functionality...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy spectrograms
        n_samples = 20
        spec_paths = []
        for i in range(n_samples):
            spec = np.random.randn(128, 256).astype(np.float32)
            path = tmpdir / f"spec_{i}.npy"
            np.save(path, spec)
            spec_paths.append(str(path))
        
        # Create dummy metadata
        metadata = pd.DataFrame({
            'FeaturePath': spec_paths,
            'VesselType': ['Cargo', 'Tanker', 'Passenger', 'Tug'] * 5,
            'VesselGroupKey': [f'vessel_{i//5}' for i in range(n_samples)]
        })
        meta_path = tmpdir / 'metadata.csv'
        metadata.to_csv(meta_path, index=False)
        
        # Create dummy normalization stats
        stats = {'mean': 0.0, 'std': 1.0}
        stats_path = tmpdir / 'norm_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        # Test dataset creation
        dataset = AcousticSpectrogramDataset(
            str(meta_path),
            str(stats_path),
            mode='train'
        )
        
        assert len(dataset) == n_samples, f"Expected {n_samples}, got {len(dataset)}"
        
        # Test __getitem__
        spec, label = dataset[0]
        assert isinstance(spec, torch.Tensor), "Spectrogram should be a tensor"
        assert spec.shape[0] == 1, "Should have channel dimension"
        assert isinstance(label, int), "Label should be an integer"
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_specs, batch_labels = next(iter(loader))
        
        assert batch_specs.shape[0] == 4, "Batch size should be 4"
        assert batch_labels.shape[0] == 4, "Batch labels size should be 4"
        
        # Test class weights
        weights = dataset.get_class_weights()
        assert len(weights) == len(dataset.classes), "Weight dimension mismatch"
        
        logging.info("✓ All dataset tests passed!")


def visualize_batch(
    dataloader: DataLoader,
    n_samples: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of spectrograms from the dataloader.
    
    Args:
        dataloader: DataLoader to sample from
        n_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("matplotlib not available for visualization")
        return
    
    # Get one batch
    specs, labels = next(iter(dataloader))
    specs = specs[:n_samples]
    labels = labels[:n_samples]
    
    # Setup plot
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    dataset = cast(AcousticSpectrogramDataset, dataloader.dataset)
    
    # Plot each sample
    for i in range(n_samples):
        spec = specs[i].squeeze().numpy()  # Remove channel dim
        label = labels[i].item()
        
        axes[i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f'Class: {dataset.idx_to_class[label]}')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved batch visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_dataset()
    logging.info("Dataset module ready for use! ✨")