# src/augmentation.py
"""
Data augmentation module for acoustic spectrograms.
Implements SpecAugment and additional audio-specific augmentations.
"""

import numpy as np
import random
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)

# Import local modules and configuration
try:
    # Import all necessary parameters from the single source of truth
    from dir_train_config import (
        TIME_MASK_RATIO_MIN,TIME_MASK_RATIO_MAX,N_TIME_MASKS,
        FREQ_MASK_RATIO_MIN,FREQ_MASK_RATIO_MAX,N_FREQ_MASKS,
        TIME_SHIFT_RANGE,ADD_NOISE,NOISE_STD,AUGMENTATION_PROBABILITY
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure you're running from the project root and all modules are available.")
    #sys.exit(1)

class SpectrogramAugmenter:
    """
    Comprehensive augmentation for log-mel spectrograms.
    
    Implements SpecAugment (time & frequency masking) plus additional
    augmentations suitable for acoustic vessel classification.
    """
    
    def __init__(
        self,
        time_mask_ratio: Tuple[float, float] = (TIME_MASK_RATIO_MIN, TIME_MASK_RATIO_MAX),
        freq_mask_ratio: Tuple[float, float] = (FREQ_MASK_RATIO_MIN, FREQ_MASK_RATIO_MAX),
        n_time_masks: int = N_TIME_MASKS,
        n_freq_masks: int = N_FREQ_MASKS,
        time_shift_range: int = TIME_SHIFT_RANGE,
        add_noise: bool = ADD_NOISE,
        noise_std: float = NOISE_STD,
        apply_probability: float = AUGMENTATION_PROBABILITY,
        mask_value: float = 0.0
    ):
        """
        Args:
            time_mask_ratio: (min, max) ratio of time frames to mask
            freq_mask_ratio: (min, max) ratio of frequency bins to mask
            n_time_masks: Number of time masks to apply
            n_freq_masks: Number of frequency masks to apply
            time_shift_range: Max frames to shift in time (±)
            add_noise: Whether to add Gaussian noise
            noise_std: Standard deviation of Gaussian noise
            apply_probability: Probability of applying augmentation
            mask_value: Value to use for masking (0.0 for log-mel spectrograms)
        """
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.time_shift_range = time_shift_range
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.apply_probability = apply_probability
        self.mask_value = mask_value
        
        logging.info(f"Initialized SpectrogramAugmenter with:")
        logging.info(f"  Time mask ratio: {time_mask_ratio}")
        logging.info(f"  Freq mask ratio: {freq_mask_ratio}")
        logging.info(f"  Time masks: {n_time_masks}, Freq masks: {n_freq_masks}")
        logging.info(f"  Apply probability: {apply_probability}")
    
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a spectrogram.
        
        Args:
            spectrogram: Input spectrogram (H, W) or (C, H, W)
        
        Returns:
            Augmented spectrogram with same shape
        """
        # Decide whether to apply augmentation
        if random.random() > self.apply_probability:
            return spectrogram
        
        # Work with copy to avoid modifying original
        spec = spectrogram.copy()
        
        # Handle both (H, W) and (C, H, W) formats
        if spec.ndim == 3:
            # Assume (C, H, W) - process each channel
            augmented_channels = []
            for channel in spec:
                aug_channel = self._augment_single_channel(channel)
                augmented_channels.append(aug_channel)
            spec = np.stack(augmented_channels, axis=0)
        else:
            # (H, W) format
            spec = self._augment_single_channel(spec)
        
        return spec
    
    def _augment_single_channel(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single channel spectrogram.
        
        CRITICAL ORDER:
        1. Add noise FIRST (so it doesn't corrupt masks)
        2. Time shifting (affects whole spectrogram)
        3. Time masking (vertical strips)
        4. Frequency masking (horizontal strips)
        """
        # 1. Add noise FIRST (before masking)
        if self.add_noise:
            spec = self._add_gaussian_noise(spec)
        
        # 2. Time shifting
        spec = self._time_shift(spec)
        
        # 3. Time masking
        spec = self._time_mask(spec)
        
        # 4. Frequency masking
        spec = self._freq_mask(spec)
        
        return spec
    
    def _time_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply time masking (vertical strips).
        Simulates brief signal dropouts or interference.
        """
        n_frames = spec.shape[1]  # Width (time dimension)
        
        for _ in range(self.n_time_masks):
            # Random mask width
            mask_ratio = random.uniform(*self.time_mask_ratio)
            mask_width = max(1, int(n_frames * mask_ratio))
            
            # Random starting position
            if n_frames > mask_width:  # Fixed: was n_frames - mask_width > 0
                mask_start = random.randint(0, n_frames - mask_width)
                spec[:, mask_start:mask_start + mask_width] = self.mask_value
        
        return spec
    
    def _freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply frequency masking (horizontal strips).
        Simulates frequency-selective fading.
        """
        n_bins = spec.shape[0]  # Height (frequency dimension)
        
        for _ in range(self.n_freq_masks):
            # Random mask height
            mask_ratio = random.uniform(*self.freq_mask_ratio)
            mask_height = max(1, int(n_bins * mask_ratio))
            
            # Random starting position
            if n_bins > mask_height:  # Fixed: was n_bins - mask_height > 0
                mask_start = random.randint(0, n_bins - mask_height)
                spec[mask_start:mask_start + mask_height, :] = self.mask_value
        
        return spec
    
    def _time_shift(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply random time shift.
        Simulates temporal misalignment or phase shifts.
        """
        if self.time_shift_range == 0:
            return spec
        
        shift = random.randint(-self.time_shift_range, self.time_shift_range)
        
        if shift == 0:
            return spec
        
        # Roll along time axis with zero-padding at boundaries
        shifted = np.roll(spec, shift, axis=1)
        
        if shift > 0:
            # Shifted right: zero-pad left side
            shifted[:, :shift] = self.mask_value
        else:
            # Shifted left: zero-pad right side
            shifted[:, shift:] = self.mask_value
        
        return shifted.astype(spec.dtype)
    
    def _add_gaussian_noise(self, spec: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to simulate environmental noise.
        """
        noise = np.random.normal(0, self.noise_std, spec.shape).astype(spec.dtype)
        return spec + noise


class NoAugmentation(SpectrogramAugmenter):
    """Identity augmentation (no transformation) for validation/test sets."""
    
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        return spectrogram


def get_augmentation_pipeline(mode: str = 'train'):
    """
    Factory function to get appropriate augmentation pipeline.
    
    Args:
        mode: 'train', 'val', or 'test'
    
    Returns:
        Augmentation pipeline
    """
    if mode == 'train':
        return SpectrogramAugmenter(
            time_mask_ratio= (TIME_MASK_RATIO_MIN, TIME_MASK_RATIO_MAX),
            freq_mask_ratio=(FREQ_MASK_RATIO_MIN, FREQ_MASK_RATIO_MAX),
            n_time_masks=N_TIME_MASKS,
            n_freq_masks=N_FREQ_MASKS,
            time_shift_range= TIME_SHIFT_RANGE,
            add_noise=ADD_NOISE,
            noise_std= NOISE_STD,
            apply_probability= AUGMENTATION_PROBABILITY
        )
    else:
        # No augmentation for validation and test
        return NoAugmentation()


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_augmentations(
    spectrogram: np.ndarray,
    augmenter: SpectrogramAugmenter,
    n_examples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize multiple augmented versions of a spectrogram.
    
    Args:
        spectrogram: Original spectrogram (H, W)
        augmenter: Augmentation pipeline
        n_examples: Number of augmented examples to generate
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(2, n_examples + 1, figsize=(20, 8))
    
    # Plot original
    axes[0, 0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Generate and plot augmented versions
    for i in range(n_examples):
        aug_spec = augmenter(spectrogram)
        
        # Top row: Augmented spectrogram
        axes[0, i + 1].imshow(aug_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i + 1].set_title(f'Augmented {i+1}', fontsize=12)
        axes[0, i + 1].axis('off')
        
        # Bottom row: Difference map
        diff = np.abs(spectrogram - aug_spec)
        im = axes[1, i + 1].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[1, i + 1].set_title(f'Difference {i+1}', fontsize=10)
        axes[1, i + 1].axis('off')
        plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_augmentation():
    """
    Test augmentation pipeline with synthetic data.
    Validates that augmentations preserve shape and data ranges.
    """
    logging.info("Testing augmentation pipeline...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create synthetic spectrogram
    test_spec = np.random.randn(128, 256).astype(np.float32)
    
    # Initialize augmenter with noise DISABLED for testing
    augmenter = SpectrogramAugmenter(
        time_mask_ratio=(0.1, 0.15),
        freq_mask_ratio=(0.1, 0.15),
        n_time_masks=2,
        n_freq_masks=2,
        add_noise=False,  # CRITICAL: Disable noise for testing masks
        apply_probability=1.0
    )
    
    # Test multiple augmentations
    n_tests = 10
    masks_detected = 0
    
    for i in range(n_tests):
        aug_spec = augmenter(test_spec)
        
        # Validate shape preservation
        assert aug_spec.shape == test_spec.shape, \
            f"Shape mismatch: {aug_spec.shape} vs {test_spec.shape}"
        
        # Validate data type
        assert aug_spec.dtype == test_spec.dtype, \
            f"Dtype mismatch: {aug_spec.dtype} vs {test_spec.dtype}"
        
        # Check that masking was applied
        # Use np.isclose for floating point comparison with tolerance
        masked_values = np.sum(np.isclose(aug_spec, augmenter.mask_value, atol=1e-6))
        
        if masked_values > 0:
            masks_detected += 1
            logging.info(f"  Test {i+1}: {masked_values} masked values detected")
    
    # At least 80% of tests should have detected masks
    assert masks_detected >= n_tests * 0.8, \
        f"Only {masks_detected}/{n_tests} tests detected masks. Expected at least {int(n_tests*0.8)}"
    
    logging.info(f"✓ All {n_tests} augmentation tests passed! ({masks_detected}/{n_tests} detected masks)")
    
    # Test 3D input (channel-first format)
    test_spec_3d = np.random.randn(1, 128, 256).astype(np.float32)
    aug_spec_3d = augmenter(test_spec_3d)
    assert aug_spec_3d.shape == test_spec_3d.shape, "3D shape mismatch"
    logging.info("✓ 3D input test passed!")
    
    # Test NoAugmentation
    no_aug = NoAugmentation()
    result = no_aug(test_spec)
    assert np.array_equal(result, test_spec), "NoAugmentation modified data!"
    logging.info("✓ NoAugmentation test passed!")
    
    # Test with noise enabled (realistic scenario)
    logging.info("\nTesting with noise enabled (realistic training scenario)...")
    augmenter_with_noise = SpectrogramAugmenter(
        time_mask_ratio=(0.1, 0.15),
        freq_mask_ratio=(0.1, 0.15),
        n_time_masks=2,
        n_freq_masks=2,
        add_noise=True,
        noise_std=0.01,
        apply_probability=1.0
    )
    
    aug_with_noise = augmenter_with_noise(test_spec)
    assert aug_with_noise.shape == test_spec.shape, "Shape changed with noise"
    assert not np.array_equal(aug_with_noise, test_spec), "No augmentation applied"
    logging.info("✓ Augmentation with noise test passed!")
    
    logging.info("\nAll tests passed! ✨")


def calculate_augmentation_statistics(
    spectrogram: np.ndarray,
    augmenter: SpectrogramAugmenter,
    n_samples: int = 100
) -> dict:
    """
    Calculate statistics about augmentation effects.
    
    Args:
        spectrogram: Original spectrogram
        augmenter: Augmentation pipeline
        n_samples: Number of augmented samples to analyze
    
    Returns:
        Dictionary with statistics
    """
    differences = []
    masked_ratios = []
    
    # Temporarily disable noise to accurately measure masking
    original_add_noise = augmenter.add_noise
    augmenter.add_noise = False
    
    for _ in range(n_samples):
        aug_spec = augmenter(spectrogram)
        
        # Calculate difference
        diff = np.mean(np.abs(spectrogram - aug_spec))
        differences.append(diff)
        
        # Calculate masked ratio (use tolerance for float comparison)
        masked_ratio = np.sum(np.isclose(aug_spec, augmenter.mask_value, atol=1e-6)) / aug_spec.size
        masked_ratios.append(masked_ratio)
    
    # Restore original noise setting
    augmenter.add_noise = original_add_noise
    
    stats = {
        'mean_difference': float(np.mean(differences)),
        'std_difference': float(np.std(differences)),
        'mean_masked_ratio': float(np.mean(masked_ratios)),
        'std_masked_ratio': float(np.std(masked_ratios)),
        'min_masked_ratio': float(np.min(masked_ratios)),
        'max_masked_ratio': float(np.max(masked_ratios))
    }
    
    logging.info("Augmentation Statistics (noise disabled for accurate measurement):")
    for key, value in stats.items():
        logging.info(f"  {key}: {value:.4f}")
    
    return stats


if __name__ == '__main__':
    # Run tests
    test_augmentation()
    
    # Demonstrate usage with random data
    logging.info("\n" + "="*70)
    logging.info("DEMONSTRATING AUGMENTATION")
    logging.info("="*70)
    
    demo_spec = np.random.randn(128, 256).astype(np.float32)
    
    augmenter = get_augmentation_pipeline('train')
    augmented = augmenter(demo_spec)
    
    logging.info(f"Original shape: {demo_spec.shape}")
    logging.info(f"Augmented shape: {augmented.shape}")
    logging.info(f"Mean absolute difference: {np.mean(np.abs(demo_spec - augmented)):.4f}")
    
    # Calculate statistics
    stats = calculate_augmentation_statistics(demo_spec, augmenter, n_samples=50)
    
    logging.info("\n" + "="*70)
    logging.info("Augmentation module ready for use! ✨")
    logging.info("="*70)