# src/config.py
"""
Central configuration file for the Acoustic Signal Classification project.
All paths, hyperparameters, and settings are defined here.
"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Project root (assumes script is run from project root)
ROOT_DIR = Path(".")

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "Processed"
PROCESSED_DIR = DATA_DIR / "Processed"
INTERIM_DIR = DATA_DIR / "Interim"

# ============================================================================
# DATA PATHS
# ============================================================================

# Raw metadata (contains VesselID, VesselName, etc.)
RAW_METADATA_PATH = RAW_DIR / "master_metadata.csv"

# Processed metadata (contains FeaturePath, VesselType, OriginalFileID)
PROCESSED_METADATA_PATH = PROCESSED_DIR / "feature_metadata.csv"

# Spectrograms directory
SPECTROGRAMS_DIR = PROCESSED_DIR / "Spectrograms"

# Output directory for processed datasets
OUTPUT_DIR = INTERIM_DIR

# Split metadata files
TRAIN_META_FILE = OUTPUT_DIR / "train_meta.csv"
VAL_META_FILE = OUTPUT_DIR / "val_meta.csv"
TEST_META_FILE = OUTPUT_DIR / "test_meta.csv"

# Normalization statistics
NORM_STATS_FILE = OUTPUT_DIR / "norm_stats.json"

# ============================================================================
# DATA SPLITTING PARAMETERS
# ============================================================================

# Train/Val/Test split ratios (vessel-level)
TEST_SIZE = 0.15        # 15% of vessels for testing
VALIDATION_SIZE = 0.15  # 15% of vessels for validation
# Training will be 70% (remaining)

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

# Spectrogram parameters (for reference - from your earlier preprocessing)
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 11000

# ============================================================================
# AUGMENTATION PARAMETERS
# ============================================================================

# SpecAugment - Time Masking
TIME_MASK_RATIO_MIN = 0.05   # Minimum 5% of time frames
TIME_MASK_RATIO_MAX = 0.15   # Maximum 15% of time frames
N_TIME_MASKS = 2             # Number of time masks per spectrogram

# SpecAugment - Frequency Masking
FREQ_MASK_RATIO_MIN = 0.05   # Minimum 5% of frequency bins
FREQ_MASK_RATIO_MAX = 0.15   # Maximum 15% of frequency bins
N_FREQ_MASKS = 2             # Number of frequency masks per spectrogram

# Time shifting
TIME_SHIFT_RANGE = 10        # Max frames to shift (±)

# Gaussian noise
ADD_NOISE = True
NOISE_STD = 0.01             # Standard deviation of noise

# Augmentation probability
AUGMENTATION_PROBABILITY = 0.8  # Apply augmentation 80% of the time

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# CNN architecture
N_CLASSES = 4                # Cargo, Passenger, Tanker, Tug
INPUT_CHANNELS = 1           # Single channel (log-mel spectrogram)
DROPOUT_RATE = 0.5           # Dropout for regularization

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Optimization
BATCH_SIZE = 8            # Batch size, CHANGE IF MEMORY ISSUES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization

# Training schedule
N_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 5    # Reduce LR if no improvement for 5 epochs
LR_SCHEDULER_FACTOR = 0.5    # Multiply LR by this factor

# Class weighting
USE_CLASS_WEIGHTS = True     # Use inverse frequency weighting for imbalanced data

# ============================================================================
# DATALOADER PARAMETERS
# ============================================================================

NUM_WORKERS = 4              # Number of worker processes for data loading
PIN_MEMORY = True            # Pin memory for faster GPU transfer (set False for CPU)
SHUFFLE_TRAIN = True         # Shuffle training data
DROP_LAST_TRAIN = True       # Drop incomplete batch in training

# ============================================================================
# CHECKPOINTING AND LOGGING
# ============================================================================

# Model checkpoints
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAVE_CHECKPOINT_EVERY = 10   # Save checkpoint every N epochs

# TensorBoard logs
TENSORBOARD_DIR = OUTPUT_DIR / "runs"

# Results
RESULTS_DIR = OUTPUT_DIR / "results"
TEST_RESULTS_FILE = RESULTS_DIR / "test_results.json"
TRAINING_HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure settings
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Will be determined at runtime, but can be overridden
# Options: 'cuda', 'cpu', 'mps' (for Apple Silicon)
DEVICE = None  # Set to None for auto-detection

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration and check if required files exist."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("CONFIGURATION VALIDATION")
    print("="*60)
    
    # Check if critical files exist
    checks = {
        'Raw Metadata': RAW_METADATA_PATH,
        'Processed Feature Metadata': PROCESSED_METADATA_PATH,
        'Spectrograms Directory': SPECTROGRAMS_DIR
    }
    
    all_exist = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    print("\nSplit ratios:")
    print(f"  Train: {1 - TEST_SIZE - VALIDATION_SIZE:.1%}")
    print(f"  Val:   {VALIDATION_SIZE:.1%}")
    print(f"  Test:  {TEST_SIZE:.1%}")
    
    print("\nAugmentation settings:")
    print(f"  Time masks: {N_TIME_MASKS} × {TIME_MASK_RATIO_MIN:.1%}-{TIME_MASK_RATIO_MAX:.1%}")
    print(f"  Freq masks: {N_FREQ_MASKS} × {FREQ_MASK_RATIO_MIN:.1%}-{FREQ_MASK_RATIO_MAX:.1%}")
    print(f"  Probability: {AUGMENTATION_PROBABILITY:.1%}")
    
    print("\nTraining settings:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    print("="*60)
    
    if not all_exist:
        print("\n⚠ WARNING: Some required files are missing!")
        print("Please ensure data preprocessing is complete before training.")
    else:
        print("\n✓ All configuration checks passed!")
    
    return all_exist


if __name__ == '__main__':
    validate_config()