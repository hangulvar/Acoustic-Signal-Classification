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

# Spectrogram parameters (for reference)
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
TIME_MASK_RATIO_MIN = 0.05
TIME_MASK_RATIO_MAX = 0.15   # Reduced slightly to be less aggressive
N_TIME_MASKS = 3

# SpecAugment - Frequency Masking
FREQ_MASK_RATIO_MIN = 0.05
FREQ_MASK_RATIO_MAX = 0.15   # Reduced slightly to be less aggressive
N_FREQ_MASKS = 3

# Time shifting
TIME_SHIFT_RANGE = 10

# Gaussian noise
ADD_NOISE = True
NOISE_STD = 0.02

# Augmentation probability
AUGMENTATION_PROBABILITY = 0.8

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

N_CLASSES = 4
INPUT_CHANNELS = 1
# Reduced dropout slightly. Goal is to first overfit, then regularize.
DROPOUT_RATE = 0.4

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Optimization
BATCH_SIZE = 32              # Switched to a power of 2, common practice
# Increased LR significantly. 4e-5 is too low for training from scratch.
# 3e-4 is a standard, robust starting point for Adam.
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4          # Slightly reduced L2 regularization

# Training schedule
N_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 15

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Class weighting
USE_CLASS_WEIGHTS = True

# ============================================================================
# DATALOADER PARAMETERS
# ============================================================================

NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True
DROP_LAST_TRAIN = True

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

DEVICE = None  # Auto-detection in training script

# ============================================================================
# VALIDATION (No changes needed here)
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
    print(f"  Dropout rate: {DROPOUT_RATE}")

    print("="*60)

    if not all_exist:
        print("\n⚠ WARNING: Some required files are missing!")
        print("Please ensure data preprocessing is complete before training.")
    else:
        print("\n✓ All configuration checks passed!")

    return all_exist


if __name__ == '__main__':
    validate_config()