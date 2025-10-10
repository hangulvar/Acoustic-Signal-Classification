##################################################################
##################################################################
# Quality Check: Refactor into a separate script or notebook as needed
# This section performs basic validation of the extracted features.
##################################################################
# Key checks include:
# - Validate shapes of a sample spectrogram
# - Value range checks can be added as needed
# - Visualize a sample spectrogram for manual verification
##################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import numpy as np
import warnings
import os

# --- 1. Configuration ---
# All your hyperparameters are correct and loaded from the previous cell.
TARGET_SR = 22050
CHUNK_SEC = 5
HOP_LENGTH = 512
N_MELS = 128

FMIN = 20
FMAX = 8000

# The final metadata file mapping spectrograms to labels
METADATA_FILE = 'data/Processed/feature_metadata.csv'

# Suppress common UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 2. Corrected Shape Validation Script ---

try:
    metadata_df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"❌ Error: Metadata file not found at '{METADATA_FILE}'")
    exit()
    

# FIX: Calculate the expected number of time frames based on constants.
# This formula calculates how many times the analysis window (hop_length) fits into the audio chunk.
# Librosa's padding and framing means the result is often N+1, so we check a small range.
expected_frames = int((CHUNK_SEC * TARGET_SR) / HOP_LENGTH) + 1

print(f"--- Starting Shape Validation ---")
print(f"Expected Spectrogram Shape: ({N_MELS}, ~{expected_frames})")
print("-" * 35)

vessel_types = metadata_df["VesselType"].unique()
SAMPLES_PER_CLASS = 3
total_checks_passed = 0

for vessel in vessel_types:
    print(f"\nVerifying class: [{vessel}]")
    
    # Get random samples for the current class
    samples = metadata_df[metadata_df["VesselType"] == vessel].sample(n=SAMPLES_PER_CLASS, replace=False)
    
    for i, row in samples.iterrows():
        feature_path = row['FeaturePath']
        
        try:
            # FIX: Load the data into a correctly named variable
            spectrogram = np.load(feature_path)
            actual_shape = spectrogram.shape
            
            # Perform the assertions
            assert actual_shape[0] == N_MELS, f"Wrong n_mels: {actual_shape[0]} != {N_MELS}"
            assert abs(actual_shape[1] - expected_frames) <= 2, f"Wrong n_frames: {actual_shape[1]} vs expected {expected_frames}"

            print(f"  ✅ PASSED: Sample {i+1} ({os.path.basename(feature_path)}) has correct shape {actual_shape}")
            total_checks_passed += 1

        except FileNotFoundError:
            print(f"  ❌ FAILED: File not found at '{feature_path}'")
        except AssertionError as e:
            print(f"  ❌ FAILED: Sample {i+1} ({os.path.basename(feature_path)}) has WRONG shape {actual_shape}. Reason: {e}")
        except Exception as e:
            print(f"  ❌ FAILED: An unexpected error occurred with {feature_path}: {e}")

print(f"\n--- Validation Complete ---")
print(f"✅ Passed {total_checks_passed} out of {len(vessel_types) * SAMPLES_PER_CLASS} checks.")


# --- 3. Visualize a sample spectrogram ---

# Extract and visualize
#spec,_ = extract_spectrogram(audio, sr)

# For demonstration, load a random sample from the metadata
for vessel in vessel_types:
    print(f"\nVerifying class: [{vessel}]")
    
    # Get random samples for the current class
    samples = metadata_df[metadata_df["VesselType"] == vessel].sample(n=SAMPLES_PER_CLASS, replace=False)

    for i, row in samples.iterrows():
        feature_path = row['FeaturePath']
        
        try:
            # FIX: Load the data into a correctly named variable
            spec = np.load(feature_path)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                spec,
                sr=TARGET_SR,
                hop_length=HOP_LENGTH,
                x_axis='time',
                y_axis='mel',
                fmin=FMIN,
                fmax=FMAX,
                cmap='viridis'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Log-Mel Spectrogram {vessel}')
            plt.tight_layout()
            plt.savefig(f'Reports/test_spectrogram_{vessel}_{i}.png', dpi=150)
            plt.show() # Display inline if using a notebook
            print("✅ Visualization saved!")
        except FileNotFoundError:
            print(f"  ❌ FAILED: File not found at '{feature_path}'")
        except Exception as e:
            print(f"  ❌ FAILED: An unexpected error occurred with {feature_path}: {e}")

# What to look for:
# --------------------------------------------------
# Clear structure (not random noise)
# Horizontal lines (sustained frequencies)
# Color variation (bright = loud, dark = quiet)
# No completely black or white regions
# --------------------------------------------------

