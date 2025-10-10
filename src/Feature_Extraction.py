##################################################################
##### Feature Extraction Module ##################################   
##################################################################
# The script preprocesses audio data for the DeepShip project.
# It normalizes audio files before performing feature extraction.
# It saves the processed features and metadata for model training.
##################################################################
# ---------------------------------------------------------------
# Key functionalities include:
# 1. Resampling audio to a consistent sample rate. (22050 Hz)
# 2. Audio normalization to ensure consistent volume levels.
# 3. Fixed length chunking to standardize input sizes. (5 seconds)
# 4. Feature extraction using log-mel spectrograms.
# 5. Saving processed features and metadata for QC & model training.
# ---------------------------------------------------------------
##################################################################
##################################################################

# Debugging: Uncomment to check current working directory
#import os
#print(os.getcwd())  # Print current working directory for debugging

# --- 0. Imports ---
import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm # For a progress bar
import matplotlib.pyplot as plt # For visualization

# --- 1. Configuration ---
METADATA_FILE = 'data/Processed/master_metadata.csv'
# The output directory for our final features (spectrograms)
OUTPUT_FEATURES_DIR = 'data/Processed/Spectrograms'
# The final metadata file mapping spectrograms to labels
OUTPUT_METADATA_FILE = 'data/Processed/feature_metadata.csv'

# Audio Processing Hyperparameters
TARGET_SR = 22050
TARGET_RMS = 0.1
CHUNK_SEC = 5
OVERLAP_SEC = 2.5

# Spectrogram Hyperparameters (based on standard practices)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000

# --- 2. Create Output Directory ---
os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

##################################################################
##################################################################
# --- 3. Core Processing Functions ---

def normalize_rms(audio, target_rms=0.1):
    """Normalizes the audio to a target RMS value."""
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms > 1e-5: # Avoid division by zero
        audio = audio * (target_rms / current_rms)
    return audio

def chunk_audio(audio, sr, chunk_sec, overlap_sec):
    """Pads audio if too short, then creates overlapping chunks."""
    chunk_samples = int(chunk_sec * sr)
    
    # --- FIX: Pad audio if it's shorter than a chunk ---
    if len(audio) < chunk_samples:
        padding_needed = chunk_samples - len(audio)
        audio = np.pad(audio, (0, padding_needed), 'constant')

    step_samples = int((chunk_sec - overlap_sec) * sr)
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, step_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
    return chunks

# --- 4. Feature extraction Log-Mel Spectrograms ---

def extract_spectrogram(audio_chunk, sr, return_shape=False):
    """
    Converts an audio chunk into a log-mel spectrogram.
    
    This function performs:
    1. Short-Time Fourier Transform (STFT)
    2. Power spectrum computation
    3. Mel filterbank application
    4. Logarithmic scaling to decibels
    
    Args:
        audio_chunk (np.ndarray): Audio time series, shape (n_samples,)
        sr (int): Sample rate in Hz
        return_shape (bool): If True, also return spectrogram shape info
        
    Returns:
        np.ndarray: Log-mel spectrogram in dB, shape (n_mels, n_frames)
        tuple (optional): (n_mels, n_frames) if return_shape=True
    """
    # Validate input
    if len(audio_chunk) == 0:
        raise ValueError("Empty audio chunk provided")
    
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")
    
    # Step 1-3: Compute mel spectrogram
    # (internally: STFT → Power spectrum → Mel filterbank)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0  # Explicitly use power spectrum (|STFT|²)
    )
    
    # Step 4: Convert to log scale with FIXED reference
    # ref=1.0 maintains absolute power levels across samples
    log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, top_db=None)
    
    # Optional: Return shape information for debugging
    if return_shape:
        return log_mel_spec, log_mel_spec.shape
    
    return log_mel_spec

##################################################################
##################################################################
# --- 5. Main Processing Pipeline ---

# Load initial metadata
metadata = pd.read_csv(METADATA_FILE)
processed_data = []

# Use tqdm for a live progress bar
for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing Audio Files"):
    file_path = row['FilePath']
    vessel_type = row['VesselType']
    # Use the original file name's stem as a robust unique ID
    original_file_id = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Load and resample in one step
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Normalize audio energy
        audio = normalize_rms(audio, target_rms=TARGET_RMS)
        
        # Create audio chunks
        audio_chunks = chunk_audio(audio, sr, CHUNK_SEC, OVERLAP_SEC)
        
        # Process each chunk into a spectrogram and save it
        for i, chunk in enumerate(audio_chunks):
            spectrogram = extract_spectrogram(chunk, sr)
            
            # --- FIX: Robust file naming and save final feature ---
            feature_filename = f'{original_file_id}_{vessel_type}_chunk{i}.npy'
            feature_path = os.path.join(OUTPUT_FEATURES_DIR, feature_filename)
            np.save(feature_path, spectrogram)
            
            processed_data.append({
                'FeaturePath': feature_path,
                'VesselType': vessel_type,
                'OriginalFileID': original_file_id
            })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# --- 6. Save Final Metadata ---
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv(OUTPUT_METADATA_FILE, index=False)

print(f"\n✅ Processing complete!")
print(f"Saved {len(processed_df)} spectrogram features to '{OUTPUT_FEATURES_DIR}'")
print(f"Final metadata saved to '{OUTPUT_METADATA_FILE}'")

##################################################################