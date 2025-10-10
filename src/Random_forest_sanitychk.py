### Random Forest Baseline Sanity Check
# This script performs a quick sanity check on the extracted features by training a simple Random Forest classifier.
# The goal is to ensure that the features contain a meaningful signal that can be used for classification
#################################################################################################################


# First, ensure the tqdm library for progress bars is installed
#!pip install tqdm

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- 1. Configuration ---
# The final metadata file mapping spectrograms to labels
METADATA_FILE = 'data/Processed/feature_metadata.csv'

# To make the sanity check run quickly, we'll use a subset of the data.
# Set to None to use the full dataset.
DATA_SUBSET_SIZE = 5000

# --- 2. Load and Prepare Data ---
print(f"--- Baseline Sanity Check Initialized ---")

try:
    metadata_df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"❌ Error: Metadata file not found at '{METADATA_FILE}'")
    exit()

if DATA_SUBSET_SIZE:
    print(f"Using a random subset of {DATA_SUBSET_SIZE} samples for the check.")
    if len(metadata_df) > DATA_SUBSET_SIZE:
        metadata_df = metadata_df.sample(n=DATA_SUBSET_SIZE, random_state=42)
    else:
        print("Subset size is larger than the dataset. Using the full dataset.")

# Lists to hold our data
X = []
y = []

print("Loading and flattening spectrograms...")
for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    try:
        # Load the 2D spectrogram
        spectrogram = np.load(row['FeaturePath'])

        # Flatten the 2D array into a 1D vector and append
        X.append(spectrogram.flatten())
        y.append(row['VesselType'])
    except FileNotFoundError:
        print(f"Warning: File not found for row {idx}. Skipping.")
        continue

if not X:
    print("❌ Error: No data was loaded. Please check file paths in metadata.")
    exit()

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# --- 3. Encode Labels ---
print("\nEncoding labels...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
# Store class names for the final report
class_names = encoder.classes_
print(f"Labels mapped to classes: {list(zip(class_names, range(len(class_names))))}")

# --- 4. Split Data ---
print("Splitting data into training and testing sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # Ensures proportional class representation
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 5. Train and Evaluate Random Forest Model ---
print("\nTraining Random Forest classifier...")
# n_jobs=-1 uses all available CPU cores for faster training
model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model performance...")
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*40)
print(f"✅ BASELINE SANITY CHECK COMPLETE ✅")
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
print("="*40)

# Compare to baseline
if accuracy > 0.40:
    print("\nConclusion: Accuracy is well above the 40% target.")
    print("This provides strong evidence that your features contain a useful signal! 🚀")
elif accuracy > 0.25:
    print("\nConclusion: Accuracy is better than random guessing (25%).")
    print("This indicates your features contain some signal, but it might be weak.")
else:
    print("\nConclusion: Accuracy is at or below random guessing.")
    print("This suggests a potential issue with the feature extraction process.")

# Display detailed report
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=class_names))