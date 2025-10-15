# src/prepare_dataset.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import logging
from pathlib import Path

from dir_train_config import (
    RAW_METADATA_PATH, PROCESSED_METADATA_PATH, OUTPUT_DIR,
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, NORM_STATS_FILE,
    TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_prepare_metadata():
    """
    Loads and merges raw and processed metadata.
    Handles both VesselID and VesselName as grouping keys.
    """
    logging.info("Loading metadata files...")
    
    try:
        raw_meta = pd.read_csv(RAW_METADATA_PATH)
        processed_meta = pd.read_csv(PROCESSED_METADATA_PATH)
    except FileNotFoundError as e:
        logging.error(f"Metadata file not found: {e}")
        raise
    
    # Normalize file paths
    processed_meta['FeaturePath'] = processed_meta['FeaturePath'].str.replace('\\', '/')
    
    # Determine grouping key (VesselID or VesselName)
    if 'VesselID' in raw_meta.columns:
        grouping_key = 'VesselID'
        logging.info("Using 'VesselID' as vessel identifier")
    elif 'VesselName' in raw_meta.columns:
        grouping_key = 'VesselName'
        logging.info("Using 'VesselName' as vessel identifier (VesselID not found)")
    else:
        raise ValueError("Neither 'VesselID' nor 'VesselName' found in raw metadata!")
    
    # Prepare raw metadata for merging
    if 'ID' in raw_meta.columns:
        raw_meta.rename(columns={'ID': 'OriginalFileID'}, inplace=True)
    
    # Merge datasets
    merge_cols = ['OriginalFileID','VesselType', grouping_key]
    if 'VesselName' in raw_meta.columns and grouping_key != 'VesselName':
        merge_cols.append('VesselName')
    
    merged_df = pd.merge(
        processed_meta,
        raw_meta[merge_cols],
        on=['OriginalFileID','VesselType'], # this is not a reliable key, we need to merge on fileID and vesseltype
        how='left'
    )
    
    # Validate merge
    missing = merged_df[grouping_key].isnull().sum()
    if missing > 0:
        logging.warning(f"{missing} spectrograms could not be mapped to a vessel. Dropping them.")
        merged_df = merged_df.dropna(subset=[grouping_key])
    
    # Add grouping key column for consistency
    merged_df['VesselGroupKey'] = merged_df[grouping_key].astype(str) + '_' + merged_df['VesselType'].astype(str)+ '_' + merged_df['VesselName'].astype(str)
    
    # Verify file existence (sample check)
    sample_size = min(10, len(merged_df))
    sample_paths = merged_df['FeaturePath'].sample(sample_size, random_state=RANDOM_STATE)
    missing_files = [p for p in sample_paths if not Path(p).exists()]
    if missing_files:
        logging.warning(f"Sample check found {len(missing_files)} missing files. "
                       f"Example: {missing_files[0]}")
    
    logging.info(f"✓ Merged metadata: {len(merged_df)} spectrogram chunks from "
                 f"{merged_df['VesselGroupKey'].nunique()} unique vessels") # make this more robust, to include VesselID and VesselType combinations to show the correct number of unique vessels
    
    # Log class distribution
    class_dist = merged_df['VesselType'].value_counts()
    logging.info(f"Class distribution:\n{class_dist}")
    
    return merged_df

def perform_vessel_aware_split(df):
    """
    Performs stratified, vessel-level splitting to prevent data leakage.
    All chunks from the same vessel go into the same split.
    """
    logging.info("Performing vessel-aware split...")
    
    # Get unique vessels with their class labels
    vessel_info = df[['VesselGroupKey', 'VesselType']].drop_duplicates().reset_index(drop=True)
    
    logging.info(f"Total unique vessels: {len(vessel_info)}")
    logging.info(f"Vessel class distribution:\n{vessel_info['VesselType'].value_counts()}")
    
    # Check if we have enough vessels per class for stratification
    min_class_count = vessel_info['VesselType'].value_counts().min()
    if min_class_count < 3:
        logging.warning(f"Minimum vessels per class is {min_class_count}. "
                       f"Stratification may fail. Consider non-stratified split.")
        stratify_col = None
    else:
        stratify_col = vessel_info['VesselType']
    
    try:
        # Split vessels into train+val and test
        train_val_vessels, test_vessels = train_test_split(
            vessel_info,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_col
        )
        
        # Calculate validation size relative to remaining data
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        
        # Split train+val into train and validation
        train_vessels, val_vessels = train_test_split(
            train_val_vessels,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=train_val_vessels['VesselType'] if stratify_col is not None else None
        )
    except ValueError as e:
        logging.error(f"Stratified split failed: {e}")
        logging.info("Falling back to non-stratified split...")
        
        train_val_vessels, test_vessels = train_test_split(
            vessel_info,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        train_vessels, val_vessels = train_test_split(
            train_val_vessels,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE
        )
    
    # Create dataframe splits based on vessel assignments
    train_df = df[df['VesselGroupKey'].isin(train_vessels['VesselGroupKey'])].copy()
    val_df = df[df['VesselGroupKey'].isin(val_vessels['VesselGroupKey'])].copy()
    test_df = df[df['VesselGroupKey'].isin(test_vessels['VesselGroupKey'])].copy()
    
    # Log detailed split statistics
    logging.info("\n" + "="*60)
    logging.info("SPLIT SUMMARY")
    logging.info("="*60)
    
    for split_name, split_df, vessel_df in [
        ('TRAIN', train_df, train_vessels),
        ('VALIDATION', val_df, val_vessels),
        ('TEST', test_df, test_vessels)
    ]:
        logging.info(f"\n{split_name}:")
        logging.info(f"  Vessels: {len(vessel_df)} ({len(vessel_df)/len(vessel_info)*100:.1f}%)")
        logging.info(f"  Chunks: {len(split_df)} ({len(split_df)/len(df)*100:.1f}%)")
        logging.info(f"  Class distribution:")
        for cls, count in split_df['VesselType'].value_counts().items():
            logging.info(f"    {cls}: {count} ({count/len(split_df)*100:.1f}%)")
    
    logging.info("="*60 + "\n")
    
    # Verify no vessel appears in multiple splits
    all_vessels = set(train_vessels['VesselGroupKey']) | \
                  set(val_vessels['VesselGroupKey']) | \
                  set(test_vessels['VesselGroupKey'])
    assert len(all_vessels) == len(vessel_info), "Vessel split integrity check failed!"
    
    return train_df, val_df, test_df

def calculate_norm_stats(train_df):
    """
    Calculates global mean and std from training spectrograms.
    Uses Welford's online algorithm for numerical stability and memory efficiency.
    """
    logging.info("Calculating normalization statistics from training set...")
    
    # Welford's algorithm for stable online variance calculation
    count = 0
    mean = 0.0
    m2 = 0.0
    
    failed_files = []
    
    for path in tqdm(train_df['FeaturePath'], desc="Processing spectrograms"):
        try:
            spectrogram = np.load(path)
            
            # Flatten and convert to float64 for precision
            spec_flat = spectrogram.flatten().astype(np.float64)
            
            for value in spec_flat:
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                m2 += delta * delta2
                
        except FileNotFoundError:
            failed_files.append(path)
            continue
        except Exception as e:
            logging.warning(f"Error processing {path}: {e}")
            failed_files.append(path)
            continue
    
    if failed_files:
        logging.warning(f"Failed to process {len(failed_files)} files. First few: {failed_files[:3]}")
    
    if count == 0:
        raise ValueError("No valid spectrograms found for normalization!")
    
    global_mean = float(mean)
    global_std = float(np.sqrt(m2 / count))
    
    logging.info(f"✓ Normalization stats computed:")
    logging.info(f"  Mean: {global_mean:.6f}")
    logging.info(f"  Std:  {global_std:.6f}")
    logging.info(f"  Processed {count:,} values from {len(train_df) - len(failed_files)} spectrograms")
    
    # Save statistics
    stats = {
        'mean': global_mean,
        'std': global_std,
        'n_samples': len(train_df),
        'n_pixels': count
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logging.info(f"✓ Saved to {NORM_STATS_FILE}")
    
    return stats

def verify_splits(train_df, val_df, test_df):
    """Additional verification of split integrity."""
    logging.info("Verifying split integrity...")
    
    # Check no overlapping vessels
    train_vessels = set(train_df['VesselGroupKey'])
    val_vessels = set(val_df['VesselGroupKey'])
    test_vessels = set(test_df['VesselGroupKey'])
    
    overlap_train_val = train_vessels & val_vessels
    overlap_train_test = train_vessels & test_vessels
    overlap_val_test = val_vessels & test_vessels
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logging.error("❌ DATA LEAKAGE DETECTED!")
        if overlap_train_val:
            logging.error(f"  Train-Val overlap: {len(overlap_train_val)} vessels")
        if overlap_train_test:
            logging.error(f"  Train-Test overlap: {len(overlap_train_test)} vessels")
        if overlap_val_test:
            logging.error(f"  Val-Test overlap: {len(overlap_val_test)} vessels")
        raise ValueError("Data leakage detected in splits!")
    
    logging.info("✓ No data leakage detected. All splits are independent.")

def main():
    """Main pipeline for dataset preparation."""
    logging.info("Starting dataset preparation pipeline...")
    logging.info(f"Configuration: Test={TEST_SIZE*100}%, Val={VALIDATION_SIZE*100}%, "
                 f"Random State={RANDOM_STATE}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and prepare metadata
    master_df = load_and_prepare_metadata()
    
    # Step 2: Perform vessel-aware split
    train_df, val_df, test_df = perform_vessel_aware_split(master_df)
    
    # Step 3: Verify split integrity
    verify_splits(train_df, val_df, test_df)
    
    # Step 4: Save metadata splits
    train_df.to_csv(TRAIN_META_FILE, index=False)
    val_df.to_csv(VAL_META_FILE, index=False)
    test_df.to_csv(TEST_META_FILE, index=False)
    logging.info(f"✓ Metadata splits saved to {OUTPUT_DIR}")
    
    # Step 5: Calculate and save normalization statistics
    norm_stats = calculate_norm_stats(train_df)
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("DATASET PREPARATION COMPLETE! ✨")
    logging.info("="*60)
    logging.info(f"Output files:")
    logging.info(f"  • {TRAIN_META_FILE}")
    logging.info(f"  • {VAL_META_FILE}")
    logging.info(f"  • {TEST_META_FILE}")
    logging.info(f"  • {NORM_STATS_FILE}")
    logging.info("="*60)
    
    return train_df, val_df, test_df, norm_stats

if __name__ == '__main__':
    main()