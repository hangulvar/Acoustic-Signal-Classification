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
    
    # VALIDATION: Check for duplicate IDs in raw metadata
    duplicates_raw = raw_meta[raw_meta.duplicated(subset=['OriginalFileID', 'VesselType'], keep=False)]
    if len(duplicates_raw) > 0:
        logging.warning(f"Found {len(duplicates_raw)} duplicate OriginalFileID+VesselType in raw metadata")
        logging.warning(f"Example duplicates: {duplicates_raw[['OriginalFileID','VesselType']].head(3).values.tolist()}")
        # Remove duplicates, keeping first occurrence
        raw_meta = raw_meta.drop_duplicates(subset=['OriginalFileID', 'VesselType'], keep='first')
        logging.info(f"Removed duplicates, {len(raw_meta)} unique records remaining")
    
    # VALIDATION: Check for duplicate IDs in processed metadata
    duplicates_proc = processed_meta[processed_meta.duplicated(subset=['OriginalFileID', 'VesselType'], keep=False)]
    if len(duplicates_proc) > 0:
        logging.warning(f"Found {len(duplicates_proc)} duplicate OriginalFileID+VesselType in processed metadata")
        logging.warning(f"Example: {duplicates_proc[['OriginalFileID','VesselType']].head(3).values.tolist()}")
    
    # Merge datasets with quality tracking
    merge_cols = ['OriginalFileID','VesselType', grouping_key]
    if 'VesselName' in raw_meta.columns and grouping_key != 'VesselName':
        merge_cols.append('VesselName')
    
    # Use indicator to track merge quality
    merged_df = pd.merge(
        processed_meta,
        raw_meta[merge_cols],
        on=['OriginalFileID','VesselType'],
        how='left',
        indicator=True  # Track which rows merged successfully
    )
    
    # Log merge quality statistics
    merge_stats = merged_df['_merge'].value_counts()
    logging.info(f"Merge statistics:")
    for merge_type, count in merge_stats.items():
        logging.info(f"  {merge_type}: {count}")
    
    # Remove the merge indicator column
    merged_df = merged_df.drop('_merge', axis=1)

    
    # Validate merge
    missing = merged_df[grouping_key].isnull().sum()
    if missing > 0:
        logging.warning(f"{missing} spectrograms could not be mapped to a vessel. Dropping them.")
        merged_df = merged_df.dropna(subset=[grouping_key])
    
    # VALIDATION: Check for null VesselName before creating VesselGroupKey
    if 'VesselName' in merged_df.columns:
        null_vessel_names = merged_df['VesselName'].isnull().sum()
        if null_vessel_names > 0:
            logging.warning(f"{null_vessel_names} records have null VesselName")
            # Fill null VesselName with a placeholder to maintain grouping integrity
            merged_df['VesselName'] = merged_df['VesselName'].fillna('UNKNOWN')
            logging.info(f"Filled null VesselName with 'UNKNOWN' placeholder")
    
    # Add grouping key column for consistency (maintain current format as requested)
    merged_df['VesselGroupKey'] = merged_df[grouping_key].astype(str) + '_' + merged_df['VesselType'].astype(str)+ '_' + merged_df['VesselName'].astype(str)
    
    # VALIDATION: Verify no null values in VesselGroupKey
    null_group_keys = merged_df['VesselGroupKey'].isnull().sum()
    if null_group_keys > 0:
        logging.error(f"{null_group_keys} records have null VesselGroupKey!")
        raise ValueError("VesselGroupKey contains null values - data integrity issue!")
    
    # VALIDATION: Check for unexpected patterns in VesselGroupKey
    has_nan_string = merged_df['VesselGroupKey'].str.contains('nan', case=False, na=False).sum()
    if has_nan_string > 0:
        logging.warning(f"{has_nan_string} VesselGroupKeys contain 'nan' string - check data quality")
    
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
    
    Includes:
    - Pre-flight file existence check
    - Sanity validation of calculated statistics
    """
    logging.info("Calculating normalization statistics from training set...")
    
    # OPTIMIZATION: Pre-flight check - verify files exist before processing
    logging.info("Pre-flight check: Verifying file existence...")
    train_paths = train_df['FeaturePath'].tolist()
    existing_paths = []
    missing_paths = []
    
    for path in train_paths:
        if Path(path).exists():
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    if missing_paths:
        logging.warning(f"Pre-flight check: {len(missing_paths)} files missing")
        logging.warning(f"Will process {len(existing_paths)} available files")
        if len(missing_paths) <= 5:
            for path in missing_paths:
                logging.warning(f"  Missing: {path}")
    else:
        logging.info(f"✓ All {len(existing_paths)} files exist")
    
    if len(existing_paths) == 0:
        raise ValueError("No valid spectrograms found for normalization!")
    
    # Welford's algorithm for stable online variance calculation
    count = 0
    mean = 0.0
    m2 = 0.0
    
    failed_files = []
    
    for path in tqdm(existing_paths, desc="Processing spectrograms"):
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
                
        except Exception as e:
            logging.warning(f"Error processing {path}: {e}")
            failed_files.append(path)
            continue
    
    if failed_files:
        logging.warning(f"Failed to process {len(failed_files)} files. First few: {failed_files[:3]}")
    
    if count == 0:
        raise ValueError("No valid spectrograms processed for normalization!")
    
    global_mean = float(mean)
    global_std = float(np.sqrt(m2 / count))
    
    # VALIDATION: Sanity check on calculated statistics
    if global_std < 1e-6:
        logging.error(f"Suspiciously small std: {global_std}")
        logging.error("This may indicate a data preprocessing issue!")
    
    if abs(global_mean) > 100:
        logging.warning(f"Unusually large mean: {global_mean}")
        logging.warning("Verify spectrogram preprocessing is correct")
    
    logging.info(f"✓ Normalization stats computed:")
    logging.info(f"  Mean: {global_mean:.6f}")
    logging.info(f"  Std:  {global_std:.6f}")
    logging.info(f"  Processed {count:,} values from {len(existing_paths) - len(failed_files)} spectrograms")
    
    # Save statistics
    stats = {
        'mean': global_mean,
        'std': global_std,
        'n_samples': len(existing_paths) - len(failed_files),
        'n_pixels': count,
        'n_missing_files': len(missing_paths),
        'n_failed_files': len(failed_files)
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logging.info(f"✓ Saved to {NORM_STATS_FILE}")
    
    return stats

def verify_splits(train_df, val_df, test_df):
    """
    Enhanced verification of split integrity with multiple data leakage checks.
    
    Checks:
    1. No overlapping vessels across splits
    2. No overlapping spectrograms across splits
    3. No VesselGroupKey integrity issues
    """
    logging.info("Verifying split integrity...")
    
    # Check 1: No overlapping vessels
    train_vessels = set(train_df['VesselGroupKey'])
    val_vessels = set(val_df['VesselGroupKey'])
    test_vessels = set(test_df['VesselGroupKey'])
    
    overlap_train_val = train_vessels & val_vessels
    overlap_train_test = train_vessels & test_vessels
    overlap_val_test = val_vessels & test_vessels
    
    vessel_leakage_detected = False
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logging.error("❌ VESSEL DATA LEAKAGE DETECTED!")
        vessel_leakage_detected = True
        if overlap_train_val:
            logging.error(f"  Train-Val overlap: {len(overlap_train_val)} vessels")
            logging.error(f"  Examples: {list(overlap_train_val)[:3]}")
        if overlap_train_test:
            logging.error(f"  Train-Test overlap: {len(overlap_train_test)} vessels")
            logging.error(f"  Examples: {list(overlap_train_test)[:3]}")
        if overlap_val_test:
            logging.error(f"  Val-Test overlap: {len(overlap_val_test)} vessels")
            logging.error(f"  Examples: {list(overlap_val_test)[:3]}")
    else:
        logging.info("✓ No vessel overlap detected across splits")
    
    # Check 2: No overlapping spectrograms (additional safety check)
    if 'FeaturePath' in train_df.columns:
        train_specs = set(train_df['FeaturePath'])
        val_specs = set(val_df['FeaturePath'])
        test_specs = set(test_df['FeaturePath'])
        
        spec_overlap_train_val = train_specs & val_specs
        spec_overlap_train_test = train_specs & test_specs
        spec_overlap_val_test = val_specs & test_specs
        
        spec_leakage_detected = False
        if spec_overlap_train_val or spec_overlap_train_test or spec_overlap_val_test:
            logging.error("❌ SPECTROGRAM DATA LEAKAGE DETECTED!")
            spec_leakage_detected = True
            if spec_overlap_train_val:
                logging.error(f"  Train-Val spectrogram overlap: {len(spec_overlap_train_val)} files")
            if spec_overlap_train_test:
                logging.error(f"  Train-Test spectrogram overlap: {len(spec_overlap_train_test)} files")
            if spec_overlap_val_test:
                logging.error(f"  Val-Test spectrogram overlap: {len(spec_overlap_val_test)} files")
        else:
            logging.info("✓ No spectrogram overlap detected across splits")
        
        if spec_leakage_detected:
            raise ValueError("Spectrogram data leakage detected in splits!")
    
    # Check 3: Verify VesselGroupKey integrity within each split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        # Check if VesselGroupKey correctly identifies unique vessels
        unique_vessels = split_df['VesselGroupKey'].nunique()
        if 'VesselID' in split_df.columns:
            unique_vessel_ids = split_df.groupby('VesselGroupKey')['VesselID'].nunique()
            if (unique_vessel_ids > 1).any():
                problematic = unique_vessel_ids[unique_vessel_ids > 1]
                logging.warning(f"{split_name}: Some VesselGroupKeys map to multiple VesselIDs:")
                logging.warning(f"  {problematic.to_dict()}")
    
    if vessel_leakage_detected:
        raise ValueError("Vessel data leakage detected in splits!")
    
    logging.info("✓ All data leakage checks passed. Splits are independent.")

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