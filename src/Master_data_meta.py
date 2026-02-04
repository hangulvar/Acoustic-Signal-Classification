#################################
# Preprocessing Module - DeepShip 
#--------------------------------
#################################
# 1. Maps audio files to metadata
# 2. Validates file existence
#################################
#################################
import pandas as pd
from pathlib import Path
import warnings
import logging
from typing import Optional

# Import centralized configuration
try:
    from dir_train_config import RAW_DIR, PROCESSED_DIR
    USE_CENTRALIZED_CONFIG = True
except ImportError:
    # Fallback to default paths if config not available
    USE_CENTRALIZED_CONFIG = False
    logging.warning("Could not import centralized config, using default paths")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


############################################
# this function appends metadata from all classes
# and validates file existence

def create_master_metadata(dataset_path: str, output_csv: str, validate: bool = True):
    """
    Scans dataset directory, matches actual audio files with metadata,
    and creates a validated master CSV file.
    
    Args:
        dataset_path (str): Root path of the dataset
        output_csv (str): Path to save the final master CSV
        validate (bool): If True, only include files that exist
    """
    dataset_root = Path(dataset_path)
    all_metadata = []
    
    vessel_types = ['Cargo', 'Passenger', 'Tanker', 'Tug']
    
    logging.info("="*60)
    logging.info("Starting metadata aggregation with file validation...")
    logging.info("="*60)
    
    total_audio_files = 0
    total_matched = 0
    total_unmatched_audio = 0
    total_unmatched_meta = 0
    
    for vessel in vessel_types:
        class_dir = dataset_root / vessel
        
        if not class_dir.is_dir():
            logging.warning(f"Directory not found for class: {vessel}")
            continue
        
        # STEP 1: Find all actual audio files
        audio_files = list(class_dir.glob('*.[wW][aA][vV]'))
        total_audio_files += len(audio_files)
        
        logging.info(f"\nProcessing {vessel}:")
        logging.info(f"  Found {len(audio_files)} audio files")
        
        # Create dictionary: {file_id: file_path}
        audio_dict = {f.stem: f for f in audio_files}
        
        # STEP 2: Find and load metafile
        try:
            metafile_path = next(class_dir.glob('*metafile*'))
            logging.info(f"  Found metafile: {metafile_path.name}")
        except StopIteration:
            logging.warning(f"  No metafile found for {vessel}, using only audio files")
            df = pd.DataFrame({
                'ID': list(audio_dict.keys()),
                'FilePath': [str(p) for p in audio_dict.values()],
                'VesselType': vessel
            })
            all_metadata.append(df)
            continue
        
        # STEP 3: Load metafile with proper column handling
        # Define 8 columns (7 data + 1 empty from trailing comma)
        col_names = ['ID', 'VesselID', 'VesselName', 'Date', 'Time', 
                     'Duration', 'ClipRange', 'EmptyCol']
        
        try:
            df_meta = pd.read_csv(
                metafile_path,
                header=None,
                names=col_names,
                on_bad_lines='skip',
                dtype=str,
                skipinitialspace=True  # Strip leading whitespace
            )
            
            # Drop the empty column created by trailing comma
            df_meta.drop('EmptyCol', axis=1, inplace=True)
            
            # Clean ID column (remove whitespace, ensure string)
            df_meta['ID'] = df_meta['ID'].astype(str).str.strip()
            
            logging.info(f"  Loaded {len(df_meta)} metadata rows")
            
        except Exception as e:
            logging.error(f"  Error reading metafile: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            continue
        
        # STEP 4: LEFT JOIN - Match metadata with existing audio files
        if validate:
            # Filter metadata to only include IDs that have audio files
            df_meta['FileExists'] = df_meta['ID'].isin(audio_dict.keys())
            
            matched = df_meta['FileExists'].sum()
            unmatched_meta = len(df_meta) - matched
            unmatched_audio = len(audio_dict) - matched
            
            total_matched += matched
            total_unmatched_meta += unmatched_meta
            total_unmatched_audio += unmatched_audio
            
            logging.info(f"  ✓ Matched: {matched}")
            if unmatched_meta > 0:
                logging.warning(f"  Metadata without audio: {unmatched_meta}")
                # Show which IDs are missing
                missing_ids = df_meta[~df_meta['FileExists']]['ID'].tolist()[:5]
                logging.warning(f"    Example missing IDs: {missing_ids}")
            
            if unmatched_audio > 0:
                logging.warning(f"  Audio without metadata: {unmatched_audio}")
                # Show which audio files have no metadata
                missing_audio = [k for k in audio_dict.keys() if k not in df_meta['ID'].values][:5]
                logging.warning(f"    Example orphaned audio: {missing_audio}")
            
            # Keep only matched records
            df_meta = df_meta[df_meta['FileExists']].copy()
            df_meta.drop('FileExists', axis=1, inplace=True)
            
            # Add file paths from actual files
            df_meta['FilePath'] = df_meta['ID'].map(
                lambda x: str(audio_dict.get(x, ''))
            )
        else:
            # Don't validate - trust metafile
            df_meta['FilePath'] = df_meta['ID'].apply(
                lambda x: str(class_dir / f"{x}.wav")
            )
        
        # Add vessel type label
        df_meta['VesselType'] = vessel
        
        # Verify FilePath is not empty
        if df_meta['FilePath'].str.len().min() == 0:
            logging.warning(f"  Warning: Some FilePaths are empty for {vessel}!")
        
        all_metadata.append(df_meta)
    
    # STEP 5: Combine all dataframes
    if not all_metadata:
        logging.error("No metadata found! Check your dataset path.")
        return None
    
    master_df = pd.concat(all_metadata, ignore_index=True)
    
    # Reorder columns for clarity
    columns_order = ['FilePath', 'VesselType', 'ID', 'VesselID', 'VesselName', 
                     'Date', 'Time', 'Duration', 'ClipRange']
    
    # Only include columns that exist
    columns_order = [col for col in columns_order if col in master_df.columns]
    master_df = master_df[columns_order]
    
    # Remove any rows with empty FilePath
    initial_count = len(master_df)
    master_df = master_df[master_df['FilePath'] != ''].reset_index(drop=True)
    removed = initial_count - len(master_df)
    if removed > 0:
        logging.warning(f"Removed {removed} rows with invalid FilePaths")
    
    # Save to CSV
    master_df.to_csv(output_csv, index=False)
    
    # STEP 6: Print comprehensive summary
    logging.info("\n" + "="*60)
    logging.info("📊 SUMMARY")
    logging.info("="*60)
    logging.info(f"Total audio files found:      {total_audio_files}")
    logging.info(f"Total matched records:        {total_matched}")
    logging.info(f"Metadata without audio:       {total_unmatched_meta}")
    logging.info(f"Audio without metadata:       {total_unmatched_audio}")
    logging.info(f"\n✓ Master metadata saved to: {output_csv}")
    logging.info(f"Final dataset size:           {len(master_df)} samples")
    
    # Class distribution
    logging.info("\n📈 Class Distribution:")
    class_counts = master_df['VesselType'].value_counts()
    for vessel, count in class_counts.items():
        percentage = (count / len(master_df)) * 100
        logging.info(f"   {vessel:12s}: {count:4d} ({percentage:5.2f}%)")
    
    # Sample data
    logging.info("\n🔍 Sample Records (first 3):")
    logging.info("-" * 60)
    sample_cols = ['ID', 'VesselType', 'VesselName', 'Duration']
    if all(col in master_df.columns for col in sample_cols):
        logging.info("\n" + master_df[sample_cols].head(3).to_string(index=False))
    else:
        logging.info("\n" + master_df.head(3).to_string(index=False))
    
    # Data quality checks
    logging.info("\n🔍 Data Quality Checks:")
    
    # Check for duplicate IDs
    duplicate_ids = master_df[master_df.duplicated(subset=['ID', 'VesselType'], keep=False)]
    if len(duplicate_ids) > 0:
        logging.warning(f"   Found {len(duplicate_ids)} duplicate ID+VesselType combinations!")
        logging.warning(f"   Example duplicates: {duplicate_ids[['ID', 'VesselType']].head(3).values.tolist()}")
    else:
        logging.info("   ✓ No duplicate ID+VesselType combinations")
    
    # Check for missing values
    missing = master_df.isnull().sum()
    if missing.sum() > 0:
        logging.warning("   Missing values detected:")
        for col, count in missing[missing > 0].items():
            logging.warning(f"      {col}: {count}")
    else:
        logging.info("   ✓ No missing values")
    
    # Verify all files exist
    logging.info("\n🔍 File Existence Verification:")
    files_exist = master_df['FilePath'].apply(lambda x: Path(x).exists())
    if files_exist.all():
        logging.info("   ✓ All audio files verified to exist")
    else:
        missing_count = (~files_exist).sum()
        logging.error(f"   {missing_count} files do not exist!")
        logging.error(f"   Example: {master_df[~files_exist]['FilePath'].iloc[0]}")
    
    return master_df


############################
############################
# this function checks for file existence and readability in master dataset
def validate_dataset(csv_path: str) -> bool:
    """
    Validates that all files in the master CSV actually exist
    and are readable audio files.
    
    Args:
        csv_path: Path to the master metadata CSV file
        
    Returns:
        bool: True if all files are valid, False otherwise
    """
    import librosa
    
    logging.info("\n🔍 Validating dataset...")
    df = pd.read_csv(csv_path)
    
    errors = []
    for idx, row in df.iterrows():
        filepath = Path(row['FilePath'])
        
        # Check 1: File exists
        if not filepath.exists():
            errors.append(f"Row {idx}: File not found - {filepath}")
            continue
        
        # Check 2: File is readable
        try:
            # Try to load just first 1 second as a quick test
            audio, sr = librosa.load(str(filepath), duration=1.0)
            if len(audio) == 0:
                errors.append(f"Row {idx}: Empty audio file - {filepath}")
        except Exception as e:
            errors.append(f"Row {idx}: Cannot read audio - {filepath} ({e})")
    
    if errors:
        logging.error(f"\nFound {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            logging.error(f"   {err}")
        if len(errors) > 10:
            logging.error(f"   ... and {len(errors) - 10} more")
    else:
        logging.info("✓ All files validated successfully!")
    
    return len(errors) == 0


##############################################
# Main execution
if __name__ == '__main__':
    # --- Configuration ---
    if USE_CENTRALIZED_CONFIG:
        # Use paths from centralized config
        DATASET_DIRECTORY = str(RAW_DIR / 'DeepShip-main')
        OUTPUT_FILE = str(PROCESSED_DIR / 'master_metadata.csv')
        logging.info("Using centralized configuration from dir_train_config.py")
    else:
        # Fallback to hardcoded paths
        DATASET_DIRECTORY = 'data/Raw/DeepShip-main'
        OUTPUT_FILE = 'data/Processed/master_metadata.csv'
        logging.warning("Using fallback paths (centralized config not available)")
    # ---------------------
    
    logging.info("🚀 Starting DeepShip Dataset Preprocessing")
    logging.info(f"📂 Dataset path: {DATASET_DIRECTORY}")
    logging.info(f"📄 Output file: {OUTPUT_FILE}\n")
    
    df = create_master_metadata(
        dataset_path=DATASET_DIRECTORY,
        output_csv=OUTPUT_FILE,
        validate=True  # Set False to skip file validation
    )
    
    if df is not None:
        logging.info("\n" + "="*60)
        logging.info("✓ PREPROCESSING COMPLETE!")
        logging.info("="*60)
        logging.info(f"\nNext steps:")
        logging.info(f"1. Review {OUTPUT_FILE} to verify data quality")
        logging.info("2. Proceed to feature extraction (log-mel spectrograms)")
        logging.info("3. Build your CNN model")
    else:
        logging.error("\nPreprocessing failed. Please check errors above.")
     
    # Validate dataset if metadata was successfully created
    if df is not None:
        validate_dataset(OUTPUT_FILE)
        
#################################
# End of Data Preprocessing Module