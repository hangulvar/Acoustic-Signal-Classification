#################################
# Preprocessing Module - DeepShip 
#################################
#################################
import pandas as pd
from pathlib import Path
import warnings


############################################
# this function appends metadata from all classes
# and validates file existence
# preprocess.py

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
    
    print("=" * 60)
    print("Starting metadata aggregation with file validation...")
    print("=" * 60)
    
    total_audio_files = 0
    total_matched = 0
    total_unmatched_audio = 0
    total_unmatched_meta = 0
    
    for vessel in vessel_types:
        class_dir = dataset_root / vessel
        
        if not class_dir.is_dir():
            print(f"⚠️  Warning: Directory not found for class: {vessel}")
            continue
        
        # STEP 1: Find all actual audio files
        audio_files = list(class_dir.glob('*.[wW][aA][vV]'))
        total_audio_files += len(audio_files)
        
        print(f"\n📁 Processing {vessel}:")
        print(f"   Found {len(audio_files)} audio files")
        
        # Create dictionary: {file_id: file_path}
        audio_dict = {f.stem: f for f in audio_files}
        
        # STEP 2: Find and load metafile
        try:
            metafile_path = next(class_dir.glob('*metafile*'))
            print(f"   Found metafile: {metafile_path.name}")
        except StopIteration:
            print(f"   ⚠️  No metafile found, using only audio files")
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
            
            print(f"   Loaded {len(df_meta)} metadata rows")
            
        except Exception as e:
            print(f"   ❌ Error reading metafile: {e}")
            import traceback
            traceback.print_exc()
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
            
            print(f"   ✅ Matched: {matched}")
            if unmatched_meta > 0:
                print(f"   ⚠️  Metadata without audio: {unmatched_meta}")
                # Show which IDs are missing
                missing_ids = df_meta[~df_meta['FileExists']]['ID'].tolist()[:5]
                print(f"      Example missing IDs: {missing_ids}")
            
            if unmatched_audio > 0:
                print(f"   ⚠️  Audio without metadata: {unmatched_audio}")
                # Show which audio files have no metadata
                missing_audio = [k for k in audio_dict.keys() if k not in df_meta['ID'].values][:5]
                print(f"      Example orphaned audio: {missing_audio}")
            
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
            print(f"   ⚠️  Warning: Some FilePaths are empty!")
        
        all_metadata.append(df_meta)
    
    # STEP 5: Combine all dataframes
    if not all_metadata:
        print("\n❌ No metadata found! Check your dataset path.")
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
        print(f"\n⚠️  Removed {removed} rows with invalid FilePaths")
    
    # Save to CSV
    master_df.to_csv(output_csv, index=False)
    
    # STEP 6: Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total audio files found:      {total_audio_files}")
    print(f"Total matched records:        {total_matched}")
    print(f"Metadata without audio:       {total_unmatched_meta}")
    print(f"Audio without metadata:       {total_unmatched_audio}")
    print(f"\n✅ Master metadata saved to: {output_csv}")
    print(f"Final dataset size:           {len(master_df)} samples")
    
    # Class distribution
    print("\n📈 Class Distribution:")
    class_counts = master_df['VesselType'].value_counts()
    for vessel, count in class_counts.items():
        percentage = (count / len(master_df)) * 100
        print(f"   {vessel:12s}: {count:4d} ({percentage:5.2f}%)")
    
    # Sample data
    print("\n🔍 Sample Records (first 3):")
    print("-" * 60)
    sample_cols = ['ID', 'VesselType', 'VesselName', 'Duration']
    if all(col in master_df.columns for col in sample_cols):
        print(master_df[sample_cols].head(3).to_string(index=False))
    else:
        print(master_df.head(3).to_string(index=False))
    
    # Data quality checks
    print("\n🔍 Data Quality Checks:")
    missing = master_df.isnull().sum()
    if missing.sum() > 0:
        print("   Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count}")
    else:
        print("   ✅ No missing values")
    
    # Verify all files exist
    print("\n🔍 File Existence Verification:")
    files_exist = master_df['FilePath'].apply(lambda x: Path(x).exists())
    if files_exist.all():
        print("   ✅ All audio files verified to exist")
    else:
        missing_count = (~files_exist).sum()
        print(f"   ⚠️  {missing_count} files do not exist!")
        print(f"   Example: {master_df[~files_exist]['FilePath'].iloc[0]}")
    
    return master_df


############################
############################
# this function checks for file existence and readability in master dataset
def validate_dataset(csv_path: str):
    """
    Validates that all files in the master CSV actually exist
    and are readable audio files.
    """
    import librosa
    
    print("\n🔍 Validating dataset...")
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
        print(f"\n❌ Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    else:
        print("✅ All files validated successfully!")
    
    return len(errors) == 0


##############################################
# Main execution
if __name__ == '__main__':
    # --- Configuration ---
    DATASET_DIRECTORY = 'data/Raw/DeepShip-main'  # ← CHANGE THIS!
    OUTPUT_FILE = 'data/Processed/master_metadata.csv'
    # ---------------------
    
    print("🚀 Starting DeepShip Dataset Preprocessing")
    print(f"📂 Dataset path: {DATASET_DIRECTORY}")
    print(f"📄 Output file: {OUTPUT_FILE}\n")
    
    df = create_master_metadata(
        dataset_path=DATASET_DIRECTORY,
        output_csv=OUTPUT_FILE,
        validate=True  # Set False to skip file validation
    )
    
    if df is not None:
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Review {OUTPUT_FILE} to verify data quality")
        print("2. Proceed to feature extraction (log-mel spectrograms)")
        print("3. Build your CNN model")
    else:
        print("\n❌ Preprocessing failed. Please check errors above.")
     
    validate_dataset(OUTPUT_FILE)
#################################
# End of Data Preprocessing Module