# src/visualize.py
"""
Visualization and analysis utilities for the acoustic classification project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Optional, List
import torch

from dir_train_config import (
    TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE,
    NORM_STATS_FILE, CHECKPOINT_DIR, RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
sns.set_style('whitegrid')


# ============================================================================
# DATA DISTRIBUTION ANALYSIS
# ============================================================================

def plot_dataset_distribution(save_path: Optional[Path] = None):
    """
    Plot dataset distribution across train/val/test splits.
    Shows both vessel counts and chunk counts.
    """
    # Load metadata
    train_df = pd.read_csv(TRAIN_META_FILE)
    val_df = pd.read_csv(VAL_META_FILE)
    test_df = pd.read_csv(TEST_META_FILE)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall split distribution (chunks)
    split_counts = {
        'Train': len(train_df),
        'Val': len(val_df),
        'Test': len(test_df)
    }
    
    axes[0, 0].bar(split_counts.keys(), split_counts.values(), color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0, 0].set_title('Spectrogram Chunks Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Chunks', fontsize=12)
    for i, (k, v) in enumerate(split_counts.items()):
        axes[0, 0].text(i, v + 20, f'{v}\n({v/sum(split_counts.values())*100:.1f}%)', 
                        ha='center', fontsize=10)
    
    # 2. Vessel distribution
    vessel_counts = {
        'Train': train_df['VesselGroupKey'].nunique(),
        'Val': val_df['VesselGroupKey'].nunique(),
        'Test': test_df['VesselGroupKey'].nunique()
    }
    
    axes[0, 1].bar(vessel_counts.keys(), vessel_counts.values(), color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0, 1].set_title('Unique Vessels Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Vessels', fontsize=12)
    for i, (k, v) in enumerate(vessel_counts.items()):
        axes[0, 1].text(i, v + 1, f'{v}\n({v/sum(vessel_counts.values())*100:.1f}%)', 
                        ha='center', fontsize=10)
    
    # 3. Class distribution per split
    class_data = []
    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        for cls, count in df['VesselType'].value_counts().items():
            class_data.append({
                'Split': split_name,
                'Class': cls,
                'Count': count
            })
    
    class_df = pd.DataFrame(class_data)
    pivot_df = class_df.pivot(index='Class', columns='Split', values='Count')
    
    pivot_df.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1, 0].set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Chunks', fontsize=12)
    axes[1, 0].set_xlabel('Vessel Class', fontsize=12)
    axes[1, 0].legend(title='Split')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Chunks per vessel histogram
    chunks_per_vessel_train = train_df.groupby('VesselGroupKey').size()
    chunks_per_vessel_val = val_df.groupby('VesselGroupKey').size()
    chunks_per_vessel_test = test_df.groupby('VesselGroupKey').size()
    
    axes[1, 1].hist([chunks_per_vessel_train, chunks_per_vessel_val, chunks_per_vessel_test],
                    bins=20, label=['Train', 'Val', 'Test'], color=['#2ecc71', '#3498db', '#e74c3c'],
                    alpha=0.7)
    axes[1, 1].set_title('Chunks per Vessel Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Chunks', fontsize=12)
    axes[1, 1].set_ylabel('Number of Vessels', fontsize=12)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================

def plot_training_history(history_path: Optional[Path] = None, save_path: Optional[Path] = None):
    """Plot training and validation metrics over epochs."""
    if history_path is None:
        history_path = CHECKPOINT_DIR / 'training_history.json'
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    train_acc = [acc * 100 for acc in history['train_acc']]
    val_acc = [acc * 100 for acc in history['val_acc']]
    
    axes[1].plot(epochs, train_acc, 'o-', label='Train Accuracy', linewidth=2, markersize=4)
    axes[1].plot(epochs, val_acc, 's-', label='Val Accuracy', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(val_acc)
    axes[1].axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    axes[1].text(best_epoch, best_val_acc + 1, f'{best_val_acc:.1f}%', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(
    results_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    normalize: bool = True
):
    """Plot confusion matrix from test results."""
    if results_path is None:
        results_path = RESULTS_DIR / 'test_results.json'
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    cm = np.array(results['confusion_matrix'])
    class_names = results['class_names']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# PER-CLASS PERFORMANCE VISUALIZATION
# ============================================================================

def plot_per_class_metrics(results_path: Optional[Path] = None, save_path: Optional[Path] = None):
    """Plot per-class precision, recall, and F1 scores."""
    if results_path is None:
        results_path = RESULTS_DIR / 'test_results.json'
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    report = results['classification_report']
    class_names = results['class_names']
    
    # Extract metrics
    metrics_data = []
    for cls in class_names:
        metrics_data.append({
            'Class': cls,
            'Precision': report[cls]['precision'] * 100,
            'Recall': report[cls]['recall'] * 100,
            'F1-Score': report[cls]['f1-score'] * 100
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Vessel Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved per-class metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# SPECTROGRAM VISUALIZATION
# ============================================================================

def plot_sample_spectrograms(
    n_samples: int = 4,
    dataset: str = 'train',
    save_path: Optional[Path] = None
):
    """Plot sample spectrograms from the dataset."""
    
    # Load metadata
    if dataset == 'train':
        meta_file = TRAIN_META_FILE
    elif dataset == 'val':
        meta_file = VAL_META_FILE
    else:
        meta_file = TEST_META_FILE
    
    df = pd.read_csv(meta_file)
    
    # Sample one from each class
    sampled_rows = []
    for cls in df['VesselType'].unique():
        cls_df = df[df['VesselType'] == cls]
        sampled_rows.append(cls_df.sample(1, random_state=42).iloc[0])
    
    n_samples = min(n_samples, len(sampled_rows))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, row in enumerate(sampled_rows[:n_samples]):
        spec = np.load(row['FeaturePath'])
        
        im = axes[i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f"{row['VesselType']} (ID: {row.get('VesselGroupKey', 'N/A')})", 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Frames', fontsize=10)
        axes[i].set_ylabel('Mel Frequency Bins', fontsize=10)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Spectrograms from {dataset.capitalize()} Set', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved sample spectrograms to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# AUGMENTATION COMPARISON
# ============================================================================

def plot_augmentation_comparison(
    spectrogram_path: str,
    n_augmented: int = 5,
    save_path: Optional[Path] = None
):
    """Compare original spectrogram with augmented versions."""
    from augmentation import get_augmentation_pipeline
    
    # Load spectrogram
    original = np.load(spectrogram_path)
    
    # Get augmenter
    augmenter = get_augmentation_pipeline('train')
    
    # Create augmented versions
    n_rows = 2
    n_cols = n_augmented + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))
    
    # Plot original
    axes[0, 0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Plot augmented versions
    for i in range(n_augmented):
        augmented = augmenter(original)
        
        # Top row: augmented
        axes[0, i + 1].imshow(augmented, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i + 1].set_title(f'Augmented {i+1}', fontsize=12)
        axes[0, i + 1].axis('off')
        
        # Bottom row: difference
        diff = np.abs(original - augmented)
        im = axes[1, i + 1].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[1, i + 1].set_title(f'Difference {i+1}', fontsize=10)
        axes[1, i + 1].axis('off')
        plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046)
    
    plt.suptitle('Augmentation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved augmentation comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MAIN VISUALIZATION PIPELINE
# ============================================================================

def generate_all_visualizations(output_dir: Optional[Path] = None):
    """Generate all visualization plots."""
    if output_dir is None:
        output_dir = RESULTS_DIR / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Generating all visualizations...")
    
    try:
        plot_dataset_distribution(output_dir / 'dataset_distribution.png')
    except Exception as e:
        logging.error(f"Failed to plot dataset distribution: {e}")
    
    try:
        plot_training_history(save_path=output_dir / 'training_history.png')
    except Exception as e:
        logging.error(f"Failed to plot training history: {e}")
    
    try:
        plot_confusion_matrix(save_path=output_dir / 'confusion_matrix.png')
    except Exception as e:
        logging.error(f"Failed to plot confusion matrix: {e}")
    
    try:
        plot_per_class_metrics(save_path=output_dir / 'per_class_metrics.png')
    except Exception as e:
        logging.error(f"Failed to plot per-class metrics: {e}")
    
    try:
        plot_sample_spectrograms(save_path=output_dir / 'sample_spectrograms.png')
    except Exception as e:
        logging.error(f"Failed to plot sample spectrograms: {e}")
    
    logging.info(f"✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--distribution', action='store_true', help='Plot dataset distribution')
    parser.add_argument('--history', action='store_true', help='Plot training history')
    parser.add_argument('--confusion', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--metrics', action='store_true', help='Plot per-class metrics')
    parser.add_argument('--samples', action='store_true', help='Plot sample spectrograms')
    
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = RESULTS_DIR / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        generate_all_visualizations(vis_dir)
    else:
        if args.distribution:
            plot_dataset_distribution(vis_dir / 'dataset_distribution.png')
        if args.history:
            plot_training_history(save_path=vis_dir / 'training_history.png')
        if args.confusion:
            plot_confusion_matrix(save_path=vis_dir / 'confusion_matrix.png')
        if args.metrics:
            plot_per_class_metrics(save_path=vis_dir / 'per_class_metrics.png')
        if args.samples:
            plot_sample_spectrograms(save_path=vis_dir / 'sample_spectrograms.png')
    
    logging.info("Visualization complete! ✨")