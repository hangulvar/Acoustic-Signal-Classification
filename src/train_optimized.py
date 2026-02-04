# src/train_optimized.py
"""
Optimized CNN training pipeline for acoustic vessel classification.
Combines best practices from train_test.py and training_test_2.py.

Key improvements:
- Centralized configuration from dir_train_config.py
- Comprehensive error handling and validation
- Configurable model depth (4 or 5 layers)
- Dual scheduler support (ReduceLROnPlateau, CosineAnnealingWarmRestarts)
- Enhanced checkpoint management with automatic cleanup
- Detailed logging and metrics
- Backward compatibility with existing checkpoints
"""

# Suppress warnings BEFORE importing anything
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Standard imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, cast, List
import sys
from collections.abc import Sized
import glob

# Import local modules
try:
    from Dataset_Pytorch import create_dataloaders, AcousticSpectrogramDataset
    from dir_train_config import (
        TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE, NORM_STATS_FILE,
        OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, TENSORBOARD_DIR,
        BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, N_EPOCHS,
        EARLY_STOPPING_PATIENCE, NUM_WORKERS, PIN_MEMORY, DROPOUT_RATE,
        N_CLASSES, USE_CLASS_WEIGHTS, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
        MODEL_DEPTH, USE_RESIDUAL, SCHEDULER_TYPE, COSINE_T0, COSINE_T_MULT, 
        COSINE_ETA_MIN, MAX_CHECKPOINTS_TO_KEEP, SAVE_CHECKPOINT_EVERY,
        TEST_RESULTS_FILE, USE_MIXED_PRECISION
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure you're running from the project root and all modules are available.")
    sys.exit(1)

# Ensure OUTPUT_DIR exists before configuring logging
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'training_optimized.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class VesselCNNOptimized(nn.Module):
    """
    Optimized CNN for vessel classification from log-mel spectrograms.
    
    Features:
    - Configurable depth (4 or 5 convolutional blocks)
    - Optional residual connections
    - Modular block construction
    - Batch normalization and dropout for regularization
    
    Args:
        n_classes: Number of output classes
        dropout: Dropout probability
        depth: Number of convolutional blocks (4 or 5)
        use_residual: Whether to add residual connections
    """
    
    def __init__(
        self, 
        n_classes: int = 4, 
        dropout: float = 0.4,
        depth: int = 4,
        use_residual: bool = False
    ):
        super(VesselCNNOptimized, self).__init__()
        
        if depth not in [4, 5]:
            raise ValueError(f"depth must be 4 or 5, got {depth}")
        
        self.n_classes = n_classes
        self.dropout = dropout
        self.depth = depth
        self.use_residual = use_residual
        
        # Build convolutional blocks
        # Conv Block 1: 1 → 64 channels
        self.conv1 = self._make_block(1, 64)
        
        # Conv Block 2: 64 → 128 channels
        self.conv2 = self._make_block(64, 128)
        
        # Conv Block 3: 128 → 256 channels
        self.conv3 = self._make_block(128, 256)
        
        # Conv Block 4: 256 → 512 channels
        self.conv4 = self._make_block(256, 512)
        
        # Optional Conv Block 5: 512 → 512 channels (for depth=5)
        if depth == 5:
            self.conv5 = self._make_block(512, 512)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block with batch norm and pooling.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        
        Returns:
            Sequential block of layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            Logits tensor of shape (B, n_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply 5th block if depth=5
        if self.depth == 5:
            x = self.conv5(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINER CLASS
# ============================================================================

class TrainerOptimized:
    """
    Optimized trainer with enhanced features:
    - Dual scheduler support (Plateau and Cosine)
    - Complete checkpoint management with cleanup
    - Comprehensive error handling
    - Backward compatibility with old checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        use_class_weights: bool = True,
        checkpoint_dir: Optional[Path] = None,
        scheduler_type: str = 'plateau'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler_type = scheduler_type
        
        # Setup loss function with class weights
        if use_class_weights and hasattr(train_loader.dataset, "get_class_weights"):
            try:
                acoustic_dataset = cast(AcousticSpectrogramDataset, train_loader.dataset)
                class_weights = acoustic_dataset.get_class_weights().to(device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
            except Exception as e:
                logger.warning(f"Failed to compute class weights: {e}. Using unweighted loss.")
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler based on config
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=COSINE_T0,
                T_mult=COSINE_T_MULT,
                eta_min=COSINE_ETA_MIN
            )
            logger.info(f"Using CosineAnnealingWarmRestarts scheduler (T_0={COSINE_T0}, T_mult={COSINE_T_MULT})")
        else:  # plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
                min_lr=1e-7
            )
            logger.info(f"Using ReduceLROnPlateau scheduler (patience={LR_SCHEDULER_PATIENCE})")
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        log_dir = TENSORBOARD_DIR / datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logs: {log_dir}")
        
        # Training state
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with enhanced progress tracking.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch:3d} [Train]',
            leave=True,
            ncols=100
        )
        
        for batch_idx, (specs, labels) in enumerate(pbar):
            try:
                specs = specs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Calculate statistics
                running_loss += loss.item() * specs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
            except RuntimeError as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                raise
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model with enhanced metrics.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch:3d} [Val]  ',
            leave=True,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, (specs, labels) in enumerate(pbar):
                try:
                    specs = specs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
                    
                    # Calculate statistics
                    running_loss += loss.item() * specs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                    
                except RuntimeError as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    raise
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, n_epochs: int = 50, early_stopping_patience: int = 15) -> Dict:
        """
        Main training loop with comprehensive logging and early stopping.
        
        Args:
            n_epochs: Maximum number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
        
        Returns:
            Training history dictionary
        """
        logger.info("="*70)
        logger.info("STARTING OPTIMIZED TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Model depth: {getattr(self.model, 'depth', 'N/A')} layers")
        logger.info(f"Scheduler: {self.scheduler_type}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Log dataset sizes
        train_dataset = cast(Sized, self.train_loader.dataset)
        val_dataset = cast(Sized, self.val_loader.dataset)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        logger.info("="*70)
        
        patience_counter = 0
        
        try:
            for epoch in range(1, n_epochs + 1):
                self.current_epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch(epoch)
                
                # Validate
                val_loss, val_acc = self.validate(epoch)
                
                # Update learning rate scheduler
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:  # cosine
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # Update history
                self.history['train_loss'].append(float(train_loss))
                self.history['train_acc'].append(float(train_acc))
                self.history['val_loss'].append(float(val_loss))
                self.history['val_acc'].append(float(val_acc))
                self.history['learning_rates'].append(float(current_lr))
                
                # Log epoch summary
                logger.info(
                    f"Epoch {epoch:3d}/{n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )
                
                # Save best model based on validation accuracy
                if val_acc > self.best_val_acc:
                    improvement = val_acc - self.best_val_acc
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"✓ New best model! Val Acc: {val_acc*100:.2f}% (+{improvement*100:.2f}%)")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epoch(s). Best: {self.best_val_acc*100:.2f}%")
                
                # Save periodic checkpoint
                if epoch % SAVE_CHECKPOINT_EVERY == 0:
                    self.save_checkpoint(epoch, is_best=False)
                    logger.info(f"Saved periodic checkpoint at epoch {epoch}")
                    self.cleanup_old_checkpoints()
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                    break
                
                # Check for learning rate minimum
                if current_lr < 1e-7:
                    logger.info(f"Learning rate too small ({current_lr:.2e}), stopping training")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            self.save_history()
            self.writer.close()
        
        logger.info("="*70)
        logger.info(f"TRAINING COMPLETE!")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info(f"Total Epochs: {self.current_epoch}")
        logger.info("="*70)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save comprehensive model checkpoint with all training state.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'scheduler_type': self.scheduler_type,
            # Save model architecture config for reconstruction
            'model_config': {
                'n_classes': self.model.n_classes,
                'dropout': self.model.dropout,
                'depth': getattr(self.model, 'depth', 4),
                'use_residual': getattr(self.model, 'use_residual', False)
            }
        }
        
        try:
            if is_best:
                path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, path)
                logger.debug(f"Saved best model to {path}")
            else:
                path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save(checkpoint, path)
                logger.debug(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def cleanup_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the latest N."""
        try:
            pattern = str(self.checkpoint_dir / 'checkpoint_epoch_*.pth')
            checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
            
            # Keep only the latest MAX_CHECKPOINTS_TO_KEEP checkpoints
            if len(checkpoints) > MAX_CHECKPOINTS_TO_KEEP:
                for old_checkpoint in checkpoints[:-MAX_CHECKPOINTS_TO_KEEP]:
                    os.remove(old_checkpoint)
                    logger.debug(f"Removed old checkpoint: {Path(old_checkpoint).name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load model checkpoint with backward compatibility.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.warning("Optimizer state not found in checkpoint")
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                logger.warning("Scheduler state not found in checkpoint")
            
            # Load training state
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.history = checkpoint.get('history', self.history)
            self.current_epoch = checkpoint.get('epoch', 0)
            
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            logger.info(f"Best val acc: {self.best_val_acc*100:.2f}%, Best val loss: {self.best_val_loss:.4f}")
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=4)
            logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    save_path: Optional[Path] = None
) -> Dict:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to use for evaluation
        save_path: Path to save results JSON (optional)
    
    Returns:
        Dictionary containing accuracy, per-class metrics, and confusion matrix
    """
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except ImportError:
        logger.error("scikit-learn is required for evaluation. Install with: pip install scikit-learn")
        raise
    
    logger.info("="*70)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in tqdm(test_loader, desc='Evaluating', ncols=100):
            specs = specs.to(device, non_blocking=True)
            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    test_dataset = cast(AcousticSpectrogramDataset, test_loader.dataset)
    class_names = test_dataset.classes
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'n_samples': len(all_labels)
    }
    
    # Print results
    logger.info(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Total samples: {len(all_labels)}")
    logger.info("\nPer-Class Metrics:")
    logger.info(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-" * 70)
    
    if isinstance(report, dict):
        for cls in class_names:
            metrics = report.get(cls)
            if isinstance(metrics, dict):
                logger.info(
                    f"{cls:<15} "
                    f"{metrics.get('precision', 0)*100:>10.1f}%  "
                    f"{metrics.get('recall', 0)*100:>10.1f}%  "
                    f"{metrics.get('f1-score', 0)*100:>10.1f}%"
                )
        
        logger.info("\nMacro Average:")
        macro_metrics = report.get('macro avg')
        if isinstance(macro_metrics, dict):
            logger.info(
                f"{'Macro Avg':<15} "
                f"{macro_metrics.get('precision', 0)*100:>10.1f}%  "
                f"{macro_metrics.get('recall', 0)*100:>10.1f}%  "
                f"{macro_metrics.get('f1-score', 0)*100:>10.1f}%"
            )
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':<15}" + "".join([f"{cls:>10}" for cls in class_names]))
    for i, cls in enumerate(class_names):
        logger.info(f"{cls:<15}" + "".join([f"{cm[i, j]:>10}" for j in range(len(class_names))]))
    
    logger.info("="*70)
    
    # Save results
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved evaluation results to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_environment():
    """
    Validate that all required files and directories exist before training.
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating environment...")
    
    required_files = [
        TRAIN_META_FILE,
        VAL_META_FILE,
        TEST_META_FILE,
        NORM_STATS_FILE
    ]
    
    all_exist = True
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            all_exist = False
        else:
            logger.debug(f"✓ Found: {file_path}")
    
    if not all_exist:
        logger.error("Please run dataset preparation scripts first!")
        return False
    
    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Environment validation passed")
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    try:
        # Device setup
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            logger.info("Using CPU (GPU not available)")
        
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_meta_path=str(TRAIN_META_FILE),
            val_meta_path=str(VAL_META_FILE),
            test_meta_path=str(TEST_META_FILE),
            norm_stats_path=str(NORM_STATS_FILE),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        # Create model with configured depth
        logger.info(f"Creating model with depth={MODEL_DEPTH}, dropout={DROPOUT_RATE:.2f}")
        model = VesselCNNOptimized(
            n_classes=N_CLASSES,
            dropout=DROPOUT_RATE,
            depth=MODEL_DEPTH,
            use_residual=USE_RESIDUAL
        )
        
        # Create trainer
        trainer = TrainerOptimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            use_class_weights=USE_CLASS_WEIGHTS,
            checkpoint_dir=CHECKPOINT_DIR,
            scheduler_type=SCHEDULER_TYPE
        )
        
        # Train model
        history = trainer.train(
            n_epochs=N_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        
        # Load best model for evaluation
        best_checkpoint_path = CHECKPOINT_DIR / 'best_model.pth'
        if best_checkpoint_path.exists():
            logger.info(f"Loading best model from {best_checkpoint_path}")
            checkpoint = trainer.load_checkpoint(best_checkpoint_path)
            logger.info(f"Best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['best_val_acc']*100:.2f}%)")
        else:
            logger.warning("Best model checkpoint not found, using current model")
        
        # Evaluate on test set
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            save_path=TEST_RESULTS_FILE
        )
        
        logger.info("\n" + "="*70)
        logger.info("OPTIMIZED TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best Validation Accuracy: {trainer.best_val_acc*100:.2f}%")
        logger.info(f"Test Accuracy: {results['accuracy']*100:.2f}%")
        logger.info(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")
        logger.info(f"Results saved to: {TEST_RESULTS_FILE}")
        logger.info(f"TensorBoard logs: {TENSORBOARD_DIR}")
        logger.info("="*70)
        
        return model, results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
