# src/train.py
"""
CNN training pipeline for acoustic vessel classification.
Production-ready implementation with robust error handling.
"""
# Suppress TensorFlow warnings BEFORE importing anything
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN message

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Standard imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torch.cuda.amp import autocast, GradScaler # For mixed precision training, in future
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, cast, Any
import sys
from collections.abc import Sized

# Import local modules
try:
    from Dataset_Pytorch import create_dataloaders, AcousticSpectrogramDataset
    from dir_train_config import (
        TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE, 
        NORM_STATS_FILE, OUTPUT_DIR, CHECKPOINT_DIR,
        RESULTS_DIR, TENSORBOARD_DIR
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
        logging.FileHandler(OUTPUT_DIR / 'training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class VesselCNN(nn.Module):
    """
    CNN for vessel classification from log-mel spectrograms.
    
    Architecture:
    - 5 convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier with dropout
    
    Args:
        n_classes: Number of output classes (default: 4)
        dropout: Dropout probability (default: 0.4)
    """
    
    def __init__(self, n_classes: int = 4, dropout: float = 0.4):
        super(VesselCNN, self).__init__()
        
        self.n_classes = n_classes
        self.dropout = dropout
        
        # Conv Block 1: 1 -> 64 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 2: 64 -> 128 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 3: 128 -> 256 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 4: 256 -> 512 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Extra Conv Block 5: 512 -> 512 channels
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
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
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """
    Handles model training, validation, and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use for training ('cuda' or 'cpu')
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        use_class_weights: Whether to use class weights in loss function
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_class_weights: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup loss function
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
        self.optimizer = optim.AdamW( # changed from Adam to AdamW
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        """
       
        # Setup optimizer - SGD with momentum and Nesterov
        self.optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate * 10,  # SGD needs higher LR (0.01)
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
        )   
         """
        """                        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            #verbose=True
            min_lr=1e-6
        )
        """
        # Learning rate scheduler - Cosine Annealing with Warm Restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6
        )
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        log_dir = TENSORBOARD_DIR / datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logs: {log_dir}")
        
        # Training state
        self.best_val_acc = 0.0
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
        Train for one epoch.
        
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
        Validate the model.
        
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
        Main training loop with early stopping.
        
        Args:
            n_epochs: Maximum number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
        
        Returns:
            Training history dictionary
        """
        logger.info("="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        # Robust Fix:
        train_dataset = cast(Sized, self.train_loader.dataset)
        val_dataset = cast(Sized, self.val_loader.dataset)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        #logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        #logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
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
                #self.scheduler.step(val_loss)
                self.scheduler.step()  # Call after each epoch
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
                
                # Save best model
                if val_acc > self.best_val_acc:
                    improvement = val_acc - self.best_val_acc
                    self.best_val_acc = val_acc
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"✓ New best model! Val Acc: {val_acc*100:.2f}% (+{improvement*100:.2f}%)")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epoch(s)")
                
                # Save periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, is_best=False)
                    logger.info(f"Saved checkpoint at epoch {epoch}")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                    break
                
                # Check for learning rate minimum
                if current_lr < 1e-6:
                    logger.info(f"Learning rate too small ({current_lr:.2e}), stopping training")
                    break
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Cleanup
            self.save_history()
            self.writer.close()
        
        logger.info("="*70)
        logger.info(f"TRAINING COMPLETE!")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        logger.info(f"Total Epochs: {self.current_epoch}")
        logger.info("="*70)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
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
            'history': self.history
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
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            self.history = checkpoint['history']
            self.current_epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
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
    Evaluate model on test set with detailed metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to use for evaluation
        save_path: Path to save results JSON (optional)
    
    Returns:
        Dictionary containing accuracy, per-class metrics, and confusion matrix
    """
    try:
        from sklearn.metrics import classification_report, confusion_matrix
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
    
    accuracy = float(np.mean(all_preds == all_labels))
    
    # Per-class metrics
    # Robust Fix:
    test_dataset = cast(AcousticSpectrogramDataset, test_loader.dataset)
    class_names = test_dataset.classes
    #class_names = test_loader.dataset.classes
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
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'n_samples': len(all_labels)
    }
    
    # Print results
    logger.info(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Total samples: {len(all_labels)}")
    logger.info("\nPer-Class Metrics:")
    logger.info(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info("-" * 70)
    
     # THE ROBUST FIX: Check the type of the 'report' variable itself first.
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
    else:
        # This case would happen if report was a string
        logger.warning("Classification report is not a dictionary. Cannot display detailed metrics.")
     
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':>15}" + "".join([f"{cls:>10}" for cls in class_names]))
    for i, cls in enumerate(class_names):
        logger.info(f"{cls:>15}" + "".join([f"{cm[i, j]:>10}" for j in range(len(class_names))]))
    
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
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    BATCH_SIZE = 40
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 80
    EARLY_STOPPING_PATIENCE = 20
    NUM_WORKERS = 6
    
    try:
        # Device setup
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            logger.info("Using CPU (GPU not available)")
        
        # Verify required files exist
        required_files = [
            TRAIN_META_FILE,
            VAL_META_FILE,
            TEST_META_FILE,
            NORM_STATS_FILE
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file not found: {file_path}")
                logger.error("Please run prepare_dataset.py first")
                sys.exit(1)
        
        # Create output directories
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            str(TRAIN_META_FILE),
            str(VAL_META_FILE),
            str(TEST_META_FILE),
            str(NORM_STATS_FILE),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=(device == 'cuda')
        )
        
        # Create model
        # Robust Fix:
        train_dataset = cast(AcousticSpectrogramDataset, train_loader.dataset)
        n_classes = len(train_dataset.classes)
        logger.info(f"Creating model for {n_classes} classes: {train_dataset.classes}")
        #n_classes = len(train_loader.dataset.classes)
        #logger.info(f"Creating model for {n_classes} classes: {train_loader.dataset.classes}")
        model = VesselCNN(n_classes=n_classes, dropout=0.5)
        
        # Log model architecture
        logger.info(f"Model: {model.__class__.__name__}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            use_class_weights=True,
            checkpoint_dir=CHECKPOINT_DIR
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
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['best_val_acc']*100:.2f}%)")
        else:
            logger.warning("Best model checkpoint not found, using current model")
        
        # Evaluate on test set
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            save_path=RESULTS_DIR / 'test_results.json'
        )
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Best Validation Accuracy: {trainer.best_val_acc*100:.2f}%")
        logger.info(f"Test Accuracy: {results['accuracy']*100:.2f}%")
        logger.info(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")
        logger.info(f"Results saved to: {RESULTS_DIR / 'test_results.json'}")
        logger.info(f"TensorBoard logs: {TENSORBOARD_DIR}")
        logger.info("="*70)
        
        return model, results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()