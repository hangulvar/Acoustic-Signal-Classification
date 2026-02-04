# src/train.py
"""
CNN training pipeline for acoustic vessel classification.
Production-ready implementation with robust error handling.
"""
# Suppress TensorFlow warnings BEFORE importing anything
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
from typing import Dict, Tuple, Optional, cast, Any
import sys
from collections.abc import Sized

# Import local modules and configuration
try:
    from Dataset_Pytorch import create_dataloaders, AcousticSpectrogramDataset
    # Import all necessary parameters from the single source of truth
    from dir_train_config import (
        TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE, NORM_STATS_FILE,
        OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, TENSORBOARD_DIR,
        BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, N_EPOCHS,TEST_RESULTS_FILE,
        EARLY_STOPPING_PATIENCE, NUM_WORKERS, PIN_MEMORY, DROPOUT_RATE,
        N_CLASSES, USE_CLASS_WEIGHTS, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE
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
    - 4 convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier with dropout

    Args:
        n_classes: Number of output classes
        dropout: Dropout probability
    """
    def __init__(self, n_classes: int, dropout: float):
        super(VesselCNN, self).__init__()
        self.n_classes = n_classes
        self.dropout = dropout

        # Conv Block 1: 1 -> 64 channels
        self.conv1 = self._make_block(1, 64)
        # Conv Block 2: 64 -> 128 channels
        self.conv2 = self._make_block(64, 128)
        # Conv Block 3: 128 -> 256 channels
        self.conv3 = self._make_block(128, 256)
        # Conv Block 4: 256 -> 512 channels
        self.conv4 = self._make_block(256, 512)
        # REMOVED 5th convolutional block to simplify the model and prevent overfitting.
        # A simpler model is often easier to train and generalizes better.

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

        self._initialize_weights()

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINER CLASS (Largely the same, already well-written)
# ============================================================================

class Trainer:
    """Handles model training, validation, and checkpointing."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        learning_rate: float,
        weight_decay: float,
        use_class_weights: bool,
        checkpoint_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, min_lr=1e-7
        )

        # Setup TensorBoard
        log_dir = TENSORBOARD_DIR / datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logs: {log_dir}")

        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:3d} [Train]', leave=True, ncols=100)

        for specs, labels in pbar:
            specs, labels = specs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(specs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        return running_loss / total, correct / total

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc=f'          [Val]  ', leave=True, ncols=100)

        with torch.no_grad():
            for specs, labels in pbar:
                specs, labels = specs.to(self.device), labels.to(self.device)
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * specs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        return running_loss / total, correct / total

    def train(self, n_epochs: int, early_stopping_patience: int) -> Dict:
        """Main training loop with early stopping."""
        logger.info("="*70)
        logger.info("STARTING TRAINING")
        logger.info(f"Device: {self.device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

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
                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc = self.validate()
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log to TensorBoard and history
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rates'].append(current_lr)

                logger.info(
                    f"Epoch {epoch:3d}/{n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

                if val_acc > self.best_val_acc:
                    improvement = val_acc - self.best_val_acc
                    self.best_val_acc = val_acc
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"✓ New best model saved! Val Acc: {val_acc*100:.2f}% (+{improvement*100:.2f}%)")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"✗ No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                    logger.info(f"No improvement for {patience_counter} epoch(s)")
                                    
                # Save periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, is_best=False)
                    logger.info(f"Saved checkpoint at epoch {epoch}")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience}).")
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
                

        logger.info(f"\nTRAINING COMPLETE! Best Validation Accuracy: {self.best_val_acc*100:.2f}%\n")
        self.writer.close()
        self.save_history()
        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
        path = self.checkpoint_dir / ('best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, path)

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")

# ============================================================================
# EVALUATION FUNCTION (Corrected)
# ============================================================================
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, save_path: Path) -> Dict:
    """Evaluate model on the test set."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except ImportError:
        logger.error("scikit-learn is required for evaluation. Run: pip install scikit-learn")
        raise

    logger.info("="*70 + "\nEVALUATING MODEL ON TEST SET\n" + "="*70)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in tqdm(test_loader, desc='Evaluating', ncols=100):
            specs = specs.to(device)
            outputs = model(specs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- ROBUST FIX STARTS HERE ---
    # 1. Calculate accuracy directly using accuracy_score for clarity and type safety.
    accuracy = accuracy_score(all_labels, all_preds)

    test_dataset = cast(AcousticSpectrogramDataset, test_loader.dataset)
    class_names = test_dataset.classes
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # 2. Use the safely calculated 'accuracy' variable in the results dictionary.
    results = {
        'accuracy': accuracy, # Use the variable calculated above
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    # --- ROBUST FIX ENDS HERE ---

    logger.info(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    logger.info("\n%s",classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    logger.info("\nConfusion Matrix:\n" + str(cm))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nSaved evaluation results to {save_path}")

    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main training and evaluation pipeline."""
    try:
        # Device setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

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

        # Create model
        logger.info(f"Creating model for {N_CLASSES} classes with {DROPOUT_RATE:.1f} dropout.")
        model = VesselCNN(n_classes=N_CLASSES, dropout=DROPOUT_RATE)

        # Create trainer
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            use_class_weights=USE_CLASS_WEIGHTS, checkpoint_dir=CHECKPOINT_DIR
        )

        # Train model
        trainer.train(n_epochs=N_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)

        # Load best model for evaluation
        best_checkpoint_path = CHECKPOINT_DIR / 'best_model.pth'
        if best_checkpoint_path.exists():
            logger.info(f"Loading best model from {best_checkpoint_path} for final evaluation.")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("Best model checkpoint not found. Evaluating with the last model state.")

        # Evaluate on test set
        evaluate_model(
            model=model, test_loader=test_loader, device=device, save_path=TEST_RESULTS_FILE
        )

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()