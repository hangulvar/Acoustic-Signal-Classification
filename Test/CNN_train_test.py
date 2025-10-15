# src/train.py
"""
CNN training pipeline for acoustic vessel classification.
Integrates data loading, augmentation, training, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, cast

from Dataset_Pytorch import create_dataloaders, AcousticSpectrogramDataset
from dir_train_config import (
    TRAIN_META_FILE, VAL_META_FILE, TEST_META_FILE, NORM_STATS_FILE,
    OUTPUT_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class VesselCNN(nn.Module):
    """
    CNN for vessel classification from log-mel spectrograms.
    
    Architecture:
    - 4 convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier
    """
    
    def __init__(self, n_classes: int = 4, dropout: float = 0.5):
        super(VesselCNN, self).__init__()
        
        # Conv Block 1: 1 -> 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # /2
        )
        
        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # /4
        )
        
        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # /8
        )
        
        # Conv Block 4: 128 -> 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # /16
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for conv layers."""
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
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Logits tensor (B, n_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class Trainer:
    """Handles training, validation, and model checkpointing."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
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
        if use_class_weights:
            class_weights = train_loader.dataset.get_class_weights().to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer with weight decay
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
            #,verbose=True
        )
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or OUTPUT_DIR / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        log_dir = OUTPUT_DIR / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(log_dir)
        logging.info(f"TensorBoard logs: {log_dir}")
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for specs, labels in pbar:
            specs, labels = specs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(specs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for specs, labels in pbar:
                specs, labels = specs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * specs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, n_epochs: int = 50, early_stopping_patience: int = 15):
        """Main training loop."""
        logging.info(f"Starting training for {n_epochs} epochs...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log epoch summary
            logging.info(
                f"Epoch {epoch}/{n_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                logging.info(f"✓ New best model! Val Acc: {val_acc*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final history
        self.save_history()
        self.writer.close()
        
        logging.info(f"Training complete! Best Val Acc: {self.best_val_acc*100:.2f}%")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logging.info(f"Saved training history to {history_path}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str = 'cuda',
    save_path: Optional[Path] = None
) -> Dict:
    """
    Evaluate model on test set with detailed metrics.
    
    Returns:
        Dictionary with accuracy, per-class metrics, confusion matrix
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for specs, labels in tqdm(test_loader, desc='Evaluating'):
            specs = specs.to(device)
            outputs = model(specs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Per-class metrics
    class_names = test_loader.dataset.classes
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # Print results
    logging.info("\n" + "="*60)
    logging.info("TEST SET EVALUATION")
    logging.info("="*60)
    logging.info(f"Overall Accuracy: {accuracy*100:.2f}%")
    logging.info("\nPer-Class Metrics:")
    for cls in class_names:
        metrics = report[cls]
        # Add a check to ensure 'metrics' is a dictionary
        # This satisfies the type checker and is good practice
        if isinstance(metrics, dict):
            logging.info(
                f"  {cls:15s}: "
                f"Precision={metrics.get('precision', 0)*100:.1f}%, "
                f"Recall={metrics.get('recall', 0)*100:.1f}%, "
                f"F1={metrics.get('f1-score', 0)*100:.1f}%"
            )
    logging.info("="*60 + "\n")
    
    # Save results
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved evaluation results to {save_path}")
    
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15
    NUM_WORKERS = 4
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
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
    # We use cast to inform the type checker of the specific Dataset subclass we are using.
    # This resolves the warning without changing the code's runtime behavior.
    acoustic_dataset = cast(AcousticSpectrogramDataset, train_loader.dataset)
    n_classes = len(acoustic_dataset.classes)
    logging.info(f"Creating model for {n_classes} classes...")
    model = VesselCNN(n_classes=n_classes, dropout=0.5)
    
    # Log model summary
    sample_input = torch.randn(1, 1, 128, 256)
    with torch.no_grad():
        sample_output = model(sample_input)
    logging.info(f"Model input shape: {sample_input.shape}")
    logging.info(f"Model output shape: {sample_output.shape}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        use_class_weights=True
    )
    
    # Train model
    trainer.train(n_epochs=N_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    
    # Load best model for evaluation
    best_checkpoint = torch.load(trainer.checkpoint_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on test set
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=OUTPUT_DIR / 'test_results.json'
    )
    
    logging.info("Training pipeline complete! ✨")
    
    return model, results


if __name__ == '__main__':
    main()