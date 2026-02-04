"""
Comprehensive test suite for the optimized training pipeline.
Tests model architecture, checkpoint compatibility, and configuration without running full training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

# Import the optimized training module
from train_optimized import VesselCNNOptimized, TrainerOptimized, validate_environment
from dir_train_config import (
    N_CLASSES, DROPOUT_RATE, MODEL_DEPTH, SCHEDULER_TYPE,
    LEARNING_RATE, WEIGHT_DECAY
)


class TestModelArchitecture:
    """Test VesselCNNOptimized model architecture."""
    
    def test_model_4_layers_creation(self):
        """Test creating a 4-layer model."""
        model = VesselCNNOptimized(n_classes=4, dropout=0.4, depth=4)
        assert model.depth == 4
        assert model.n_classes == 4
        assert model.dropout == 0.4
        # Should have conv1-4 but not conv5
        assert hasattr(model, 'conv4')
        assert not hasattr(model, 'conv5') or model.conv5 is None
        print("✓ 4-layer model created successfully")
    
    def test_model_5_layers_creation(self):
        """Test creating a 5-layer model."""
        model = VesselCNNOptimized(n_classes=4, dropout=0.5, depth=5)
        assert model.depth == 5
        assert hasattr(model, 'conv5')
        print("✓ 5-layer model created successfully")
    
    def test_model_invalid_depth(self):
        """Test that invalid depth raises error."""
        with pytest.raises(ValueError, match="depth must be 4 or 5"):
            VesselCNNOptimized(n_classes=4, depth=3)
        
        with pytest.raises(ValueError, match="depth must be 4 or 5"):
            VesselCNNOptimized(n_classes=4, depth=6)
        print("✓ Invalid depth validation works")
    
    def test_model_forward_pass_4_layers(self):
        """Test forward pass through 4-layer model."""
        model = VesselCNNOptimized(n_classes=4, depth=4)
        model.eval()
        
        # Create dummy input: batch_size=2, channels=1, height=128, width=216
        dummy_input = torch.randn(2, 1, 128, 216)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
        print(f"✓ 4-layer model forward pass successful. Output shape: {output.shape}")
    
    def test_model_forward_pass_5_layers(self):
        """Test forward pass through 5-layer model."""
        model = VesselCNNOptimized(n_classes=4, depth=5)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(2, 1, 128, 216)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
        print(f"✓ 5-layer model forward pass successful. Output shape: {output.shape}")
    
    def test_model_parameter_count(self):
        """Test that 5-layer model has more parameters than 4-layer."""
        model_4 = VesselCNNOptimized(n_classes=4, depth=4)
        model_5 = VesselCNNOptimized(n_classes=4, depth=5)
        
        params_4 = sum(p.numel() for p in model_4.parameters())
        params_5 = sum(p.numel() for p in model_5.parameters())
        
        assert params_5 > params_4, "5-layer model should have more parameters"
        print(f"✓ Parameter count: 4-layer={params_4:,}, 5-layer={params_5:,}")
    
    def test_model_different_classes(self):
        """Test model with different number of classes."""
        for n_classes in [2, 4, 10]:
            model = VesselCNNOptimized(n_classes=n_classes, depth=4)
            dummy_input = torch.randn(1, 1, 128, 216)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (1, n_classes)
        print("✓ Model works with different number of classes")
    
    def test_model_batch_norm(self):
        """Test that model uses batch normalization."""
        model = VesselCNNOptimized(n_classes=4, depth=4)
        
        # Check that batch norm layers exist
        has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        assert has_batch_norm, "Model should contain BatchNorm2d layers"
        print("✓ Model uses batch normalization")
    
    def test_model_dropout(self):
        """Test that model uses dropout in classifier."""
        model = VesselCNNOptimized(n_classes=4, dropout=0.5, depth=4)
        
        # Check that dropout layers exist in classifier
        dropout_layers = [m for m in model.classifier.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0, "Model should contain Dropout layers"
        
        # Check dropout rate
        for dropout in dropout_layers:
            assert dropout.p == 0.5
        print(f"✓ Model uses dropout (rate={dropout_layers[0].p})")


class TestCheckpointCompatibility:
    """Test checkpoint saving and loading functionality."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dummy_dataloaders(self):
        """Create minimal dummy dataloaders for testing."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create dummy data
        num_samples = 10
        dummy_specs = torch.randn(num_samples, 1, 128, 216)
        dummy_labels = torch.randint(0, 4, (num_samples,))
        
        dataset = TensorDataset(dummy_specs, dummy_labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        return loader, loader  # Use same for train and val
    
    def test_checkpoint_save_and_load(self, temp_checkpoint_dir, dummy_dataloaders):
        """Test saving and loading checkpoint."""
        train_loader, val_loader = dummy_dataloaders
        
        # Create model and trainer
        model = VesselCNNOptimized(n_classes=4, depth=4)
        trainer = TrainerOptimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Save a checkpoint
        trainer.save_checkpoint(epoch=1, is_best=True)
        
        # Check that checkpoint was created
        checkpoint_path = temp_checkpoint_dir / 'best_model.pth'
        assert checkpoint_path.exists(), "Checkpoint file should exist"
        
        # Create new model and load checkpoint
        new_model = VesselCNNOptimized(n_classes=4, depth=4)
        new_trainer = TrainerOptimized(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir
        )
        
        loaded_checkpoint = new_trainer.load_checkpoint(checkpoint_path)
        
        assert loaded_checkpoint['epoch'] == 1
        assert 'model_state_dict' in loaded_checkpoint
        assert 'optimizer_state_dict' in loaded_checkpoint
        print("✓ Checkpoint save and load successful")
    
    def test_checkpoint_contains_required_fields(self, temp_checkpoint_dir, dummy_dataloaders):
        """Test that checkpoint contains all required fields."""
        train_loader, val_loader = dummy_dataloaders
        
        model = VesselCNNOptimized(n_classes=4, depth=4)
        trainer = TrainerOptimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir
        )
        
        trainer.save_checkpoint(epoch=5, is_best=True)
        checkpoint_path = temp_checkpoint_dir / 'best_model.pth'
        
        # Load and verify fields
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_fields = [
            'epoch', 'model_state_dict', 'optimizer_state_dict',
            'scheduler_state_dict', 'best_val_acc', 'best_val_loss',
            'history', 'scheduler_type', 'model_config'
        ]
        
        for field in required_fields:
            assert field in checkpoint, f"Checkpoint missing required field: {field}"
        
        # Verify model config
        assert checkpoint['model_config']['n_classes'] == 4
        assert checkpoint['model_config']['depth'] == 4
        print("✓ Checkpoint contains all required fields")
    
    def test_periodic_checkpoint_cleanup(self, temp_checkpoint_dir, dummy_dataloaders):
        """Test that old periodic checkpoints are removed."""
        train_loader, val_loader = dummy_dataloaders
        
        model = VesselCNNOptimized(n_classes=4, depth=4)
        trainer = TrainerOptimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Save multiple periodic checkpoints
        for epoch in [10, 20, 30, 40, 50]:
            trainer.save_checkpoint(epoch=epoch, is_best=False)
        
        # Cleanup should keep only MAX_CHECKPOINTS_TO_KEEP (3 by default)
        trainer.cleanup_old_checkpoints()
        
        # Count remaining periodic checkpoints
        checkpoints = list(temp_checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        # Should have at most 3 checkpoints (from config)
        from dir_train_config import MAX_CHECKPOINTS_TO_KEEP
        assert len(checkpoints) <= MAX_CHECKPOINTS_TO_KEEP, \
            f"Should keep at most {MAX_CHECKPOINTS_TO_KEEP} checkpoints, found {len(checkpoints)}"
        print(f"✓ Checkpoint cleanup works (kept {len(checkpoints)} checkpoints)")


class TestConfigValidation:
    """Test configuration validation and parameter usage."""
    
    def test_model_uses_config_depth(self):
        """Test that model respects MODEL_DEPTH from config."""
        from dir_train_config import MODEL_DEPTH
        
        model = VesselCNNOptimized(
            n_classes=N_CLASSES,
            dropout=DROPOUT_RATE,
            depth=MODEL_DEPTH
        )
        
        assert model.depth == MODEL_DEPTH
        print(f"✓ Model uses configured depth: {MODEL_DEPTH}")
    
    def test_scheduler_type_configuration(self):
        """Test that both scheduler types can be configured."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create dummy data
        dummy_specs = torch.randn(10, 1, 128, 216)
        dummy_labels = torch.randint(0, 4, (10,))
        dataset = TensorDataset(dummy_specs, dummy_labels)
        loader = DataLoader(dataset, batch_size=2)
        
        model = VesselCNNOptimized(n_classes=4, depth=4)
        
        # Test plateau scheduler
        trainer_plateau = TrainerOptimized(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device='cpu',
            scheduler_type='plateau'
        )
        assert isinstance(trainer_plateau.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        print("✓ Plateau scheduler configured correctly")
        
        # Test cosine scheduler
        trainer_cosine = TrainerOptimized(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device='cpu',
            scheduler_type='cosine'
        )
        assert isinstance(trainer_cosine.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        print("✓ Cosine scheduler configured correctly")
    
    def test_optimizer_configuration(self):
        """Test that optimizer uses configured learning rate and weight decay."""
        from torch.utils.data import TensorDataset, DataLoader
        
        dummy_specs = torch.randn(10, 1, 128, 216)
        dummy_labels = torch.randint(0, 4, (10,))
        dataset = TensorDataset(dummy_specs, dummy_labels)
        loader = DataLoader(dataset, batch_size=2)
        
        model = VesselCNNOptimized(n_classes=4, depth=4)
        trainer = TrainerOptimized(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device='cpu',
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Check optimizer parameters
        assert trainer.optimizer.param_groups[0]['lr'] == LEARNING_RATE
        assert trainer.optimizer.param_groups[0]['weight_decay'] == WEIGHT_DECAY
        print(f"✓ Optimizer configured with LR={LEARNING_RATE}, WD={WEIGHT_DECAY}")


class TestTrainingFunctionality:
    """Test training-related functionality (without full training run)."""
    
    @pytest.fixture
    def setup_trainer(self):
        """Setup a minimal trainer for testing."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create small dummy dataset
        num_samples = 20
        dummy_specs = torch.randn(num_samples, 1, 128, 216)
        dummy_labels = torch.randint(0, 4, (num_samples,))
        
        dataset = TensorDataset(dummy_specs, dummy_labels)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        model = VesselCNNOptimized(n_classes=4, depth=4)
        
        temp_dir = tempfile.mkdtemp()
        trainer = TrainerOptimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            checkpoint_dir=Path(temp_dir)
        )
        
        yield trainer, train_loader, val_loader
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_single_training_epoch(self, setup_trainer):
        """Test running a single training epoch."""
        trainer, _, _ = setup_trainer
        
        # Run one training epoch
        train_loss, train_acc = trainer.train_epoch(epoch=1)
        
        # Verify outputs are reasonable
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0 <= train_acc <= 1
        assert train_loss >= 0
        print(f"✓ Single epoch ran successfully. Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
    
    def test_single_validation_epoch(self, setup_trainer):
        """Test running a single validation epoch."""
        trainer, _, _ = setup_trainer
        
        # Run one validation epoch
        val_loss, val_acc = trainer.validate(epoch=1)
        
        # Verify outputs are reasonable
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 1
        assert val_loss >= 0
        print(f"✓ Single validation ran successfully. Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
    
    def test_history_tracking(self, setup_trainer):
        """Test that training history is tracked correctly."""
        trainer, _, _ = setup_trainer
        
        # Run one epoch
        train_loss, train_acc = trainer.train_epoch(epoch=1)
        val_loss, val_acc = trainer.validate(epoch=1)
        
        # Manually update history (normally done in train())
        trainer.history['train_loss'].append(float(train_loss))
        trainer.history['train_acc'].append(float(train_acc))
        trainer.history['val_loss'].append(float(val_loss))
        trainer.history['val_acc'].append(float(val_acc))
        
        # Verify history was updated
        assert len(trainer.history['train_loss']) == 1
        assert len(trainer.history['val_acc']) == 1
        print("✓ Training history tracking works")
    
    def test_learning_rate_scheduler_step(self, setup_trainer):
        """Test that learning rate scheduler can step."""
        trainer, _, _ = setup_trainer
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Run validation to get loss
        val_loss, _ = trainer.validate(epoch=1)
        
        # Step the scheduler
        if trainer.scheduler_type == 'plateau':
            trainer.scheduler.step(val_loss)
        else:
            trainer.scheduler.step()
        
        # LR should remain same or change depending on scheduler
        current_lr = trainer.optimizer.param_groups[0]['lr']
        assert current_lr > 0
        print(f"✓ Scheduler step works. Initial LR: {initial_lr:.6f}, Current LR: {current_lr:.6f}")


class TestBackwardCompatibility:
    """Test backward compatibility with old checkpoint formats."""
    
    def test_load_checkpoint_missing_fields(self):
        """Test loading checkpoint with missing optional fields."""
        import tempfile
        
        # Create a minimal checkpoint (simulating old format)
        model = VesselCNNOptimized(n_classes=4, depth=4)
        old_checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'best_val_acc': 0.85
            # Missing: optimizer_state_dict, scheduler_state_dict, history, etc.
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            torch.save(old_checkpoint, f.name)
            temp_path = Path(f.name)
        
        try:
            # Create dummy loaders
            from torch.utils.data import TensorDataset, DataLoader
            dummy_specs = torch.randn(10, 1, 128, 216)
            dummy_labels = torch.randint(0, 4, (10,))
            dataset = TensorDataset(dummy_specs, dummy_labels)
            loader = DataLoader(dataset, batch_size=2)
            
            # Try to load with new trainer
            new_model = VesselCNNOptimized(n_classes=4, depth=4)
            trainer = TrainerOptimized(
                model=new_model,
                train_loader=loader,
                val_loader=loader,
                device='cpu'
            )
            
            # Should load successfully despite missing fields
            loaded = trainer.load_checkpoint(temp_path)
            
            assert loaded['epoch'] == 10
            assert trainer.best_val_acc == 0.85
            print("✓ Backward compatibility with old checkpoint format works")
        
        finally:
            temp_path.unlink()


def run_all_tests():
    """Run all tests with verbose output."""
    print("\n" + "="*70)
    print("RUNNING OPTIMIZED TRAINING PIPELINE TESTS")
    print("="*70 + "\n")
    
    # Run pytest with verbose output
    pytest.main([__file__, '-v', '--tb=short', '-s'])


if __name__ == '__main__':
    run_all_tests()
