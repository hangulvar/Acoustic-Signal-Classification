# test_gpu_memory.py
import torch
import numpy as np
from pathlib import Path

print("="*70)
print("GPU MEMORY TEST")
print("="*70)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Simulate your model and batch
print("\nTesting memory requirements...")

try:
    # Simulate spectrogram batch (B, C, H, W)
    for batch_size in [8, 16, 24, 32]:
        torch.cuda.empty_cache()  # Clear memory
        
        # Create dummy batch
        batch = torch.randn(batch_size, 1, 128, 256, device=device)
        
        # Create dummy model (simplified version of VesselCNN)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 4)
        ).to(device)
        
        # Forward pass
        output = model(batch)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check memory
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        memory_cached = torch.cuda.max_memory_reserved() / 1e9
        
        print(f"  Batch size {batch_size:2d}: {memory_used:.2f} GB used, "
              f"{memory_cached:.2f} GB cached", end="")
        
        if memory_used < 3.5:  # Safe margin
            print(" ✓ SAFE")
        elif memory_used < 3.8:
            print(" ⚠ BORDERLINE")
        else:
            print(" ✗ TOO HIGH")
        
        # Cleanup
        del batch, model, output, loss
        torch.cuda.empty_cache()

except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"\n\n✗ OUT OF MEMORY at batch size {batch_size}")
        print(f"Maximum safe batch size: {batch_size - 8}")
    else:
        print(f"\n\nError: {e}")

print("\n" + "="*70)
print("RECOMMENDATION:")
if torch.cuda.get_device_properties(0).total_memory < 5e9:
    print("  Use BATCH_SIZE = 8 for your 4GB GPU")
else:
    print("  Use BATCH_SIZE = 16 or 32")
print("="*70)