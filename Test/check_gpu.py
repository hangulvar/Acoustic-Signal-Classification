# check_gpu.py
import torch
import sys
import torch.version

print("="*70)
print("GPU DIAGNOSTIC CHECK")
print("="*70)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}") 
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Test GPU computation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("\n✓ GPU computation test PASSED")
    except Exception as e:
        print(f"\n✗ GPU computation test FAILED: {e}")
else:
    print("\n⚠ CUDA NOT AVAILABLE - PyTorch will use CPU only")
    print("\nPossible reasons:")
    print("  1. PyTorch CPU-only version installed")
    print("  2. NVIDIA drivers not installed")
    print("  3. CUDA Toolkit not installed")

print("="*70)