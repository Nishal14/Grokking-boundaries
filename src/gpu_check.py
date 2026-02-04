"""
GPU availability check and enforcement.

This module ensures CUDA GPU is available and raises an error if not.
ALL scripts must call check_cuda_available() at startup.
"""
import torch
import sys


def check_cuda_available():
    """
    Check if CUDA is available and raise error if not.

    This function MUST be called at the start of all training/inference scripts.
    The project requires CUDA GPU execution - no CPU fallbacks are provided.

    Raises:
        RuntimeError: If CUDA is not available

    Returns:
        torch.device: CUDA device object
    """
    if not torch.cuda.is_available():
        print("\n" + "="*70, file=sys.stderr)
        print("ERROR: CUDA is required but not available", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print("\nThis project requires a CUDA-capable GPU.", file=sys.stderr)
        print("No CPU fallback is provided.\n", file=sys.stderr)
        print("Possible issues:", file=sys.stderr)
        print("  1. No NVIDIA GPU detected", file=sys.stderr)
        print("  2. CUDA drivers not installed", file=sys.stderr)
        print("  3. PyTorch installed without CUDA support", file=sys.stderr)
        print("\nTo fix:", file=sys.stderr)
        print("  - Ensure NVIDIA GPU is available", file=sys.stderr)
        print("  - Install CUDA drivers: https://developer.nvidia.com/cuda-downloads", file=sys.stderr)
        print("  - Reinstall PyTorch with CUDA:", file=sys.stderr)
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu118", file=sys.stderr)
        print("\n" + "="*70 + "\n", file=sys.stderr)
        raise RuntimeError("CUDA is required. No GPU detected.")

    # Return CUDA device
    device = torch.device("cuda")

    # Print GPU info
    print(f"[OK] CUDA available: {torch.version.cuda}")
    print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"[OK] Device: {device}")
    print()

    return device


def get_device():
    """
    Get CUDA device. Assumes check_cuda_available() has been called.

    Returns:
        torch.device: CUDA device object
    """
    return torch.device("cuda")
