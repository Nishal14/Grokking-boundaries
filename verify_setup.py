"""
Verification script to ensure the environment is correctly set up.
Run this after installation to verify all components work.

REQUIREMENTS:
    - CUDA-capable GPU required
    - Script will fail if CUDA is not available
"""
import sys
import torch
from config import Config
from src.model import DecoderOnlyTransformer
from src.tokenizer import ModularArithmeticTokenizer
from src.dataset import ModularArithmeticDataset
from src.gpu_check import check_cuda_available

def verify_setup():
    """Run verification tests."""
    print("="*60)
    print("Grokking Detection Setup Verification (GPU Required)")
    print("="*60)
    print()

    # Test 1: Python version
    print(f"[OK] Python version: {sys.version.split()[0]}")

    # Test 2: PyTorch installation
    print(f"[OK] PyTorch version: {torch.__version__}")

    # Test 3: CUDA availability (CRITICAL - must be available)
    try:
        device = check_cuda_available()
        print(f"[OK] CUDA check passed")
    except RuntimeError as e:
        print(f"[FAIL] CUDA check failed: {e}")
        return False

    # Test 4: Configuration
    config = Config()
    print(f"[OK] Configuration loaded")
    print(f"  - Model: {config.model.n_layers} layers, {config.model.d_model} d_model")
    print(f"  - Training: {config.training.num_steps:,} steps")

    # Test 5: Tokenizer
    tokenizer = ModularArithmeticTokenizer(modulus_p=97)
    test_text = "<bos> 5 + 3 mod 97 = 8 <eos>"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text, "Tokenizer encoding/decoding failed"
    print(f"[OK] Tokenizer working (vocab size: {tokenizer.vocab_size})")

    # Test 6: Dataset
    dataset = ModularArithmeticDataset(modulus_p=97, split='train', train_split=0.8, seed=42)
    sample = dataset[0]
    assert 'input_ids' in sample, "Dataset missing input_ids"
    assert 'labels' in sample, "Dataset missing labels"
    print(f"[OK] Dataset working (train size: {len(dataset):,})")

    # Test 7: Model initialization on GPU
    model = DecoderOnlyTransformer(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.n_layers,
        num_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    model = model.to(device)  # Move to GPU
    param_count = model.count_parameters()
    print(f"[OK] Model initialized on GPU ({param_count:,} parameters)")

    # Test 8: Forward pass on GPU
    batch_size = 4
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, 8), device=device)
    logits, attentions = model(input_ids)
    assert logits.shape == (batch_size, 8, config.model.vocab_size), "Logits shape mismatch"
    assert logits.device.type == 'cuda', "Output not on GPU"
    print(f"[OK] Model forward pass successful on GPU")

    # Test 9: Analysis hooks
    assert len(attentions) == config.model.n_layers, "Wrong number of attention layers"
    assert attentions[0].device.type == 'cuda', "Attention not on GPU"
    print(f"[OK] Analysis hooks working (attentions returned on GPU)")

    print()
    print("="*60)
    print("[OK] All verification tests passed!")
    print("="*60)
    print()
    print("Your environment is ready. You can now run:")
    print("  python scripts/train.py")
    print()

if __name__ == '__main__':
    try:
        verify_setup()
    except Exception as e:
        print()
        print("="*60)
        print("[FAIL] Verification failed!")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPlease check SETUP.md for troubleshooting steps.")
        sys.exit(1)
