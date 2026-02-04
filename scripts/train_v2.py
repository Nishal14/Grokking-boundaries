"""
V2 training script with structured sum-based split.

EXPERIMENT: v2 (Conflict-Based Split - Full Training)
    - Training: (a + b) mod p < p/2  (results in [0, 47])
    - Test:     (a + b) mod p >= p/2 (results in [48, 96])
    - Training steps: 400,000

VALIDATED BY SANITY CHECK:
    v2b_conflict_split (20k steps) confirmed:
    - Test accuracy stayed at ~0% (no interpolation)
    - Rank remained high ~30 (no premature compression)
    - Grokking successfully delayed vs v1 baseline

REQUIREMENTS:
    - CUDA-capable GPU required
    - No CPU fallback provided
    - Script will fail if CUDA is not available

This script orchestrates the entire pipeline:
1. Check CUDA availability (fails if not available)
2. Load configuration
3. Create datasets with STRUCTURED split
4. Initialize model on GPU
5. Train with integrated analysis
6. Save results

Usage:
    python scripts/train_v2.py
"""
import os
import sys
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset_structured import create_structured_dataloaders
from src.model import DecoderOnlyTransformer
from src.training import Trainer
from src.gpu_check import check_cuda_available


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def main():
    """Main training pipeline for v2."""
    # ========================================================================
    # CRITICAL: Check CUDA availability (fails if not available)
    # ========================================================================
    device = check_cuda_available()

    # Configuration (v2 modifications)
    config = Config()

    # V2 MODIFICATION 1: Use structured split (hardcoded in dataset)
    # No train_split parameter - split is determined by sum-based rule

    # V2 MODIFICATION 2: Extended training duration
    config.training.num_steps = 400_000  # 400k steps (vs v1: 200k)

    # Set random seed
    set_seed(config.data.seed)

    # Output directory (v2)
    output_dir = 'experiments/modular_arithmetic/v2'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'version': 'v2_conflict_split',
            'split_type': 'structured_sum_based',
            'split_rule': '(a+b)%p < p/2 (train) vs >= p/2 (test)',
            'sanity_check': 'Passed at 20k steps (see v2b_conflict_split/)',
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'analysis': config.analysis.__dict__
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # Create datasets with STRUCTURED split
    print("Creating datasets with STRUCTURED sum-based split...")
    print("  Train: (a + b) mod 97 < 48  (results in [0, 47])")
    print("  Test:  (a + b) mod 97 >= 48 (results in [48, 96])")
    print()

    train_loader, test_loader = create_structured_dataloaders(
        modulus_p=config.data.modulus_p,
        batch_size=config.training.batch_size,
        seed=config.data.seed,
        num_workers=0  # Set to 0 on Windows
    )

    print(f"Train set size: {len(train_loader.dataset):,} examples")
    print(f"Test set size: {len(test_loader.dataset):,} examples")
    print(f"Train/Test split: {len(train_loader.dataset)/9409*100:.1f}% / {len(test_loader.dataset)/9409*100:.1f}%")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print()

    # Initialize model on GPU (identical to v1)
    print("Initializing model on GPU...")
    model = DecoderOnlyTransformer(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.n_layers,
        num_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    model = model.to(device)  # Move model to GPU

    total_params = model.count_parameters()
    print(f"Model architecture:")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  Heads: {config.model.n_heads}")
    print(f"  d_model: {config.model.d_model}")
    print(f"  d_ff: {config.model.d_ff}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    print()

    # Verify parameter count is close to target (~2M)
    target_params = 2_000_000
    param_ratio = total_params / target_params
    if 0.8 <= param_ratio <= 1.2:
        print(f"[OK] Parameter count is within target range (+/-20% of {target_params:,})")
    else:
        print(f"[WARNING] Parameter count deviates from target {target_params:,}")
    print()

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        training_config=config.training,
        analysis_config=config.analysis,
        device=device,
        output_dir=output_dir
    )
    print()

    # Print training plan
    print("="*70)
    print("V2 TRAINING PLAN (FULL RUN)")
    print("="*70)
    print(f"  Experiment: v2 (Conflict-Based Split)")
    print(f"  Total steps: {config.training.num_steps:,}")
    print(f"  Split: Structured sum-based (disjoint result ranges)")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Logging interval: {config.training.log_interval} steps")
    print(f"  Analysis interval: {config.analysis.compute_rank_interval} steps")
    print(f"  Checkpoint interval: {config.training.checkpoint_interval} steps")
    print()

    # Expected behavior for v2
    print("Expected V2 Behavior (Based on Sanity Check):")
    print("  Phase 1 (0-100k steps): Extended Memorization Plateau")
    print("    - Train loss decreases")
    print("    - Test accuracy: ~0-5% (no early generalization)")
    print("    - Representation rank: HIGH (~20-30)")
    print("    - Attention entropy: LOW but stable")
    print()
    print("  Phase 2 (100k-300k steps): Delayed Grokking Transition")
    print("    - Train loss plateaus")
    print("    - Test accuracy: SHARP jump 0% -> 90%+")
    print("    - Representation rank: SHARP collapse ~30 -> ~5")
    print("    - Attention entropy: Further compression")
    print()
    print("  Phase 3 (300k-400k steps): Generalization")
    print("    - Both losses stabilize")
    print("    - Test accuracy: 90-99%")
    print("    - Low stable rank")
    print()
    print("Comparison with V1:")
    print("  V1 grokking: ~33.7k steps (random 80% split)")
    print("  V2 expected: 100k-300k steps (structured split)")
    print("  Delay factor: ~3-10x slower grokking")
    print("="*70)
    print()

    input("Press Enter to start v2 full training (400k steps)...")
    print()

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.metrics_logger.data['step'][-1] if trainer.metrics_logger.data['step'] else 0)
        trainer.save_metrics()
        trainer.save_analysis_metrics()
        print("Checkpoint saved.")
        sys.exit(0)

    # Final summary
    print("\n" + "="*70)
    print("V2 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: V2 training configuration")
    print(f"  - logs/metrics.json: Training and test metrics")
    print(f"  - logs/analysis.json: Internal signal analysis")
    print(f"  - checkpoints/: Model checkpoints")
    print()
    print("Next steps:")
    print("  1. Compare v2 to v1 baseline")
    print("  2. Analyze delayed grokking transition timing")
    print("  3. Compare rank/entropy collapse between v1 and v2")
    print("  4. Generate comparison plots")
    print()
    print("Sanity check results preserved in:")
    print("  experiments/modular_arithmetic/v2b_conflict_split/")
    print()


if __name__ == '__main__':
    main()
