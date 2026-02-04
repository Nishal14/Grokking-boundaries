"""
V3.1 training script with reduced overlapping constraint-conflict split.

EXPERIMENT: v3_1_reduced_overlap
    - Training: (a + b) mod 97 ∈ [0, 40]
    - Test:     (a + b) mod 97 ∈ [30, 96]
    - Overlap:  [30, 40] (~16% overlap, reduced from v3's 49%)
    - Train-only: [0, 29]
    - Test-only:  [41, 96]

V3 FAILURE ANALYSIS:
    v3 failed sanity check with 47.5% test accuracy at 20k steps.
    Root cause: 49% overlap allowed easy interpolation.

V3.1 DESIGN CORRECTION:
    Reduced overlap from 49% to ~16%.
    This increases constraint conflict while maintaining interpolation capability.
    Test-only region increased from 51% to 84% of test data.

TRAINING CONFIGURATION:
    - First run: 20,000 steps (SANITY CHECK)
    - If sanity passes: Update to 400,000 steps (FULL TRAINING)

SANITY CHECK CRITERIA (ALL MUST PASS):
    1. Test accuracy ≤ 5% throughout 20k steps
    2. No early accuracy ramp (no sudden >10% jump)
    3. Representation rank remains high (≥ 20) and flat
    4. No entropy collapse (no sharp drops)

    If ANY criterion fails:
        → STOP immediately
        → DO NOT proceed to full training
        → Diagnose failure mode

REQUIREMENTS:
    - CUDA-capable GPU required
    - No CPU fallback provided

Usage:
    python scripts/train_v3_1.py
"""
import os
import sys
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset_overlap_v31 import create_overlap_v31_dataloaders
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
    """Main training pipeline for v3.1."""
    # ========================================================================
    # CRITICAL: Check CUDA availability (fails if not available)
    # ========================================================================
    device = check_cuda_available()

    # Configuration (v3.1 modifications)
    config = Config()

    # V3.1 MODIFICATION 1: Use reduced overlapping split (hardcoded in dataset)
    # Train: [0, 40], Test: [30, 96], Overlap: [30, 40] (~16%)

    # V3.1 MODIFICATION 2: Training duration
    # SANITY CHECK: 20k steps first
    # FULL TRAINING: 400k steps (update after sanity passes)
    config.training.num_steps = 20_000  # SANITY CHECK (update to 400k if passes)

    # Set random seed
    set_seed(config.data.seed)

    # Output directory (v3.1)
    output_dir = 'experiments/modular_arithmetic/v3_1_reduced_overlap'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'version': 'v3_1_reduced_overlap',
            'split_type': 'overlapping_output_based_reduced',
            'split_rule': 'Train [0,40], Test [30,96], Overlap [30,40]',
            'design_rationale': 'Reduced overlap from v3 (49%->16%) to increase constraint conflict',
            'v3_failure': 'v3 achieved 47.5% at 20k (too easy to interpolate)',
            'current_phase': 'sanity_check' if config.training.num_steps <= 20000 else 'full_training',
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'analysis': config.analysis.__dict__
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # Create datasets with REDUCED OVERLAPPING split
    print("Creating datasets with REDUCED OVERLAPPING constraint-conflict split...")
    print("  Train: (a + b) mod 97 in [0, 40]")
    print("  Test:  (a + b) mod 97 in [30, 96]")
    print("  Overlap region: [30, 40] (reduced from v3)")
    print()

    train_loader, test_loader = create_overlap_v31_dataloaders(
        modulus_p=config.data.modulus_p,
        batch_size=config.training.batch_size,
        seed=config.data.seed,
        num_workers=0  # Set to 0 on Windows
    )

    print(f"Train set size: {len(train_loader.dataset):,} examples")
    print(f"Test set size: {len(test_loader.dataset):,} examples")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print()

    # Initialize model on GPU (identical to v1/v2/v3)
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
    if config.training.num_steps <= 20000:
        print("V3.1 SANITY CHECK (MANDATORY BEFORE FULL TRAINING)")
    else:
        print("V3.1 FULL TRAINING")
    print("="*70)
    print(f"  Experiment: v3_1_reduced_overlap")
    print(f"  Total steps: {config.training.num_steps:,}")
    print(f"  Split: Reduced overlapping output-based")
    print(f"    - Train range: [0, 40]")
    print(f"    - Test range: [30, 96]")
    print(f"    - Overlap: [30, 40] (~16%, reduced from v3's 49%)")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Logging interval: {config.training.log_interval} steps")
    print(f"  Analysis interval: {config.analysis.compute_rank_interval} steps")
    print(f"  Checkpoint interval: {config.training.checkpoint_interval} steps")
    print()

    if config.training.num_steps <= 20000:
        # Sanity check expectations
        print("V3 FAILURE RECAP:")
        print("  - Overlap: 49% (too large)")
        print("  - Result: 47.5% test accuracy at 20k steps")
        print("  - Failed: Allowed easy interpolation")
        print()
        print("V3.1 IMPROVEMENT:")
        print("  - Overlap: ~16% (reduced)")
        print("  - Train-only: [0, 29] (30 values)")
        print("  - Test-only: [41, 96] (56 values, 84% of test data)")
        print("  - Goal: Increase constraint conflict while maintaining interpolation")
        print()
        print("SANITY CHECK SUCCESS CRITERIA (ALL MUST PASS):")
        print("  1. Test accuracy <= 5% throughout")
        print("  2. No early accuracy ramp (no >10% jump)")
        print("  3. Representation rank >= 20 and flat")
        print("  4. No entropy collapse")
        print()
        print("If sanity check PASSES:")
        print("  -> Update num_steps to 400,000 in train_v3_1.py")
        print("  -> Run full training")
        print()
        print("If sanity check FAILS:")
        print("  -> STOP immediately")
        print("  -> DO NOT proceed to full training")
        print("  -> Diagnose failure mode (may need further split adjustment)")
        print()
    else:
        # Full training expectations
        print("Expected V3.1 Behavior (Full Training):")
        print("  Phase 1: Extended Memorization Plateau")
        print("    - Train loss decreases")
        print("    - Test accuracy: ~0-5% (delayed generalization)")
        print("    - Representation rank: HIGH (~20-40)")
        print("    - Attention entropy: Stable")
        print()
        print("  Phase 2: Delayed Grokking Transition")
        print("    - Test accuracy: SHARP jump 0-5% -> 80-95%")
        print("    - Representation rank: SHARP collapse ~30 -> ~5")
        print("    - Attention entropy: Sharp drop")
        print()
        print("  Phase 3: Generalization")
        print("    - Both losses stabilize")
        print("    - Test accuracy: 85-99%")
        print("    - Low stable rank")
        print()
        print("Comparison with Previous Experiments:")
        print("  V1 grokking: ~17k steps (random 80% split)")
        print("  V2 grokking: FAILED (pure extrapolation, 0% at 400k)")
        print("  V3 grokking: FAILED (49% overlap, 47.5% at 20k)")
        print("  V3.1 expected: 50k-250k steps (16% overlap, balanced conflict)")
        print("  Delay factor: ~3-15x slower than v1")
        print()

    print("="*70)
    print()

    if config.training.num_steps <= 20000:
        input("Press Enter to start v3.1 SANITY CHECK (20k steps)...")
    else:
        input("Press Enter to start v3.1 FULL TRAINING (400k steps)...")
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
    if config.training.num_steps <= 20000:
        print("V3.1 SANITY CHECK COMPLETE!")
    else:
        print("V3.1 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: V3.1 training configuration")
    print(f"  - logs/metrics.json: Training and test metrics")
    print(f"  - logs/analysis.json: Internal signal analysis")
    print(f"  - checkpoints/: Model checkpoints")
    print()

    if config.training.num_steps <= 20000:
        print("NEXT STEPS:")
        print("  1. Analyze sanity check results:")
        print("     - Check test accuracy stayed <= 5%")
        print("     - Check no early accuracy ramp")
        print("     - Check rank remained high (>= 20) and flat")
        print("     - Check no entropy collapse")
        print()
        print("  2. If ALL criteria pass:")
        print("     - Update train_v3_1.py: num_steps = 400_000")
        print("     - Run full training")
        print()
        print("  3. If ANY criterion fails:")
        print("     - STOP and diagnose failure mode")
        print("     - May need further split adjustment")
        print()
    else:
        print("Next steps:")
        print("  1. Compare v3.1 to v1 baseline")
        print("  2. Analyze delayed grokking transition timing")
        print("  3. Compare rank/entropy collapse between v1, v3.1")
        print("  4. Generate comparison plots")
        print()


if __name__ == '__main__':
    main()
