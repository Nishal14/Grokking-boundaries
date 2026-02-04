"""
V3 training script with overlapping constraint-conflict split.

EXPERIMENT: v3_overlap_conflict
    - Training: (a + b) mod 97 ∈ [0, 62]
    - Test:     (a + b) mod 97 ∈ [32, 96]
    - Overlap:  [32, 62] (~32% overlap)
    - Train-only: [0, 31]
    - Test-only:  [63, 96]

DESIGN RATIONALE:
    v2b failed due to pure extrapolation (disjoint result ranges).
    v3 introduces constraint conflict while maintaining interpolation capability.
    The overlap ensures the task remains solvable via interpolation,
    while the partial disjoint region creates conflict that delays grokking.

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
    python scripts/train_v3.py
"""
import os
import sys
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset_overlap import create_overlap_dataloaders
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
    """Main training pipeline for v3."""
    # ========================================================================
    # CRITICAL: Check CUDA availability (fails if not available)
    # ========================================================================
    device = check_cuda_available()

    # Configuration (v3 modifications)
    config = Config()

    # V3 MODIFICATION 1: Use overlapping split (hardcoded in dataset)
    # Train: [0, 62], Test: [32, 96], Overlap: [32, 62]

    # V3 MODIFICATION 2: Training duration
    # SANITY CHECK: 20k steps first
    # FULL TRAINING: 400k steps (update after sanity passes)
    config.training.num_steps = 20_000  # SANITY CHECK (update to 400k if passes)

    # Set random seed
    set_seed(config.data.seed)

    # Output directory (v3)
    output_dir = 'experiments/modular_arithmetic/v3_overlap_conflict'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'version': 'v3_overlap_conflict',
            'split_type': 'overlapping_output_based',
            'split_rule': 'Train [0,62], Test [32,96], Overlap [32,62]',
            'design_rationale': 'Constraint conflict with interpolation capability',
            'current_phase': 'sanity_check' if config.training.num_steps <= 20000 else 'full_training',
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'analysis': config.analysis.__dict__
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # Create datasets with OVERLAPPING split
    print("Creating datasets with OVERLAPPING constraint-conflict split...")
    print("  Train: (a + b) mod 97 in [0, 62]")
    print("  Test:  (a + b) mod 97 in [32, 96]")
    print("  Overlap region: [32, 62]")
    print()

    train_loader, test_loader = create_overlap_dataloaders(
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

    # Initialize model on GPU (identical to v1/v2)
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
        print("V3 SANITY CHECK (MANDATORY BEFORE FULL TRAINING)")
    else:
        print("V3 FULL TRAINING")
    print("="*70)
    print(f"  Experiment: v3_overlap_conflict")
    print(f"  Total steps: {config.training.num_steps:,}")
    print(f"  Split: Overlapping output-based")
    print(f"    - Train range: [0, 62]")
    print(f"    - Test range: [32, 96]")
    print(f"    - Overlap: [32, 62] (~32%)")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Logging interval: {config.training.log_interval} steps")
    print(f"  Analysis interval: {config.analysis.compute_rank_interval} steps")
    print(f"  Checkpoint interval: {config.training.checkpoint_interval} steps")
    print()

    if config.training.num_steps <= 20000:
        # Sanity check expectations
        print("SANITY CHECK SUCCESS CRITERIA (ALL MUST PASS):")
        print("  1. Test accuracy <= 5% throughout")
        print("  2. No early accuracy ramp (no >10% jump)")
        print("  3. Representation rank >= 20 and flat")
        print("  4. No entropy collapse")
        print()
        print("If sanity check PASSES:")
        print("  -> Update num_steps to 400,000 in train_v3.py")
        print("  -> Run full training")
        print()
        print("If sanity check FAILS:")
        print("  -> STOP immediately")
        print("  -> DO NOT proceed to full training")
        print("  -> Diagnose failure mode")
        print()
    else:
        # Full training expectations
        print("Expected V3 Behavior (Full Training):")
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
        print("Comparison with V1 and V2:")
        print("  V1 grokking: ~17k steps (random 80% split)")
        print("  V2 grokking: FAILED (pure extrapolation too hard)")
        print("  V3 expected: 50k-200k steps (overlap allows interpolation)")
        print("  Delay factor: ~3-12x slower than v1")
        print()

    print("="*70)
    print()

    if config.training.num_steps <= 20000:
        input("Press Enter to start v3 SANITY CHECK (20k steps)...")
    else:
        input("Press Enter to start v3 FULL TRAINING (400k steps)...")
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
        print("V3 SANITY CHECK COMPLETE!")
    else:
        print("V3 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: V3 training configuration")
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
        print("     - Update train_v3.py: num_steps = 400_000")
        print("     - Run full training")
        print()
        print("  3. If ANY criterion fails:")
        print("     - STOP and diagnose failure mode")
        print("     - DO NOT proceed to full training")
        print()
    else:
        print("Next steps:")
        print("  1. Compare v3 to v1 baseline and v2")
        print("  2. Analyze delayed grokking transition timing")
        print("  3. Compare rank/entropy collapse between v1, v2, v3")
        print("  4. Generate comparison plots")
        print()


if __name__ == '__main__':
    main()
