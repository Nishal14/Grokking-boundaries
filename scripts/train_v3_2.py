"""
V3.2 training script with MINIMAL overlapping constraint-conflict split.

EXPERIMENT: v3_2_minimal_overlap (FINAL ATTEMPT)
    - Training: (a + b) mod 97 ∈ [0, 25]
    - Test:     (a + b) mod 97 ∈ [20, 96]
    - Overlap:  [20, 25] (6 values, ~8% of test set)
    - Train-only: [0, 19]
    - Test-only:  [26, 96]

DESIGN PROGRESSION:
    v2:   0% overlap   -> 0.00% accuracy at 400k (pure extrapolation impossible)
    v3:   49% overlap  -> 47.5% accuracy at 20k (too easy to interpolate)
    v3.1: 27% overlap  -> 16.4% accuracy at 20k (still too easy)
    v3.2: 8% overlap   -> ??? (minimal viable overlap)

RATIONALE:
    This is the final attempt to find the balance between:
    - Pure extrapolation (impossible to learn)
    - Easy interpolation (no delayed grokking)

    8% overlap represents the minimum viable overlap that still allows
    some interpolation capability while maximizing constraint conflict.

TRAINING CONFIGURATION:
    - First run: 20,000 steps (SANITY CHECK)
    - If sanity passes: Update to 400,000 steps (FULL TRAINING)

SANITY CHECK CRITERIA (ALL MUST PASS):
    1. Test accuracy ≤ 5% throughout 20k steps
    2. No early accuracy ramp (no sudden >10% jump)
    3. Representation rank remains high (≥ 20) and flat
    4. No entropy collapse (no sharp drops)

    If ANY criterion fails:
        → Accept experimental findings
        → V2/V3/V3.1/V3.2 demonstrate feasibility boundaries
        → Conclude with V1 as successful baseline

REQUIREMENTS:
    - CUDA-capable GPU required
    - No CPU fallback provided

Usage:
    python scripts/train_v3_2.py
"""
import os
import sys
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset_overlap_v32 import create_overlap_v32_dataloaders
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
    """Main training pipeline for v3.2."""
    # ========================================================================
    # CRITICAL: Check CUDA availability (fails if not available)
    # ========================================================================
    device = check_cuda_available()

    # Configuration (v3.2 modifications)
    config = Config()

    # V3.2 MODIFICATION 1: Use minimal overlapping split (hardcoded in dataset)
    # Train: [0, 25], Test: [20, 96], Overlap: [20, 25] (~8% of test)

    # V3.2 MODIFICATION 2: Training duration
    # SANITY CHECK: 20k steps first
    # FULL TRAINING: 400k steps (update after sanity passes)
    config.training.num_steps = 20_000  # SANITY CHECK (update to 400k if passes)

    # Set random seed
    set_seed(config.data.seed)

    # Output directory (v3.2)
    output_dir = 'experiments/modular_arithmetic/v3_2_minimal_overlap'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'version': 'v3_2_minimal_overlap',
            'split_type': 'overlapping_output_based_minimal',
            'split_rule': 'Train [0,25], Test [20,96], Overlap [20,25]',
            'design_rationale': 'Final minimal overlap attempt (8% of test) after v3.1 failure',
            'v3_failure': 'v3 achieved 47.5% at 20k (49% overlap, too easy)',
            'v3_1_failure': 'v3.1 achieved 16.4% at 20k (27% overlap, still too easy)',
            'current_phase': 'sanity_check' if config.training.num_steps <= 20000 else 'full_training',
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'analysis': config.analysis.__dict__
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # Create datasets with MINIMAL OVERLAPPING split
    print("Creating datasets with MINIMAL OVERLAPPING constraint-conflict split...")
    print("  Train: (a + b) mod 97 in [0, 25]")
    print("  Test:  (a + b) mod 97 in [20, 96]")
    print("  Overlap region: [20, 25] (minimal)")
    print()

    train_loader, test_loader = create_overlap_v32_dataloaders(
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

    # Initialize model on GPU (identical to v1/v2/v3/v3.1)
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
        print("V3.2 SANITY CHECK (FINAL ATTEMPT)")
    else:
        print("V3.2 FULL TRAINING")
    print("="*70)
    print(f"  Experiment: v3_2_minimal_overlap")
    print(f"  Total steps: {config.training.num_steps:,}")
    print(f"  Split: MINIMAL overlapping output-based")
    print(f"    - Train range: [0, 25]")
    print(f"    - Test range: [20, 96]")
    print(f"    - Overlap: [20, 25] (~8% of test, minimized)")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Logging interval: {config.training.log_interval} steps")
    print(f"  Analysis interval: {config.analysis.compute_rank_interval} steps")
    print(f"  Checkpoint interval: {config.training.checkpoint_interval} steps")
    print()

    if config.training.num_steps <= 20000:
        # Sanity check expectations
        print("EXPERIMENTAL PROGRESSION:")
        print("  v2:   0% overlap   -> 0.00% at 400k (impossible)")
        print("  v3:   49% overlap  -> 47.5% at 20k (too easy)")
        print("  v3.1: 27% overlap  -> 16.4% at 20k (still too easy)")
        print("  v3.2: 8% overlap   -> ??? (final minimal attempt)")
        print()
        print("V3.2 DESIGN:")
        print("  - Overlap: 6 values (8% of test set)")
        print("  - Train-only: [0, 19] (20 values, 77% of train)")
        print("  - Test-only: [26, 96] (71 values, 92% of test)")
        print("  - Goal: Maximum constraint conflict with minimal interpolation")
        print()
        print("SANITY CHECK SUCCESS CRITERIA (ALL MUST PASS):")
        print("  1. Test accuracy <= 5% throughout")
        print("  2. No early accuracy ramp (no >10% jump)")
        print("  3. Representation rank >= 20 and flat")
        print("  4. No entropy collapse")
        print()
        print("If sanity check PASSES:")
        print("  -> Update num_steps to 400,000 in train_v3_2.py")
        print("  -> Run full training")
        print("  -> Expect delayed grokking at 50k-250k steps")
        print()
        print("If sanity check FAILS:")
        print("  -> Accept that overlap-based delay is infeasible")
        print("  -> V1 provides successful baseline")
        print("  -> V2/V3/V3.1/V3.2 demonstrate feasibility boundaries")
        print("  -> Conclude experiments and generate final plots")
        print()
    else:
        # Full training expectations
        print("Expected V3.2 Behavior (Full Training):")
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
        print("Comparison:")
        print("  V1:   ~17k steps (random 80%)")
        print("  V3.2: 50k-250k steps expected (8% overlap)")
        print("  Delay factor: ~3-15x")
        print()

    print("="*70)
    print()

    if config.training.num_steps <= 20000:
        input("Press Enter to start v3.2 SANITY CHECK (FINAL ATTEMPT, 20k steps)...")
    else:
        input("Press Enter to start v3.2 FULL TRAINING (400k steps)...")
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
        print("V3.2 SANITY CHECK COMPLETE!")
    else:
        print("V3.2 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: V3.2 training configuration")
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
        print("     - Update train_v3_2.py: num_steps = 400_000")
        print("     - Run full training")
        print()
        print("  3. If ANY criterion fails:")
        print("     - Accept experimental findings")
        print("     - Conclude with V1 baseline + boundary demonstrations")
        print("     - Generate final comparison plots")
        print()
    else:
        print("Next steps:")
        print("  1. Compare v3.2 to v1 baseline")
        print("  2. Analyze delayed grokking transition timing")
        print("  3. Generate comprehensive comparison plots")
        print()


if __name__ == '__main__':
    main()
