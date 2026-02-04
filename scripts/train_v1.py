"""
V1 Baseline training script for grokking detection experiments.

EXPERIMENT: v1 (Baseline)
    - Training coverage: 80%
    - Test coverage: 20%
    - Training steps: 200,000

REQUIREMENTS:
    - CUDA-capable GPU required
    - No CPU fallback provided
    - Script will fail if CUDA is not available

This script orchestrates the entire pipeline:
1. Check CUDA availability (fails if not available)
2. Load configuration
3. Create datasets
4. Initialize model on GPU
5. Train with integrated analysis
6. Save results

Usage:
    python scripts/train_v1.py
"""
import os
import sys
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset import create_dataloaders
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
    """Main training pipeline."""
    # ========================================================================
    # CRITICAL: Check CUDA availability (fails if not available)
    # ========================================================================
    device = check_cuda_available()

    # Configuration
    config = Config()

    # Set random seed
    set_seed(config.data.seed)

    # Output directory (v1 baseline)
    output_dir = 'experiments/modular_arithmetic/v1'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'analysis': config.analysis.__dict__
        }, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_loader, test_loader = create_dataloaders(
        modulus_p=config.data.modulus_p,
        train_split=config.data.train_split,
        batch_size=config.training.batch_size,
        seed=config.data.seed,
        num_workers=0  # Set to 0 on Windows
    )

    print(f"Train set size: {len(train_loader.dataset):,} examples")
    print(f"Test set size: {len(test_loader.dataset):,} examples")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print()

    # Initialize model on GPU
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
    print("Training Plan:")
    print(f"  Total steps: {config.training.num_steps:,}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Logging interval: {config.training.log_interval} steps")
    print(f"  Analysis interval: {config.analysis.compute_rank_interval} steps")
    print(f"  Checkpoint interval: {config.training.checkpoint_interval} steps")
    print()

    # Expected behavior
    print("Expected Grokking Behavior:")
    print("  Phase 1 (0-50k steps): Memorization")
    print("    - Train loss decreases rapidly")
    print("    - Test accuracy: ~50-70%")
    print("    - High representation rank")
    print()
    print("  Phase 2 (50k-150k steps): Grokking Transition")
    print("    - Train loss plateaus")
    print("    - Test accuracy jumps to ~95-100%")
    print("    - Representation rank drops sharply")
    print()
    print("  Phase 3 (150k-200k steps): Generalization")
    print("    - Both losses stabilize")
    print("    - Test accuracy ~99-100%")
    print("    - Low stable rank")
    print()

    input("Press Enter to start training...")
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
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: Training configuration")
    print(f"  - logs/metrics.json: Training and test metrics")
    print(f"  - logs/analysis.json: Internal signal analysis")
    print(f"  - checkpoints/: Model checkpoints")
    print()
    print("Next steps:")
    print("  1. Visualize results: python scripts/visualize.py")
    print("  2. Analyze grokking transition timing")
    print("  3. Examine correlation between rank drop and accuracy increase")
    print()


if __name__ == '__main__':
    main()
