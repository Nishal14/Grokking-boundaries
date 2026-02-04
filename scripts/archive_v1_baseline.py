"""
Archive and visualize v1 baseline experiment results.

This script:
1. Archives baseline results into v1_baseline directory
2. Generates plots showing grokking transition
3. Creates a summary of key metrics

IMPORTANT: Only operates on existing data. Does not retrain or modify experiments.

Usage:
    python scripts/archive_v1_baseline.py
"""
import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_baseline_archive(source_dir: str, baseline_dir: str):
    """
    Create v1 baseline archive by copying results.

    Args:
        source_dir: Source experiment directory
        baseline_dir: Target baseline directory
    """
    print("="*70)
    print("ARCHIVING V1 BASELINE RESULTS")
    print("="*70)
    print()

    source_path = Path(source_dir)
    baseline_path = Path(baseline_dir)

    # Create baseline directory
    baseline_path.mkdir(parents=True, exist_ok=True)
    print(f"Created baseline directory: {baseline_path}")

    # Copy logs
    source_logs = source_path / 'logs'
    target_logs = baseline_path / 'logs'
    if source_logs.exists():
        if target_logs.exists():
            shutil.rmtree(target_logs)
        shutil.copytree(source_logs, target_logs)
        print(f"✓ Copied logs to {target_logs}")

    # Copy post_analysis.json if exists
    source_analysis = source_path / 'post_analysis.json'
    target_analysis = baseline_path / 'post_analysis.json'
    if source_analysis.exists():
        shutil.copy2(source_analysis, target_analysis)
        print(f"✓ Copied post_analysis.json")

    # Copy config.json
    source_config = source_path / 'config.json'
    target_config = baseline_path / 'config.json'
    if source_config.exists():
        shutil.copy2(source_config, target_config)
        print(f"✓ Copied config.json")

    # Copy checkpoints (or create symlinks to save space)
    source_checkpoints = source_path / 'checkpoints'
    target_checkpoints = baseline_path / 'checkpoints'
    if source_checkpoints.exists():
        if not target_checkpoints.exists():
            # Create directory with a README instead of copying all checkpoints
            target_checkpoints.mkdir(parents=True, exist_ok=True)
            readme_path = target_checkpoints / 'README.txt'
            with open(readme_path, 'w') as f:
                f.write(f"Checkpoints are stored in: {source_checkpoints.absolute()}\n")
                f.write(f"To access checkpoints, use the original directory.\n")
            print(f"✓ Created checkpoint reference at {target_checkpoints}")
            print(f"  (Checkpoints remain in original location to save space)")

    print()
    print("Archive created successfully!")
    print()


def load_data(baseline_dir: str):
    """
    Load all necessary data for plotting.

    Args:
        baseline_dir: Baseline directory path

    Returns:
        Dictionary containing all loaded data
    """
    baseline_path = Path(baseline_dir)

    # Load training metrics
    metrics_path = baseline_path / 'logs' / 'metrics.json'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load post-analysis if available
    analysis_path = baseline_path / 'post_analysis.json'
    analysis = None
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)

    # Load analysis metrics if available
    analysis_metrics_path = baseline_path / 'logs' / 'analysis.json'
    analysis_metrics = None
    if analysis_metrics_path.exists():
        with open(analysis_metrics_path, 'r') as f:
            analysis_metrics = json.load(f)

    return {
        'metrics': metrics,
        'analysis': analysis,
        'analysis_metrics': analysis_metrics
    }


def create_plots(data: dict, baseline_dir: str, t_grok: int):
    """
    Generate plots showing grokking transition.

    Args:
        data: Loaded data dictionary
        baseline_dir: Directory to save plots
        t_grok: Grokking transition step
    """
    print("="*70)
    print("GENERATING V1 BASELINE PLOTS")
    print("="*70)
    print()

    baseline_path = Path(baseline_dir)
    plots_dir = baseline_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    metrics = data['metrics']
    analysis_metrics = data['analysis_metrics']

    # Convert to numpy arrays
    steps = np.array(metrics['step'])
    test_accuracy = np.array(metrics['test_accuracy'])
    train_loss = np.array(metrics['train_loss'])
    test_loss = np.array(metrics['test_loss'])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('V1 Baseline: Grokking Transition Analysis', fontsize=16, fontweight='bold')

    # Plot A: Test Accuracy
    ax1 = axes[0]
    ax1.plot(steps, test_accuracy, 'b-', linewidth=2, label='Test Accuracy')
    ax1.axvline(t_grok, color='r', linestyle='--', linewidth=2, label=f't_grok = {t_grok}')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Training Step', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])

    # Plot B: Representation Rank (if available)
    ax2 = axes[1]
    if analysis_metrics and 'step' in analysis_metrics:
        analysis_steps = np.array(analysis_metrics['step'])
        # Get final layer rank
        final_layer_ranks = []
        for rank_list in analysis_metrics['representation_rank']:
            if isinstance(rank_list, list) and len(rank_list) > 0:
                final_layer_ranks.append(rank_list[-1])  # Last layer
            else:
                final_layer_ranks.append(np.nan)

        final_layer_ranks = np.array(final_layer_ranks)

        ax2.plot(analysis_steps, final_layer_ranks, 'g-', linewidth=2, label='Final Layer Rank')
        ax2.axvline(t_grok, color='r', linestyle='--', linewidth=2, label=f't_grok = {t_grok}')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Effective Rank', fontsize=12)
        ax2.set_title('Representation Rank vs Training Step', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Analysis metrics not available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Effective Rank', fontsize=12)
        ax2.set_title('Representation Rank vs Training Step', fontsize=13, fontweight='bold')

    # Plot C: Attention Entropy (if available)
    ax3 = axes[2]
    if analysis_metrics and 'step' in analysis_metrics:
        analysis_steps = np.array(analysis_metrics['step'])
        # Compute mean entropy across layers
        mean_entropies = []
        for entropy_list in analysis_metrics['attention_entropy']:
            if isinstance(entropy_list, list) and len(entropy_list) > 0:
                # Average across all layers and heads
                flat_entropies = []
                for layer_entropy in entropy_list:
                    if isinstance(layer_entropy, (list, np.ndarray)):
                        flat_entropies.extend(layer_entropy)
                    else:
                        flat_entropies.append(layer_entropy)
                mean_entropies.append(np.mean(flat_entropies))
            else:
                mean_entropies.append(np.nan)

        mean_entropies = np.array(mean_entropies)

        ax3.plot(analysis_steps, mean_entropies, 'm-', linewidth=2, label='Mean Attention Entropy')
        ax3.axvline(t_grok, color='r', linestyle='--', linewidth=2, label=f't_grok = {t_grok}')
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel('Entropy (nats)', fontsize=12)
        ax3.set_title('Attention Entropy vs Training Step', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Analysis metrics not available',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel('Entropy (nats)', fontsize=12)
        ax3.set_title('Attention Entropy vs Training Step', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save combined plot
    combined_path = plots_dir / 'v1_baseline_grokking_analysis.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {combined_path}")

    combined_path_pdf = plots_dir / 'v1_baseline_grokking_analysis.pdf'
    plt.savefig(combined_path_pdf, bbox_inches='tight')
    print(f"✓ Saved combined plot (PDF): {combined_path_pdf}")

    plt.close()

    # Create individual plots

    # Individual Plot A: Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, test_accuracy, 'b-', linewidth=2, label='Test Accuracy')
    ax.axvline(t_grok, color='r', linestyle='--', linewidth=2, label=f't_grok = {t_grok}')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('V1 Baseline: Test Accuracy vs Training Step', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    acc_path = plots_dir / 'v1_accuracy.png'
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy plot: {acc_path}")
    plt.close()

    # Individual Plot B: Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_loss, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    ax.plot(steps, test_loss, 'orange', linewidth=2, label='Test Loss')
    ax.axvline(t_grok, color='r', linestyle='--', linewidth=2, label=f't_grok = {t_grok}')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('V1 Baseline: Loss Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    loss_path = plots_dir / 'v1_loss_curves.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss curves plot: {loss_path}")
    plt.close()

    print()
    print("All plots generated successfully!")
    print()


def generate_summary(data: dict, baseline_dir: str, t_grok: int):
    """
    Generate summary of key metrics before/after grokking.

    Args:
        data: Loaded data dictionary
        baseline_dir: Directory to save summary
        t_grok: Grokking transition step
    """
    print("="*70)
    print("GENERATING V1 BASELINE SUMMARY")
    print("="*70)
    print()

    baseline_path = Path(baseline_dir)
    metrics = data['metrics']
    analysis = data['analysis']
    analysis_metrics = data['analysis_metrics']

    # Find metrics before and after t_grok
    steps = np.array(metrics['step'])
    test_accuracy = np.array(metrics['test_accuracy'])

    # Before grokking: average of steps < t_grok - 5000
    before_mask = steps < (t_grok - 5000)
    if np.any(before_mask):
        acc_before = np.mean(test_accuracy[before_mask])
    else:
        acc_before = test_accuracy[0]

    # After grokking: average of steps > t_grok + 5000
    after_mask = steps > (t_grok + 5000)
    if np.any(after_mask):
        acc_after = np.mean(test_accuracy[after_mask])
    else:
        acc_after = test_accuracy[-1]

    # Get rank and entropy if available
    rank_before = None
    rank_after = None
    entropy_before = None
    entropy_after = None

    if analysis_metrics and 'step' in analysis_metrics:
        analysis_steps = np.array(analysis_metrics['step'])

        # Extract final layer ranks
        final_layer_ranks = []
        for rank_list in analysis_metrics['representation_rank']:
            if isinstance(rank_list, list) and len(rank_list) > 0:
                final_layer_ranks.append(rank_list[-1])
            else:
                final_layer_ranks.append(np.nan)
        final_layer_ranks = np.array(final_layer_ranks)

        # Extract mean entropies
        mean_entropies = []
        for entropy_list in analysis_metrics['attention_entropy']:
            if isinstance(entropy_list, list) and len(entropy_list) > 0:
                flat_entropies = []
                for layer_entropy in entropy_list:
                    if isinstance(layer_entropy, (list, np.ndarray)):
                        flat_entropies.extend(layer_entropy)
                    else:
                        flat_entropies.append(layer_entropy)
                mean_entropies.append(np.mean(flat_entropies))
            else:
                mean_entropies.append(np.nan)
        mean_entropies = np.array(mean_entropies)

        # Before grokking
        before_analysis_mask = analysis_steps < (t_grok - 5000)
        if np.any(before_analysis_mask):
            rank_before = float(np.mean(final_layer_ranks[before_analysis_mask]))
            entropy_before = float(np.mean(mean_entropies[before_analysis_mask]))

        # After grokking
        after_analysis_mask = analysis_steps > (t_grok + 5000)
        if np.any(after_analysis_mask):
            rank_after = float(np.mean(final_layer_ranks[after_analysis_mask]))
            entropy_after = float(np.mean(mean_entropies[after_analysis_mask]))

    # Create summary
    summary = {
        'version': 'v1_baseline',
        't_grok': int(t_grok),
        'accuracy': {
            'before_grokking': float(acc_before),
            'after_grokking': float(acc_after),
            'change': float(acc_after - acc_before)
        },
        'representation_rank': {
            'before_grokking': rank_before,
            'after_grokking': rank_after,
            'change': float(rank_after - rank_before) if (rank_before is not None and rank_after is not None) else None
        },
        'attention_entropy': {
            'before_grokking': entropy_before,
            'after_grokking': entropy_after,
            'change': float(entropy_after - entropy_before) if (entropy_before is not None and entropy_after is not None) else None
        },
        'total_steps': int(steps[-1]),
        'note': 'Metrics computed as averages over windows: before = steps < t_grok - 5000, after = steps > t_grok + 5000'
    }

    # Save summary
    summary_path = baseline_path / 'v1_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary: {summary_path}")
    print()

    # Print summary to console
    print("V1 BASELINE SUMMARY")
    print("-" * 70)
    print(f"Grokking Transition: t_grok = {t_grok}")
    print()
    print("Test Accuracy:")
    print(f"  Before grokking: {acc_before:.4f}")
    print(f"  After grokking:  {acc_after:.4f}")
    print(f"  Change:          {acc_after - acc_before:+.4f}")
    print()
    if rank_before is not None:
        print("Representation Rank:")
        print(f"  Before grokking: {rank_before:.2f}")
        print(f"  After grokking:  {rank_after:.2f}")
        print(f"  Change:          {rank_after - rank_before:+.2f}")
        print()
    if entropy_before is not None:
        print("Attention Entropy:")
        print(f"  Before grokking: {entropy_before:.4f}")
        print(f"  After grokking:  {entropy_after:.4f}")
        print(f"  Change:          {entropy_after - entropy_before:+.4f}")
        print()
    print("-" * 70)
    print()


def main():
    """
    Main entry point for archiving v1 baseline.
    """
    # Configuration
    source_dir = 'experiments/modular_arithmetic'
    baseline_dir = 'experiments/modular_arithmetic/v1_baseline'

    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    # Step 1: Create archive
    create_baseline_archive(source_dir, baseline_dir)

    # Step 2: Load data
    print("Loading data...")
    data = load_data(baseline_dir)
    print("✓ Data loaded")
    print()

    # Get t_grok
    if data['analysis'] and 't_grok' in data['analysis']:
        t_grok = data['analysis']['t_grok']
    else:
        # Detect t_grok from metrics if not in analysis
        print("Warning: t_grok not found in analysis. Detecting from metrics...")
        steps = np.array(data['metrics']['step'])
        accuracies = np.array(data['metrics']['test_accuracy'])

        # Simple detection: find largest jump
        derivatives = np.diff(accuracies)
        max_idx = np.argmax(derivatives)
        t_grok = int(steps[max_idx + 1])
        print(f"Detected t_grok = {t_grok}")
        print()

    # Step 3: Create plots
    create_plots(data, baseline_dir, t_grok)

    # Step 4: Generate summary
    generate_summary(data, baseline_dir, t_grok)

    print("="*70)
    print("V1 BASELINE ARCHIVING COMPLETE")
    print("="*70)
    print()
    print(f"All results saved to: {baseline_dir}")
    print()
    print("Generated files:")
    print(f"  - logs/                          (training logs)")
    print(f"  - post_analysis.json             (analysis results)")
    print(f"  - plots/v1_baseline_*.png/pdf    (visualization)")
    print(f"  - v1_summary.json                (key metrics)")
    print()


if __name__ == '__main__':
    main()
