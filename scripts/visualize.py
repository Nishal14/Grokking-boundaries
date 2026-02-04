"""
Visualization script for grokking experiment results.

Generates publication-quality plots:
1. Loss curves (train/test)
2. Test accuracy over time
3. Representation rank evolution
4. Attention entropy evolution
5. Combined view (loss + accuracy + rank aligned)

Usage:
    python scripts/visualize.py
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_metrics(experiment_dir: str):
    """Load metrics and analysis data."""
    metrics_path = os.path.join(experiment_dir, 'logs', 'metrics.json')
    analysis_path = os.path.join(experiment_dir, 'logs', 'analysis.json')

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    with open(analysis_path, 'r') as f:
        analysis = json.load(f)

    return metrics, analysis


def plot_loss_curves(metrics: dict, output_dir: str):
    """Plot training and test loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = metrics['step']
    train_loss = metrics['train_loss']
    test_loss = metrics['test_loss']

    ax.plot(steps, train_loss, label='Train Loss', alpha=0.7, linewidth=2)
    ax.plot(steps, test_loss, label='Test Loss', alpha=0.7, linewidth=2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'))
    plt.close()

    print(f"✓ Loss curves saved")


def plot_accuracy(metrics: dict, output_dir: str):
    """Plot test accuracy over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = metrics['step']
    test_acc = metrics['test_accuracy']

    ax.plot(steps, test_acc, color='green', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect accuracy')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title('Test Accuracy (Exact Match)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_accuracy.png'))
    plt.savefig(os.path.join(output_dir, 'test_accuracy.pdf'))
    plt.close()

    print(f"✓ Test accuracy plot saved")


def plot_representation_rank(analysis: dict, output_dir: str):
    """Plot representation rank evolution by layer."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = analysis['step']
    ranks = np.array(analysis['representation_rank'])  # Shape: (num_steps, n_layers)

    n_layers = ranks.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for layer_idx in range(n_layers):
        ax.plot(
            steps,
            ranks[:, layer_idx],
            label=f'Layer {layer_idx + 1}',
            alpha=0.8,
            linewidth=2,
            color=colors[layer_idx]
        )

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Representation Rank Evolution by Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'representation_rank.png'))
    plt.savefig(os.path.join(output_dir, 'representation_rank.pdf'))
    plt.close()

    print(f"✓ Representation rank plot saved")


def plot_attention_entropy(analysis: dict, output_dir: str):
    """Plot attention entropy evolution by layer."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = analysis['step']
    entropies = analysis['attention_entropy']  # List of lists: [step][layer][head]

    # Average entropy across heads for each layer
    n_layers = len(entropies[0])
    colors = plt.cm.plasma(np.linspace(0, 1, n_layers))

    for layer_idx in range(n_layers):
        layer_entropies = [
            np.mean(entropies[step_idx][layer_idx])
            for step_idx in range(len(steps))
        ]
        ax.plot(
            steps,
            layer_entropies,
            label=f'Layer {layer_idx + 1}',
            alpha=0.8,
            linewidth=2,
            color=colors[layer_idx]
        )

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Average Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy Evolution by Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_entropy.png'))
    plt.savefig(os.path.join(output_dir, 'attention_entropy.pdf'))
    plt.close()

    print(f"✓ Attention entropy plot saved")


def plot_combined_view(metrics: dict, analysis: dict, output_dir: str):
    """
    Combined plot showing loss, accuracy, and rank on aligned x-axis.
    This view highlights the correlation between internal signals and grokking.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Loss curves
    steps_metrics = metrics['step']
    train_loss = metrics['train_loss']
    test_loss = metrics['test_loss']

    axes[0].plot(steps_metrics, train_loss, label='Train Loss', alpha=0.7, linewidth=2)
    axes[0].plot(steps_metrics, test_loss, label='Test Loss', alpha=0.7, linewidth=2)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Combined View: Loss, Accuracy, and Internal Signals',
                     fontsize=14, fontweight='bold')

    # 2. Test accuracy
    test_acc = metrics['test_accuracy']
    axes[1].plot(steps_metrics, test_acc, color='green', linewidth=2)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Test Accuracy', fontsize=11)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)

    # 3. Final layer representation rank
    steps_analysis = analysis['step']
    ranks = np.array(analysis['representation_rank'])
    final_layer_rank = ranks[:, -1]  # Last layer

    axes[2].plot(steps_analysis, final_layer_rank, color='purple', linewidth=2)
    axes[2].set_ylabel('Final Layer Rank', fontsize=11)
    axes[2].set_xlabel('Training Steps', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_view.png'))
    plt.savefig(os.path.join(output_dir, 'combined_view.pdf'))
    plt.close()

    print(f"✓ Combined view plot saved")


def plot_grokking_transition_analysis(metrics: dict, analysis: dict, output_dir: str):
    """
    Analyze and visualize the grokking transition point.
    Identifies when test accuracy crosses threshold and when rank drops.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    steps_metrics = metrics['step']
    test_acc = metrics['test_accuracy']

    steps_analysis = analysis['step']
    ranks = np.array(analysis['representation_rank'])
    final_layer_rank = ranks[:, -1]

    # Plot accuracy on primary axis
    color1 = 'tab:green'
    ax.plot(steps_metrics, test_acc, color=color1, linewidth=2, label='Test Accuracy')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Test Accuracy', color=color1, fontsize=12)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Plot rank on secondary axis
    ax2 = ax.twinx()
    color2 = 'tab:purple'
    ax2.plot(steps_analysis, final_layer_rank, color=color2, linewidth=2, label='Final Layer Rank')
    ax2.set_ylabel('Representation Rank', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Identify grokking transition (when accuracy crosses 0.9)
    acc_threshold = 0.9
    acc_array = np.array(test_acc)
    grokking_steps = np.where(acc_array >= acc_threshold)[0]
    if len(grokking_steps) > 0:
        grokking_step = steps_metrics[grokking_steps[0]]
        ax.axvline(x=grokking_step, color='red', linestyle='--', alpha=0.7,
                  label=f'Grokking transition (~{grokking_step:,} steps)')

    ax.set_title('Grokking Transition: Accuracy vs. Representation Rank',
                fontsize=14, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grokking_transition.png'))
    plt.savefig(os.path.join(output_dir, 'grokking_transition.pdf'))
    plt.close()

    print(f"✓ Grokking transition analysis saved")

    # Print analysis
    if len(grokking_steps) > 0:
        print(f"\nGrokking Transition Analysis:")
        print(f"  Accuracy threshold (0.9) reached at step: {grokking_step:,}")

        # Find rank at grokking step
        closest_analysis_idx = min(range(len(steps_analysis)),
                                  key=lambda i: abs(steps_analysis[i] - grokking_step))
        rank_at_grokking = final_layer_rank[closest_analysis_idx]
        print(f"  Final layer rank at grokking: {rank_at_grokking:.1f}")


def main():
    """Main visualization pipeline."""
    experiment_dir = 'experiments/run_01'
    output_dir = os.path.join(experiment_dir, 'plots')

    print("Loading data...")
    try:
        metrics, analysis = load_metrics(experiment_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find metrics files in {experiment_dir}")
        print(f"Make sure you've run training first: python scripts/train.py")
        sys.exit(1)

    print(f"Loaded {len(metrics['step'])} metric points")
    print(f"Loaded {len(analysis['step'])} analysis points")
    print()

    print("Generating plots...")
    print()

    # Generate all plots
    plot_loss_curves(metrics, output_dir)
    plot_accuracy(metrics, output_dir)
    plot_representation_rank(analysis, output_dir)
    plot_attention_entropy(analysis, output_dir)
    plot_combined_view(metrics, analysis, output_dir)
    plot_grokking_transition_analysis(metrics, analysis, output_dir)

    print()
    print("="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  1. loss_curves.png/pdf - Training and test loss")
    print("  2. test_accuracy.png/pdf - Test accuracy evolution")
    print("  3. representation_rank.png/pdf - Rank by layer")
    print("  4. attention_entropy.png/pdf - Entropy by layer")
    print("  5. combined_view.png/pdf - Loss + Accuracy + Rank aligned")
    print("  6. grokking_transition.png/pdf - Transition analysis")
    print()


if __name__ == '__main__':
    main()
