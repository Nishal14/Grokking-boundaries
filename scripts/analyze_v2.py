"""
Post-training analysis for v2/v2b experiments (structured split).

This script is a wrapper around the existing GrokkingAnalyzer that handles
the structured sum-based split used in v2/v2b experiments.

Usage:
    # Analyze v2b sanity check (20k steps)
    python scripts/analyze_v2.py --experiment v2b_conflict_split

    # Analyze v2 full training (400k steps)
    python scripts/analyze_v2.py --experiment v2

    # Analyze with custom t_grok
    python scripts/analyze_v2.py --experiment v2 --t_grok 150000
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DecoderOnlyTransformer
from src.dataset_structured import create_structured_dataloaders
from src.analysis import RepresentationAnalyzer, AttentionAnalyzer


class V2GrokkingAnalyzer:
    """
    Analyzer for v2/v2b experiments with structured split.

    Modified version of GrokkingAnalyzer that uses structured dataloaders.
    """

    def __init__(self, experiment_dir: str, device: Optional[torch.device] = None):
        """
        Initialize analyzer.

        Args:
            experiment_dir: Directory containing checkpoints and logs
            device: Device for inference (defaults to cuda if available)
        """
        self.experiment_dir = Path(experiment_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.config = self._load_config()

        # Load metrics
        self.metrics = self._load_metrics()
        self.analysis_metrics = self._load_analysis_metrics()

        # Create dataloader with STRUCTURED split
        _, self.test_loader = create_structured_dataloaders(
            modulus_p=self.config['data']['modulus_p'],
            batch_size=self.config['training']['batch_size'],
            seed=self.config['data']['seed'],
            num_workers=0
        )

        # Get fixed evaluation batch
        self.eval_batch = next(iter(self.test_loader))

        print(f"V2 Analyzer initialized")
        print(f"  Device: {self.device}")
        print(f"  Experiment directory: {self.experiment_dir}")
        print(f"  Split type: Structured sum-based")
        print(f"  Total logged steps: {len(self.metrics['step'])}")

    def _load_config(self) -> Dict:
        """Load experiment configuration."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_metrics(self) -> Dict:
        """Load training metrics."""
        metrics_path = self.experiment_dir / 'logs' / 'metrics.json'
        with open(metrics_path, 'r') as f:
            return json.load(f)

    def _load_analysis_metrics(self) -> Optional[Dict]:
        """Load analysis metrics if available."""
        analysis_path = self.experiment_dir / 'logs' / 'analysis.json'
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                return json.load(f)
        return None

    def identify_grokking_transition(self, window_size: int = 5, threshold: float = 0.15) -> int:
        """
        Identify the grokking transition step (t_grok).

        Uses a sliding window to detect sharp increases in test accuracy.

        Args:
            window_size: Window size for computing accuracy derivative
            threshold: Minimum increase to consider a grokking transition

        Returns:
            t_grok: Step number of grokking transition
        """
        steps = np.array(self.metrics['step'])
        accuracies = np.array(self.metrics['test_accuracy'])

        # Compute moving average derivative
        derivatives = []
        for i in range(window_size, len(accuracies)):
            before = np.mean(accuracies[i-window_size:i])
            after = accuracies[i]
            derivative = after - before
            derivatives.append(derivative)

        derivatives = np.array(derivatives)

        # Find the maximum derivative (sharpest increase)
        if len(derivatives) == 0:
            print("Warning: Not enough data to detect transition. Using midpoint.")
            return steps[len(steps) // 2]

        max_idx = np.argmax(derivatives)
        t_grok = steps[max_idx + window_size]

        # Check if the increase is significant
        if derivatives[max_idx] < threshold:
            print(f"Warning: Maximum accuracy increase ({derivatives[max_idx]:.3f}) below threshold ({threshold})")
            print(f"Using detected transition at step {t_grok} anyway.")
            print(f"Note: For v2b sanity check, no grokking is expected at 20k steps.")

        print(f"\nGrokking transition detected:")
        print(f"  t_grok: {t_grok}")
        print(f"  Accuracy before: {accuracies[max_idx + window_size - 1]:.4f}")
        print(f"  Accuracy after: {accuracies[max_idx + window_size]:.4f}")
        print(f"  Accuracy increase: {derivatives[max_idx]:.4f}")

        return int(t_grok)

    def select_checkpoints(self, t_grok: int, num_before: int = 3, num_near: int = 2, num_after: int = 3) -> List[int]:
        """
        Select checkpoints for analysis.

        Args:
            t_grok: Grokking transition step
            num_before: Number of checkpoints before t_grok
            num_near: Number of checkpoints near t_grok
            num_after: Number of checkpoints after t_grok

        Returns:
            List of checkpoint step numbers
        """
        checkpoints_dir = self.experiment_dir / 'checkpoints'
        available_checkpoints = []

        # Find all available checkpoints
        for ckpt_file in checkpoints_dir.glob('checkpoint_step_*.pt'):
            step = int(ckpt_file.stem.split('_')[-1])
            available_checkpoints.append(step)

        available_checkpoints = sorted(available_checkpoints)

        if not available_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

        # Select checkpoints
        selected = []

        # Before t_grok
        before_ckpts = [s for s in available_checkpoints if s < t_grok - 10000]
        if before_ckpts:
            selected.extend(before_ckpts[-num_before:])

        # Near t_grok
        near_ckpts = [s for s in available_checkpoints if t_grok - 10000 <= s <= t_grok + 10000]
        if near_ckpts:
            selected.extend(near_ckpts[:num_near])

        # After t_grok
        after_ckpts = [s for s in available_checkpoints if s > t_grok + 10000]
        if after_ckpts:
            selected.extend(after_ckpts[:num_after])

        # Ensure we have at least some checkpoints
        if not selected:
            print("Warning: Selected checkpoints strategy returned empty. Using all available.")
            selected = available_checkpoints

        # Remove duplicates and sort
        selected = sorted(list(set(selected)))

        print(f"\nSelected checkpoints for analysis:")
        print(f"  Total: {len(selected)}")
        print(f"  Before t_grok: {len([s for s in selected if s < t_grok])}")
        print(f"  Near t_grok: {len([s for s in selected if abs(s - t_grok) <= 10000])}")
        print(f"  After t_grok: {len([s for s in selected if s > t_grok])}")
        print(f"  Steps: {selected}")

        return selected

    def load_checkpoint(self, step: int) -> nn.Module:
        """
        Load a model checkpoint.

        Args:
            step: Training step number

        Returns:
            Loaded model in eval mode
        """
        checkpoint_path = self.experiment_dir / 'checkpoints' / f'checkpoint_step_{step}.pt'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model
        model = DecoderOnlyTransformer(
            vocab_size=self.config['model']['vocab_size'],
            d_model=self.config['model']['d_model'],
            num_layers=self.config['model']['n_layers'],
            num_heads=self.config['model']['n_heads'],
            d_ff=self.config['model']['d_ff'],
            max_seq_len=self.config['model']['max_seq_len'],
            dropout=self.config['model']['dropout']
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def compute_representation_rank(self, model: nn.Module) -> Tuple[List[float], List[np.ndarray]]:
        """
        Compute representation rank for all layers.

        Args:
            model: Model to analyze

        Returns:
            ranks: Effective rank per layer
            singular_values: Singular values per layer
        """
        model.eval()

        # Get hidden states
        input_ids = self.eval_batch['input_ids'].to(self.device)

        with torch.no_grad():
            output = model(input_ids, return_hidden_states=True)
            hidden_states_list = output['hidden_states']

        # Compute rank for each layer
        ranks = []
        singular_values = []

        for hidden_states in hidden_states_list:
            rank, sv = RepresentationAnalyzer.compute_effective_rank(
                hidden_states,
                threshold=self.config['analysis']['rank_threshold']
            )
            ranks.append(rank)
            singular_values.append(sv)

        return ranks, singular_values

    def compute_attention_entropy(self, model: nn.Module) -> List[float]:
        """
        Compute attention entropy for all layers.

        Args:
            model: Model to analyze

        Returns:
            entropies: List of average entropy values, one per layer
        """
        model.eval()

        # Get attention weights
        input_ids = self.eval_batch['input_ids'].to(self.device)

        with torch.no_grad():
            _, attentions = model(input_ids)

        # Compute entropy for each layer
        entropies = []
        for attn in attentions:
            entropy = AttentionAnalyzer.compute_entropy(attn)
            entropies.append(entropy)

        return entropies

    def analyze_checkpoint(self, step: int) -> Dict:
        """
        Perform complete analysis on a single checkpoint.

        Args:
            step: Checkpoint step number

        Returns:
            Analysis results dictionary
        """
        print(f"  Analyzing checkpoint at step {step}...")

        # Load model
        model = self.load_checkpoint(step)

        # Compute metrics
        ranks, singular_values = self.compute_representation_rank(model)
        entropies = self.compute_attention_entropy(model)

        # Compute aggregate metrics
        final_layer_rank = ranks[-1]
        avg_rank = np.mean(ranks)
        avg_entropy = np.mean(entropies)

        return {
            'step': step,
            'representation_rank': ranks,
            'singular_values': singular_values,
            'attention_entropy': entropies,
            'final_layer_rank': final_layer_rank,
            'avg_rank': avg_rank,
            'avg_entropy': avg_entropy
        }

    def run_analysis(self, t_grok: Optional[int] = None) -> Dict:
        """
        Run complete post-training analysis.

        Args:
            t_grok: Grokking transition step (auto-detected if None)

        Returns:
            Complete analysis results
        """
        print("="*70)
        print("POST-TRAINING GROKKING ANALYSIS (V2/V2B)")
        print("="*70)
        print()

        # Step 1: Identify grokking transition
        if t_grok is None:
            t_grok = self.identify_grokking_transition()
        else:
            print(f"Using provided t_grok: {t_grok}")

        # Step 2: Select checkpoints
        checkpoint_steps = self.select_checkpoints(t_grok)

        # Step 3: Analyze each checkpoint
        print(f"\nAnalyzing {len(checkpoint_steps)} checkpoints...")
        checkpoint_analyses = []

        for step in checkpoint_steps:
            try:
                analysis = self.analyze_checkpoint(step)
                checkpoint_analyses.append(analysis)
            except Exception as e:
                print(f"  Error analyzing checkpoint {step}: {e}")
                continue

        # Step 4: Align with performance metrics
        print("\nAligning signals with performance metrics...")
        aligned_data = self._align_signals(checkpoint_analyses, t_grok)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

        return aligned_data

    def _align_signals(self, checkpoint_analyses: List[Dict], t_grok: int) -> Dict:
        """
        Align all signals (accuracy, rank, entropy) for analysis.

        Args:
            checkpoint_analyses: List of checkpoint analysis results
            t_grok: Grokking transition step

        Returns:
            Aligned data dictionary
        """
        # Extract checkpoint steps
        checkpoint_steps = [a['step'] for a in checkpoint_analyses]

        # Get accuracy at checkpoint steps
        accuracies = []
        for step in checkpoint_steps:
            # Find closest logged step
            step_diffs = [abs(s - step) for s in self.metrics['step']]
            closest_idx = np.argmin(step_diffs)
            accuracies.append(self.metrics['test_accuracy'][closest_idx])

        # Extract signals
        final_layer_ranks = [a['final_layer_rank'] for a in checkpoint_analyses]
        avg_ranks = [a['avg_rank'] for a in checkpoint_analyses]
        avg_entropies = [a['avg_entropy'] for a in checkpoint_analyses]

        # Compute relative step (distance from t_grok)
        relative_steps = [s - t_grok for s in checkpoint_steps]

        aligned_data = {
            't_grok': t_grok,
            'checkpoint_steps': checkpoint_steps,
            'relative_steps': relative_steps,
            'test_accuracy': accuracies,
            'final_layer_rank': final_layer_ranks,
            'avg_rank': avg_ranks,
            'avg_entropy': avg_entropies,
            'full_analyses': checkpoint_analyses,
            'training_metrics': self.metrics,
            'analysis_metrics': self.analysis_metrics
        }

        # Print summary
        print("\nAlignment Summary:")
        print(f"  t_grok: {t_grok}")
        print(f"  Checkpoints analyzed: {len(checkpoint_steps)}")
        print(f"\n  {'Step':<10} {'Rel.Step':<10} {'Accuracy':<12} {'Rank':<10} {'Entropy':<10}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

        for i in range(len(checkpoint_steps)):
            print(f"  {checkpoint_steps[i]:<10} "
                  f"{relative_steps[i]:<10} "
                  f"{accuracies[i]:<12.4f} "
                  f"{final_layer_ranks[i]:<10.2f} "
                  f"{avg_entropies[i]:<10.4f}")

        return aligned_data

    def save_results(self, results: Dict, output_file: str = 'post_analysis.json'):
        """
        Save analysis results to JSON.

        Args:
            results: Analysis results
            output_file: Output filename
        """
        output_path = self.experiment_dir / output_file

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'full_analyses':
                # Handle nested structure
                serializable_analyses = []
                for analysis in value:
                    serializable_analysis = {}
                    for k, v in analysis.items():
                        if isinstance(v, (list, np.ndarray)):
                            if isinstance(v, np.ndarray):
                                serializable_analysis[k] = v.tolist()
                            else:
                                # Handle list of arrays
                                serializable_analysis[k] = [
                                    arr.tolist() if isinstance(arr, np.ndarray) else arr
                                    for arr in v
                                ]
                        else:
                            serializable_analysis[k] = v
                    serializable_analyses.append(serializable_analysis)
                serializable_results[key] = serializable_analyses
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point for v2/v2b post-training analysis."""
    parser = argparse.ArgumentParser(description='Analyze v2/v2b experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., v2, v2b_conflict_split)')
    parser.add_argument('--t_grok', type=int, default=None,
                        help='Manual t_grok override (auto-detect if not provided)')
    args = parser.parse_args()

    # Build experiment directory path
    experiment_dir = f'experiments/modular_arithmetic/{args.experiment}'

    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        print(f"\nAvailable experiments:")
        base_dir = 'experiments/modular_arithmetic'
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    print(f"  - {item}")
        return

    # Create analyzer
    analyzer = V2GrokkingAnalyzer(experiment_dir)

    # Run analysis
    results = analyzer.run_analysis(t_grok=args.t_grok)

    # Save results
    analyzer.save_results(results)

    print("\nPost-training analysis complete!")
    print(f"Results saved to: {experiment_dir}/post_analysis.json")
    print("\nNext steps:")
    print("  1. Generate plots from post_analysis.json")
    print("  2. Compare with v1 baseline results")


if __name__ == '__main__':
    main()
