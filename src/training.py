"""
Training loop with integrated internal signal analysis.
"""
import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm

from config import TrainingConfig, AnalysisConfig
from src.analysis import RepresentationAnalyzer, AttentionAnalyzer, AnalysisLogger


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self):
        self.data = {
            'step': [],
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': [],
            'learning_rate': []
        }

    def log(self, step: int, train_loss: float, test_loss: float, test_accuracy: float, learning_rate: float):
        """Log metrics for a step."""
        self.data['step'].append(step)
        self.data['train_loss'].append(train_loss)
        self.data['test_loss'].append(test_loss)
        self.data['test_accuracy'].append(test_accuracy)
        self.data['learning_rate'].append(learning_rate)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.data


class Trainer:
    """
    Trainer for grokking experiments with integrated internal signal analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        training_config: TrainingConfig,
        analysis_config: AnalysisConfig,
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize trainer.

        Args:
            model: The transformer model
            train_loader: Training data loader
            test_loader: Test data loader
            training_config: Training hyperparameters
            analysis_config: Analysis configuration
            device: Device for training
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_config = training_config
        self.analysis_config = analysis_config
        self.device = device
        self.output_dir = output_dir

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.metrics_logger = MetricsLogger()
        self.analysis_logger = AnalysisLogger()

        # Create output directories
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    def train_step(self, batch: Dict) -> float:
        """
        Single training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Loss value
        """
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        logits, _ = self.model(input_ids)

        # Compute loss (flatten for cross-entropy)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """
        Evaluate on test set.

        Returns:
            test_loss: Average test loss
            test_accuracy: Exact match accuracy on result token
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        num_batches = len(self.test_loader)
        if self.training_config.eval_num_batches is not None:
            num_batches = min(num_batches, self.training_config.eval_num_batches)

        for i, batch in enumerate(self.test_loader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            result = batch['result']

            # Forward pass
            logits, _ = self.model(input_ids)

            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()

            # Accuracy: check if prediction at position 6 (result token 'c') is correct
            # Sequence: <bos> a + b mod p = c <eos>
            # Positions: 0    1 2 3 4   5 6 7
            # We predict position 7 from position 6, so check logits[:, 6]
            pred_tokens = logits.argmax(dim=-1)
            pred_result_token = pred_tokens[:, 6]  # Token at position 6 predicts 'c'

            # Convert predicted token back to integer (tokens 0-96 map to integers 0-96)
            pred_result = pred_result_token.cpu()

            # Compare with ground truth
            correct += (pred_result == result).sum().item()
            total += input_ids.size(0)

        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def run_analysis(self, step: int):
        """
        Run internal signal analysis.

        Args:
            step: Current training step
        """
        self.model.eval()

        # Compute representation rank
        ranks, singular_values = RepresentationAnalyzer.compute_rank_per_layer(
            model=self.model,
            dataloader=self.test_loader,
            device=self.device,
            num_batches=self.analysis_config.analysis_num_batches,
            threshold=self.analysis_config.rank_threshold
        )

        # Compute attention entropy
        entropies = AttentionAnalyzer.compute_entropy_per_layer(
            model=self.model,
            dataloader=self.test_loader,
            device=self.device,
            num_batches=self.analysis_config.analysis_num_batches
        )

        # Log analysis metrics
        self.analysis_logger.log(step, ranks, singular_values, entropies)

        return ranks, entropies

    def save_checkpoint(self, step: int):
        """
        Save model checkpoint.

        Args:
            step: Current training step
        """
        checkpoint_path = os.path.join(
            self.output_dir,
            'checkpoints',
            f'checkpoint_step_{step}.pt'
        )

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_config': self.training_config.__dict__,
            'analysis_config': self.analysis_config.__dict__
        }, checkpoint_path)

    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = os.path.join(self.output_dir, 'logs', 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_logger.to_dict(), f, indent=2)

    def save_analysis_metrics(self):
        """Save analysis metrics to JSON."""
        analysis_path = os.path.join(self.output_dir, 'logs', 'analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(self.analysis_logger.to_dict(), f, indent=2)

    def train(self):
        """
        Main training loop for 200k steps.
        """
        print("Starting training...")
        print(f"Total steps: {self.training_config.num_steps}")
        print(f"Logging every {self.training_config.log_interval} steps")
        print(f"Analysis every {self.analysis_config.compute_rank_interval} steps")
        print(f"Checkpoints every {self.training_config.checkpoint_interval} steps")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print()

        step = 0
        train_iter = iter(self.train_loader)

        # Progress bar
        pbar = tqdm(total=self.training_config.num_steps, desc="Training")

        while step < self.training_config.num_steps:
            # Get next batch (cycle through dataset indefinitely)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            train_loss = self.train_step(batch)
            step += 1
            pbar.update(1)

            # Logging
            if step % self.training_config.log_interval == 0:
                test_loss, test_acc = self.evaluate()

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log metrics
                self.metrics_logger.log(step, train_loss, test_loss, test_acc, current_lr)

                # Print progress
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'test_loss': f'{test_loss:.4f}',
                    'test_acc': f'{test_acc:.4f}'
                })

                # Save metrics
                self.save_metrics()

            # Analysis
            if step % self.analysis_config.compute_rank_interval == 0:
                ranks, entropies = self.run_analysis(step)

                # Print analysis summary
                final_layer_rank = ranks[-1]
                avg_entropy = sum(e.mean() for e in entropies) / len(entropies)

                print(f"\n[Step {step}] Analysis:")
                print(f"  Final layer rank: {final_layer_rank:.1f}")
                print(f"  Average attention entropy: {avg_entropy:.3f}")

                # Save analysis metrics
                self.save_analysis_metrics()

            # Checkpointing
            if step % self.training_config.checkpoint_interval == 0:
                self.save_checkpoint(step)
                print(f"\n[Step {step}] Checkpoint saved")

        pbar.close()

        # Final checkpoint and metrics
        self.save_checkpoint(step)
        self.save_metrics()
        self.save_analysis_metrics()

        print("\nTraining complete!")
        print(f"Outputs saved to: {self.output_dir}")
