"""
Evaluation utilities for comprehensive model assessment.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from src.tokenizer import ModularArithmeticTokenizer


class Evaluator:
    """
    Comprehensive evaluator for modular arithmetic models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        tokenizer: ModularArithmeticTokenizer,
        device: torch.device
    ):
        """
        Initialize evaluator.

        Args:
            model: The transformer model
            test_loader: Test data loader
            tokenizer: Tokenizer for decoding
            device: Device for inference
        """
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def full_evaluation(self) -> Dict:
        """
        Comprehensive evaluation with detailed metrics.

        Returns:
            Dictionary with:
                - exact_match_accuracy: Accuracy on result token
                - per_position_accuracy: Accuracy for each position in sequence
                - test_loss: Average cross-entropy loss
                - predictions: List of predicted results
                - labels: List of ground truth results
        """
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        all_predictions = []
        all_labels = []
        all_a = []
        all_b = []
        total_loss = 0.0

        # Track per-position accuracy (8 positions in sequence)
        # Positions: <bos> a + b mod p = c <eos>
        #            0    1 2 3 4   5 6 7
        per_position_correct = [0] * 8
        total_samples = 0

        for batch in self.test_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            result = batch['result']
            a = batch['a']
            b = batch['b']

            # Forward pass
            output = self.model(input_ids)
            logits = output['logits']

            # Loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()

            # Predictions
            pred_tokens = logits.argmax(dim=-1)  # (batch_size, seq_len)

            # Store predictions for result position
            pred_result = pred_tokens[:, 6].cpu()  # Position 6 predicts the result 'c'
            all_predictions.extend(pred_result.tolist())
            all_labels.extend(result.tolist())
            all_a.extend(a.tolist())
            all_b.extend(b.tolist())

            # Per-position accuracy
            correct_per_pos = (pred_tokens == labels).cpu()  # (batch_size, seq_len)
            for pos in range(8):
                per_position_correct[pos] += correct_per_pos[:, pos].sum().item()

            total_samples += input_ids.size(0)

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        exact_match_acc = (all_predictions == all_labels).mean()
        per_position_acc = [correct / total_samples for correct in per_position_correct]
        avg_loss = total_loss / len(self.test_loader)

        return {
            'exact_match_accuracy': exact_match_acc,
            'per_position_accuracy': per_position_acc,
            'test_loss': avg_loss,
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'a_values': all_a,
            'b_values': all_b
        }

    def error_analysis(
        self,
        predictions: List[int],
        labels: List[int],
        a_values: List[int],
        b_values: List[int]
    ) -> List[Dict]:
        """
        Identify examples where the model makes errors.

        Args:
            predictions: Predicted result values
            labels: Ground truth result values
            a_values: First operands
            b_values: Second operands

        Returns:
            List of error dictionaries with a, b, true_result, predicted_result
        """
        errors = []

        for i, (pred, label, a, b) in enumerate(zip(predictions, labels, a_values, b_values)):
            if pred != label:
                errors.append({
                    'index': i,
                    'a': a,
                    'b': b,
                    'true_result': label,
                    'predicted_result': pred,
                    'error_magnitude': abs(pred - label)
                })

        return errors

    def analyze_error_patterns(self, errors: List[Dict]) -> Dict:
        """
        Analyze patterns in model errors.

        Args:
            errors: List of error dictionaries from error_analysis()

        Returns:
            Dictionary with error statistics
        """
        if not errors:
            return {
                'num_errors': 0,
                'error_rate': 0.0
            }

        error_magnitudes = [e['error_magnitude'] for e in errors]

        return {
            'num_errors': len(errors),
            'error_rate': len(errors) / (len(errors) + 1),  # Approximate
            'mean_error_magnitude': np.mean(error_magnitudes),
            'median_error_magnitude': np.median(error_magnitudes),
            'max_error_magnitude': np.max(error_magnitudes)
        }

    def sample_predictions(self, num_samples: int = 10) -> List[str]:
        """
        Sample predictions and format them as readable strings.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of formatted prediction strings
        """
        self.model.eval()

        samples = []
        count = 0

        for batch in self.test_loader:
            if count >= num_samples:
                break

            input_ids = batch['input_ids'].to(self.device)
            result = batch['result']
            a = batch['a']
            b = batch['b']

            # Forward pass
            output = self.model(input_ids)
            logits = output['logits']
            pred_tokens = logits.argmax(dim=-1)

            # Get predictions
            pred_result = pred_tokens[:, 6].cpu()

            # Format samples
            for i in range(min(input_ids.size(0), num_samples - count)):
                true_val = result[i].item()
                pred_val = pred_result[i].item()
                a_val = a[i].item()
                b_val = b[i].item()

                status = "✓" if pred_val == true_val else "✗"

                sample_str = (
                    f"{status} {a_val} + {b_val} mod {self.tokenizer.modulus_p} = "
                    f"{pred_val} (true: {true_val})"
                )
                samples.append(sample_str)

                count += 1
                if count >= num_samples:
                    break

        return samples
