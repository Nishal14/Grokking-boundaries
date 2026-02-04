"""
Configuration classes for Grokking Detection experiment.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Dataset configuration."""
    modulus_p: int = 97
    train_split: float = 0.8
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration (~2M parameters)."""
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256  # Increased from 128 to reach ~2M params
    d_ff: int = 1024    # Increased from 512 to reach ~2M params
    dropout: float = 0.1
    max_seq_len: int = 16
    vocab_size: int = 104  # 0-97 (98 integers) + 6 special tokens


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    num_steps: int = 200_000
    log_interval: int = 100
    checkpoint_interval: int = 5000
    eval_num_batches: int = None  # None = full evaluation


@dataclass
class AnalysisConfig:
    """Internal signal analysis configuration."""
    compute_rank_interval: int = 500
    compute_entropy_interval: int = 500
    rank_threshold: float = 0.01  # For effective rank calculation
    analysis_num_batches: int = 10  # Batches to use for analysis


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
