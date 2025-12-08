"""
Configuration classes for the Artemis router training pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DBConfig:
    """Database connection configuration."""

    user: str = "vlmrouter"
    password: str = "vlmrouter"
    host: str = "localhost"
    port: int = 5432
    name: str = "vlmrouter"

    @classmethod
    def from_env(cls) -> "DBConfig":
        """Load database config from environment variables."""
        return cls(
            user=os.getenv("DB_USER", cls.user),
            password=os.getenv("DB_PASS", cls.password),
            host=os.getenv("DB_HOST", cls.host),
            port=int(os.getenv("DB_PORT", str(cls.port))),
            name=os.getenv("DB_NAME", cls.name),
        )

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RewardWeights:
    """Hyperparameters for reward function computation."""

    # Accuracy mode: maximize (A^exp) * H
    accuracy_exp: float = 2.0

    # Cheap mode: A * H - weight * (cost_norm^exp)
    cheap_cost_weight: float = 0.7
    cheap_cost_exp: float = 1.2

    # Fast mode: A * H - weight * (lat_norm^exp)
    fast_lat_weight: float = 0.7
    fast_lat_exp: float = 1.2

    # Balanced mode: (A^a_exp) * H + c_weight * (C^c_exp) - cost_weight * (cost^cost_exp) - lat_weight * (lat^lat_exp)
    balanced_acc_exp: float = 2.0
    balanced_conf_weight: float = 0.3
    balanced_conf_exp: float = 0.5
    balanced_cost_weight: float = 0.3
    balanced_cost_exp: float = 1.1
    balanced_lat_weight: float = 0.3
    balanced_lat_exp: float = 1.1

    # Normalization quantiles
    cost_quantile: float = 0.95
    latency_quantile: float = 0.95


@dataclass
class RouterModelConfig:
    """Configuration for the reward router model."""

    # Text encoder
    text_encoder_name: str = "distilbert-base-uncased"
    freeze_text_encoder: bool = True
    text_pooling: str = "cls"  # "cls" or "mean"

    # Embedding dimensions
    model_emb_dim: int = 32
    mode_emb_dim: int = 16

    # MLP architecture
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"  # "gelu", "relu", "tanh"

    # Input processing
    max_seq_length: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Training
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Metadata augmentation (for generalization)
    enable_metadata_augmentation: bool = True
    question_only_ratio: float = 0.70  # Question + image metadata (no task/dataset labels)
    image_metadata_only_ratio: float = 0.20  # Minimal image metadata only
    full_metadata_ratio: float = 0.10  # Full metadata (original format)

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # "cosine", "linear", "constant"

    # Device and performance
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    save_best_only: bool = True
    metric_for_best: str = "val_mse"  # lower is better
    early_stopping_patience: Optional[int] = None

    # Logging
    log_interval: int = 10  # batches
    eval_interval: int = 1  # epochs

    # Reproducibility
    seed: int = 42


@dataclass
class PathsConfig:
    """File paths for the pipeline."""

    # Base directory
    base_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__)
    ))

    # Data paths
    data_dir: str = field(default_factory=lambda: "data")
    dataset_file: str = field(default_factory=lambda: "router_reward_dataset.parquet")
    model_index_file: str = field(default_factory=lambda: "model_index.json")
    mode_index_file: str = field(default_factory=lambda: "mode_index.json")

    # Model paths
    models_dir: str = field(default_factory=lambda: "models")
    checkpoints_dir: str = field(default_factory=lambda: "models/checkpoints")
    best_model_file: str = field(default_factory=lambda: "best_reward_router.pt")

    # Results paths
    results_dir: str = field(default_factory=lambda: "results")
    eval_summary_file: str = field(default_factory=lambda: "eval_summary.csv")
    plots_dir: str = field(default_factory=lambda: "results/plots")

    def get_full_path(self, relative_path: str) -> str:
        """Get absolute path from relative path."""
        return os.path.join(self.base_dir, relative_path)

    def get_data_path(self, filename: str = None) -> str:
        """Get path in data directory."""
        if filename is None:
            filename = self.dataset_file
        return self.get_full_path(os.path.join(self.data_dir, filename))

    def get_checkpoint_path(self, filename: str = None) -> str:
        """Get path in checkpoints directory."""
        if filename is None:
            filename = self.best_model_file
        return self.get_full_path(os.path.join(self.checkpoints_dir, filename))

    def get_results_path(self, filename: str = None) -> str:
        """Get path in results directory."""
        if filename is None:
            filename = self.eval_summary_file
        return self.get_full_path(os.path.join(self.results_dir, filename))

    def ensure_dirs(self):
        """Create all necessary directories."""
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.results_dir,
            self.plots_dir,
        ]:
            full_path = self.get_full_path(dir_path)
            os.makedirs(full_path, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and baselines."""

    # Model ranking for baseline (smallest to largest)
    # Used for "always_biggest_model" baseline
    model_size_ranking: List[str] = field(default_factory=lambda: [
        "blip2",
        "llava_7b",
        "llava_13b",
        "llava_34b",
        "gpt4v",
        "claude-3-opus",
    ])

    # Modes to evaluate
    mode_names: List[str] = field(default_factory=lambda: [
        "accuracy",
        "cheap",
        "fast",
        "balanced",
    ])

    # Baseline methods
    baselines: List[str] = field(default_factory=lambda: [
        "always_biggest_model",
        "always_cheapest_model",
        "random_model",
    ])

    # Evaluation metrics to track
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "reward",
        "primary_acc",
        "cost_usd",
        "latency_ms",
        "routing_accuracy",  # % match with oracle
    ])

    # Plotting
    generate_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    plot_dpi: int = 150


@dataclass
class Config:
    """Main configuration object combining all sub-configs."""

    db: DBConfig = field(default_factory=DBConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    model: RouterModelConfig = field(default_factory=RouterModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        config = cls()
        config.db = DBConfig.from_env()
        return config

    def __post_init__(self):
        """Post-initialization setup."""
        self.paths.ensure_dirs()


# Default configuration instance
DEFAULT_CONFIG = Config.default()
