from dataclasses import dataclass

import torch


@dataclass
class DDRMVCConfig:
    """Runtime configuration for minimal DDR-MVC training."""

    project_name: str = "DDR-MVC"
    dataset_name: str = "MSRCV1"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    feature_dim: int = 128
    hidden_dim: int = 256

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5

    alpha_rec: float = 1.0
    alpha_con: float = 0.1

    seed: int = 42

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
