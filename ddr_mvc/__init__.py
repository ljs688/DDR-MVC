"""Core package for the minimal DDR-MVC project."""

from .config import DDRMVCConfig
from .trainer import train_ddr_mvc

__all__ = ["DDRMVCConfig", "train_ddr_mvc"]
