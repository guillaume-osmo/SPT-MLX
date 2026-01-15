"""
SPT-MLX: Standalone MLX implementation of SMILES2PropertiesTransformer

A clean, independent MLX version for predicting activity coefficients
from SMILES strings with test-time augmentation.
"""

__version__ = "1.0.0"

from .model import GPT, ModelConfig
from .inference import predict, predict_batch, predict_with_augmentation
from .load_model import load_model
from .tokenizer import load_vocabulary

__all__ = [
    "GPT",
    "ModelConfig", 
    "predict",
    "predict_batch",
    "predict_with_augmentation",
    "load_model",
    "load_vocabulary",
]
