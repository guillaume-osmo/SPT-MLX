"""
Model Loading

Load MLX model weights and configuration.
"""

import pickle
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Tuple

from .model import GPT, ModelConfig


def load_model(model_path: str, model_name: str) -> Tuple[GPT, ModelConfig]:
    """
    Load MLX model from checkpoint.
    
    Supports MLX format (.npz + .pkl) only.
    
    Args:
        model_path: Path to model directory
        model_name: Model name (without extension)
    
    Returns:
        (model, config) tuple
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    model_path = Path(model_path)
    checkpoint_path = model_path / f"{model_name}.npz"
    config_path = model_path / f"{model_name}.pkl"
    
    # Check if MLX format exists
    if checkpoint_path.exists() and config_path.exists():
        return _load_mlx_model(checkpoint_path, config_path)
    
    raise FileNotFoundError(
        f"MLX model files not found. Expected:\n"
        f"  - {checkpoint_path}\n"
        f"  - {config_path}\n"
        f"\nNote: SPT-MLX only supports MLX format (.npz + .pkl).\n"
        f"PyTorch models (.pth) are not supported - use MLX format only."
    )


class ConfigUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing transprop module"""
    
    def find_class(self, module, name):
        # If trying to import from transprop, create a mock class
        if module.startswith('transprop'):
            # Create a simple dataclass-like object
            class MockConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            return MockConfig
        return super().find_class(module, name)


def _load_mlx_model(checkpoint_path: Path, config_path: Path) -> Tuple[GPT, ModelConfig]:
    """Load MLX model from .npz checkpoint"""
    # Load config - handle pickle files that reference original modules
    with open(config_path, 'rb') as f:
        unpickler = ConfigUnpickler(f)
        config_dict = unpickler.load()
    
    # Extract config values - handle both object and dict formats
    if hasattr(config_dict, 'vocab_size'):
        # Object with attributes
        config = ModelConfig(
            vocab_size=config_dict.vocab_size,
            block_size=config_dict.block_size,
            embed_size=config_dict.embed_size,
            num_layers=config_dict.num_layers,
            num_heads=config_dict.num_heads,
            hidden_factor=config_dict.hidden_factor,
            dropout=config_dict.dropout,
            xT=config_dict.xT,
            mode=config_dict.mode
        )
    elif isinstance(config_dict, dict):
        # Dictionary format
        config = ModelConfig(
            vocab_size=config_dict.get('vocab_size', 100),
            block_size=config_dict.get('block_size', 128),
            embed_size=config_dict.get('embed_size', 512),
            num_layers=config_dict.get('num_layers', 6),
            num_heads=config_dict.get('num_heads', 16),
            hidden_factor=config_dict.get('hidden_factor', 4),
            dropout=config_dict.get('dropout', 0.0),
            xT=config_dict.get('xT', 1),
            mode=config_dict.get('mode', 'reg')
        )
    elif isinstance(config_dict, ModelConfig):
        # Already ModelConfig
        config = config_dict
    else:
        raise ValueError(f"Unknown config format: {type(config_dict)}")
    
    # Create model
    model = GPT(config)
    
    # Load weights
    npz_data = np.load(checkpoint_path, allow_pickle=True)
    
    def convert_to_mlx(obj):
        """Recursively convert numpy arrays to MLX arrays"""
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                if obj.shape == ():
                    return convert_to_mlx(obj.item())
                else:
                    return [convert_to_mlx(obj[i]) for i in range(len(obj))]
            else:
                return mx.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_mlx(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_mlx(item) for item in obj)
        else:
            return obj
    
    weights_flat = {}
    for k in npz_data.files:
        # Skip 'mask' - it's not a trainable parameter
        if 'mask' in k:
            continue
        arr = npz_data[k]
        weights_flat[k] = convert_to_mlx(arr)
    
    npz_data.close()
    
    # Unflatten weights - handle list indices specially
    # For 'blocks.0.ln1.weight', we need to structure it so MLX can update list items
    weights = {}
    block_weights = {}  # Separate dict for block parameters
    
    for flat_key, value in weights_flat.items():
        parts = flat_key.split('.')
        
        # Check if this is a block parameter
        if len(parts) > 1 and parts[0] == 'blocks' and parts[1].isdigit():
            # This is a block parameter: blocks.0.ln1.weight
            block_idx = int(parts[1])
            block_key = '.'.join(parts[2:])  # 'ln1.weight'
            
            if block_idx not in block_weights:
                block_weights[block_idx] = {}
            
            # Build nested structure for this block
            current = block_weights[block_idx]
            block_parts = block_key.split('.')
            for part in block_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[block_parts[-1]] = value
        else:
            # Regular parameter (not in blocks list)
            current = weights
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    
    # Update non-block parameters first
    if weights:
        model.update(weights)
    
    # Update block parameters individually
    # Filter out non-parameter attributes like 'mask'
    if block_weights:
        for block_idx, block_params_dict in block_weights.items():
            if block_idx < len(model.blocks):
                # Get actual parameters of this block
                block_actual_params = dict(model.blocks[block_idx].parameters())
                
                # Build nested structure, skipping 'mask' and other non-parameters
                final_params = {}
                for key, value in block_params_dict.items():
                    parts = key.split('.')
                    
                    # Skip 'mask' - it's not a parameter
                    if 'mask' in parts:
                        continue
                    
                    # Build nested structure
                    current = final_params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                
                # Only update if we have valid parameters
                if final_params:
                    try:
                        model.blocks[block_idx].update(final_params)
                    except Exception as e:
                        # If update fails, try updating submodules individually
                        # This handles nested structures better
                        for submodule_name, submodule_params in final_params.items():
                            if hasattr(model.blocks[block_idx], submodule_name):
                                submodule = getattr(model.blocks[block_idx], submodule_name)
                                if hasattr(submodule, 'update'):
                                    submodule.update(submodule_params)
    
    return model, config
    
    return model, config
