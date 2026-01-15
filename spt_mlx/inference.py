"""
Inference Functions

Predict activity coefficients with optional augmentation.
"""

import numpy as np
import mlx.core as mx
from typing import List, Tuple, Optional

from .model import GPT, ModelConfig
from .tokenizer import smiles_to_tokens, create_sequence, preprocess_xT
from .augmentation import generate_3x3_combinations


def predict(
    model: GPT,
    config: ModelConfig,
    vocab_dict: dict,
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float = 298.15,
    composition: float = 0.0  # Changed from 0.5 to 0.0 for infinite dilution
) -> float:
    """
    Predict ln(γ∞) for a single solute-solvent pair.
    
    Args:
        model: Loaded GPT model
        config: Model configuration
        vocab_dict: Vocabulary dictionary
        solute_smiles: Solute SMILES string
        solvent_smiles: Solvent SMILES string
        temperature: Temperature in Kelvin (default: 298.15)
        composition: Composition value (default: 0.5 for infinite dilution)
    
    Returns:
        Predicted ln(γ∞) value
    """
    # Convert SMILES to tokens
    solute_tokens = smiles_to_tokens(solute_smiles, vocab_dict)
    solvent_tokens = smiles_to_tokens(solvent_smiles, vocab_dict)
    
    # Create sequence
    sequence = create_sequence(solute_tokens, solvent_tokens, config.block_size)
    
    # Preprocess xT
    xt = preprocess_xT(composition, temperature)
    
    # Convert to MLX arrays
    seq_mlx = mx.array(sequence[np.newaxis, :])  # Add batch dimension
    xt_mlx = mx.array(xt[np.newaxis, :])  # Add batch dimension
    
    # Predict
    prediction = model(seq_mlx, xt_mlx, training=False)
    
    # Convert to float
    if prediction.ndim == 0:
        return float(prediction)
    else:
        return float(prediction[0])


def predict_batch(
    model: GPT,
    config: ModelConfig,
    vocab_dict: dict,
    solute_smiles_list: List[str],
    solvent_smiles_list: List[str],
    temperatures: List[float],
    compositions: Optional[List[float]] = None,
    batch_size: int = 512
) -> np.ndarray:
    """
    Predict ln(γ∞) for multiple solute-solvent pairs (batched).
    
    Args:
        model: Loaded GPT model
        config: Model configuration
        vocab_dict: Vocabulary dictionary
        solute_smiles_list: List of solute SMILES strings
        solvent_smiles_list: List of solvent SMILES strings
        temperatures: List of temperatures in Kelvin
        compositions: List of composition values (default: 0.5 for all)
        batch_size: Batch size for inference
    
    Returns:
        Array of predicted ln(γ∞) values
    """
    n_samples = len(solute_smiles_list)
    if compositions is None:
        compositions = [0.0] * n_samples  # Changed from 0.5 to 0.0 for infinite dilution
    
    all_predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_solute = solute_smiles_list[i:batch_end]
        batch_solvent = solvent_smiles_list[i:batch_end]
        batch_temp = temperatures[i:batch_end]
        batch_comp = compositions[i:batch_end]
        
        # Prepare batch
        sequences = []
        xts = []
        
        for sol_smi, solv_smi, temp, comp in zip(batch_solute, batch_solvent, batch_temp, batch_comp):
            # Ensure SMILES are strings
            sol_smi = str(sol_smi)
            solv_smi = str(solv_smi)
            solute_tokens = smiles_to_tokens(sol_smi, vocab_dict)
            solvent_tokens = smiles_to_tokens(solv_smi, vocab_dict)
            seq = create_sequence(solute_tokens, solvent_tokens, config.block_size)
            xt = preprocess_xT(comp, temp)
            sequences.append(seq)
            xts.append(xt)
        
        # Convert to MLX arrays
        seq_batch = mx.array(np.array(sequences))
        xt_batch = mx.array(np.array(xts))
        
        # Predict
        predictions = model(seq_batch, xt_batch, training=False)
        
        # Convert to numpy
        if predictions.ndim == 0:
            all_predictions.append(float(predictions))
        else:
            all_predictions.extend([float(p) for p in predictions])
    
    return np.array(all_predictions)


def predict_with_augmentation(
    model: GPT,
    config: ModelConfig,
    vocab_dict: dict,
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float = 298.15,
    composition: float = 0.0,  # Changed from 0.5 to 0.0 for infinite dilution
    n_aug_per_smiles: int = 10,
    n_unique: int = 3
) -> Tuple[float, Optional[float]]:
    """
    Predict with 3×3 SMILES augmentation and average results.
    
    This implements the recommended augmentation strategy:
    - Generate 10 augmentations per SMILES
    - Keep 3 unique variants
    - Create 3×3 = 9 combinations
    - Average predictions
    
    Args:
        model: Loaded GPT model
        config: Model configuration
        vocab_dict: Vocabulary dictionary
        solute_smiles: Solute SMILES string
        solvent_smiles: Solvent SMILES string
        temperature: Temperature in Kelvin
        composition: Composition value
        n_aug_per_smiles: Number of augmentations to generate per SMILES
        n_unique: Number of unique SMILES to keep (default: 3 for 3×3)
    
    Returns:
        (mean_prediction, std_prediction) tuple
    """
    # Generate 3×3 combinations
    combinations = generate_3x3_combinations(
        solute_smiles, solvent_smiles, n_aug_per_smiles, n_unique
    )
    
    # Predict for each combination
    predictions = []
    for sol_smi, solv_smi in combinations:
        pred = predict(model, config, vocab_dict, sol_smi, solv_smi, temperature, composition)
        predictions.append(pred)
    
    # Average predictions
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions) if len(predictions) > 1 else None
    
    return float(mean_pred), float(std_pred) if std_pred is not None else None
