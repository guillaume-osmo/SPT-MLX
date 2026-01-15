"""
SMILES Tokenization

Converts SMILES strings to token sequences for the model.
"""

import csv
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary dictionary from CSV file.
    
    Args:
        vocab_path: Path to vocabulary CSV file (format: "token index")
    
    Returns:
        Dictionary mapping tokens to indices
    """
    vocab_dict = {}
    with open(vocab_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            if len(row) >= 2:
                vocab_dict[row[0]] = int(row[1])
    return vocab_dict


def smiles_to_tokens(smiles: str, vocab_dict: Dict[str, int]) -> np.ndarray:
    """
    Convert SMILES string to token sequence.
    
    Args:
        smiles: SMILES string
        vocab_dict: Vocabulary dictionary
    
    Returns:
        Array of token indices
    """
    # Handle special case for water
    if smiles == 'O':
        return np.array([vocab_dict.get('H2O', vocab_dict.get('O', 0))])
    
    # Convert each character to token
    tokens = [vocab_dict.get(char, 0) for char in smiles]
    return np.array(tokens)


def create_sequence(
    solute_tokens: np.ndarray,
    solvent_tokens: np.ndarray,
    max_length: int = 128
) -> np.ndarray:
    """
    Create input sequence: <SOS> solute <MOS> solvent <EOS>
    
    Args:
        solute_tokens: Token sequence for solute
        solvent_tokens: Token sequence for solvent
        max_length: Maximum sequence length
    
    Returns:
        Padded sequence array
    """
    sos = np.array([1], dtype=int)  # Start of sequence
    mos = np.array([2], dtype=int)  # Middle of sequence
    eos = np.array([3], dtype=int)  # End of sequence
    
    seq = np.concatenate([sos, solute_tokens, mos, solvent_tokens, eos])
    
    # Pad or truncate to max_length
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        padded = np.zeros(max_length, dtype=int)
        padded[:len(seq)] = seq
        seq = padded
    
    return seq


def preprocess_xT(composition: float, temperature: float) -> np.ndarray:
    """
    Preprocess temperature and composition for model input.
    
    Args:
        composition: Composition value (typically 0.5 for infinite dilution)
        temperature: Temperature in Kelvin
    
    Returns:
        Preprocessed xT array [x, T]
    """
    # Same preprocessing as original model
    x = composition - 0.5
    T = temperature / 298.5 - 1.0
    return np.array([x, T], dtype=np.float32)
