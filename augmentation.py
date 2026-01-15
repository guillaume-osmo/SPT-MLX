"""
SMILES Augmentation

Generates diverse SMILES representations for test-time augmentation.
"""

from typing import List, Set, Tuple
from rdkit import Chem


class SmilesEnumerator:
    """
    Generate diverse SMILES representations of molecules.
    
    Based on the approach from Bjerrum (2017) for SMILES enumeration.
    """
    
    def __init__(self):
        self._cache = {}
    
    def randomize_smiles(self, smiles: str) -> str:
        """
        Generate a randomized SMILES representation.
        
        Args:
            smiles: Canonical SMILES string
        
        Returns:
            Randomized SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        
        # Try to generate a different SMILES
        try:
            randomized = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            return randomized
        except Exception:
            return smiles


def generate_diverse_smiles(
    smiles: str,
    n_augmentations: int = 10,
    n_unique_to_keep: int = 3,
    max_attempts: int = 100
) -> List[str]:
    """
    Generate diverse SMILES representations.
    
    Generates n_augmentations SMILES, keeps first n_unique_to_keep unique ones.
    This implements the 3×3 augmentation strategy: generate 10, keep 3 unique.
    
    Args:
        smiles: Original SMILES string
        n_augmentations: Number of augmented SMILES to generate
        n_unique_to_keep: Number of unique SMILES to keep (default: 3 for 3×3)
        max_attempts: Maximum attempts to generate unique SMILES
    
    Returns:
        List of unique SMILES (original + augmented, up to n_unique_to_keep + 1)
    """
    sme = SmilesEnumerator()
    unique_smiles: Set[str] = {smiles}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    attempts = 0
    while len(unique_smiles) < n_augmentations + 1 and attempts < max_attempts:
        try:
            augmented = sme.randomize_smiles(smiles)
            if augmented and augmented != smiles and len(augmented) > 0:
                unique_smiles.add(augmented)
        except Exception:
            pass
        attempts += 1
    
    result = [smiles]  # Original first
    augmented_list = [s for s in unique_smiles if s != smiles]
    result.extend(augmented_list)
    
    # Return up to n_unique_to_keep + 1 (original + augmented)
    return result[:n_unique_to_keep + 1]


def generate_3x3_combinations(
    solute_smiles: str,
    solvent_smiles: str,
    n_aug_per_smiles: int = 10,
    n_unique: int = 3
) -> List[Tuple[str, str]]:
    """
    Generate 3×3 SMILES combinations for augmentation.
    
    This implements the strategy: generate 10 augs per SMILES, keep 3 unique,
    create 3×3 = 9 combinations.
    
    Args:
        solute_smiles: Solute SMILES
        solvent_smiles: Solvent SMILES
        n_aug_per_smiles: Number of augmentations to generate per SMILES
        n_unique: Number of unique SMILES to keep (default: 3)
    
    Returns:
        List of (solute, solvent) SMILES pairs (9 combinations)
    """
    solute_variants = generate_diverse_smiles(
        solute_smiles, n_aug_per_smiles, n_unique
    )
    solvent_variants = generate_diverse_smiles(
        solvent_smiles, n_aug_per_smiles, n_unique
    )
    
    # Create 3×3 combinations (use first 3 of each)
    combinations = []
    for solute_var in solute_variants[:n_unique]:
        for solvent_var in solvent_variants[:n_unique]:
            combinations.append((solute_var, solvent_var))
    
    return combinations
