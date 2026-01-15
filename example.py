#!/usr/bin/env python3
"""
Example usage of SPT-MLX

Demonstrates basic prediction, augmentation, and batch processing.
"""

from spt_mlx import load_model, predict, predict_with_augmentation, predict_batch
from spt_mlx.tokenizer import load_vocabulary


def main():
    # Paths to model files (adjust as needed)
    model_path = "Models"
    model_name = "model_512_brouwer"
    vocab_path = "../vocab/vocab_dict_aug.csv"
    
    print("Loading model and vocabulary...")
    model, config = load_model(model_path, model_name)
    vocab_dict = load_vocabulary(vocab_path)
    print("✓ Model loaded successfully!")
    
    # Example 1: Basic prediction
    print("\n" + "="*60)
    print("Example 1: Basic Prediction")
    print("="*60)
    solute = "CC#N"  # Acetonitrile
    solvent = "CCO"  # Ethanol
    temp = 298.15
    
    ln_gamma = predict(model, config, vocab_dict, solute, solvent, temp)
    print(f"Solute: {solute}")
    print(f"Solvent: {solvent}")
    print(f"Temperature: {temp} K")
    print(f"Predicted ln(γ∞) = {ln_gamma:.4f}")
    
    # Example 2: Prediction with augmentation (recommended)
    print("\n" + "="*60)
    print("Example 2: Prediction with 3×3 Augmentation")
    print("="*60)
    ln_gamma_aug, std = predict_with_augmentation(
        model, config, vocab_dict, solute, solvent, temp,
        n_aug_per_smiles=10,
        n_unique=3
    )
    print(f"Predicted ln(γ∞) = {ln_gamma_aug:.4f} ± {std:.4f}")
    print(f"(Using 3×3 = 9 SMILES combinations)")
    
    # Example 3: Batch prediction
    print("\n" + "="*60)
    print("Example 3: Batch Prediction")
    print("="*60)
    solute_list = ["CC#N", "CCO", "CC(=O)O"]
    solvent_list = ["CCO", "CC#N", "CCO"]
    temperatures = [298.15, 323.15, 298.15]
    
    predictions = predict_batch(
        model, config, vocab_dict,
        solute_list, solvent_list, temperatures
    )
    
    print("Batch predictions:")
    for i, (sol, solv, T, pred) in enumerate(zip(solute_list, solvent_list, temperatures, predictions)):
        print(f"  {i+1}. {sol} / {solv} @ {T}K: ln(γ∞) = {pred:.4f}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
