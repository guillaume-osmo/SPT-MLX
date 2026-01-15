#!/usr/bin/env python3
"""
Test script for SPT-MLX package

Tests model loading, tokenization, and prediction functionality.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spt_mlx import load_model, load_vocabulary, predict, predict_with_augmentation, predict_batch


def test_model_loading():
    """Test model loading"""
    print("="*60)
    print("Test 1: Model Loading")
    print("="*60)
    
    model_path = "Models"
    model_name = "model_512_brouwer"
    
    try:
        model, config = load_model(model_path, model_name)
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Config vocab_size: {config.vocab_size}")
        print(f"  Config embed_size: {config.embed_size}")
        print(f"  Config num_layers: {config.num_layers}")
        return model, config
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None


def test_vocabulary_loading():
    """Test vocabulary loading"""
    print("\n" + "="*60)
    print("Test 2: Vocabulary Loading")
    print("="*60)
    
    vocab_path = "vocab/vocab_dict_aug.csv"
    
    try:
        vocab_dict = load_vocabulary(vocab_path)
        print(f"✓ Vocabulary loaded successfully!")
        print(f"  Vocabulary size: {len(vocab_dict)}")
        print(f"  Sample tokens: {list(vocab_dict.items())[:5]}")
        return vocab_dict
    except Exception as e:
        print(f"✗ Error loading vocabulary: {e}")
        print(f"  Tried path: {vocab_path}")
        return None


def test_basic_prediction(model, config, vocab_dict):
    """Test basic prediction without augmentation"""
    print("\n" + "="*60)
    print("Test 3: Basic Prediction (No Augmentation)")
    print("="*60)
    
    if model is None or vocab_dict is None:
        print("✗ Skipping - model or vocabulary not loaded")
        return
    
    solute = "CC#N"  # Acetonitrile
    solvent = "CCO"  # Ethanol
    temp = 298.15
    
    try:
        ln_gamma = predict(model, config, vocab_dict, solute, solvent, temp)
        print(f"✓ Prediction successful!")
        print(f"  Solute: {solute}")
        print(f"  Solvent: {solvent}")
        print(f"  Temperature: {temp} K")
        print(f"  Predicted ln(γ∞) = {ln_gamma:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmented_prediction(model, config, vocab_dict):
    """Test prediction with 3×3 augmentation"""
    print("\n" + "="*60)
    print("Test 4: Prediction with 3×3 Augmentation")
    print("="*60)
    
    if model is None or vocab_dict is None:
        print("✗ Skipping - model or vocabulary not loaded")
        return
    
    solute = "CC#N"  # Acetonitrile
    solvent = "CCO"  # Ethanol
    temp = 298.15
    
    try:
        ln_gamma, std = predict_with_augmentation(
            model, config, vocab_dict,
            solute, solvent, temp,
            n_aug_per_smiles=10,
            n_unique=3
        )
        print(f"✓ Augmented prediction successful!")
        print(f"  Solute: {solute}")
        print(f"  Solvent: {solvent}")
        print(f"  Temperature: {temp} K")
        print(f"  Predicted ln(γ∞) = {ln_gamma:.4f} ± {std:.4f}")
        print(f"  (Using 3×3 = 9 SMILES combinations)")
        return True
    except Exception as e:
        print(f"✗ Error in augmented prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction(model, config, vocab_dict):
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Test 5: Batch Prediction")
    print("="*60)
    
    if model is None or vocab_dict is None:
        print("✗ Skipping - model or vocabulary not loaded")
        return
    
    solute_list = ["CC#N", "CCO", "CC(=O)O"]
    solvent_list = ["CCO", "CC#N", "CCO"]
    temperatures = [298.15, 323.15, 298.15]
    
    try:
        predictions = predict_batch(
            model, config, vocab_dict,
            solute_list, solvent_list, temperatures,
            batch_size=512
        )
        print(f"✓ Batch prediction successful!")
        print(f"  Processed {len(predictions)} predictions:")
        for i, (sol, solv, T, pred) in enumerate(zip(solute_list, solvent_list, temperatures, predictions)):
            print(f"    {i+1}. {sol} / {solv} @ {T}K: ln(γ∞) = {pred:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error in batch prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SPT-MLX Package Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Load model
    model, config = test_model_loading()
    
    # Test 2: Load vocabulary
    vocab_dict = test_vocabulary_loading()
    
    # Test 3: Basic prediction
    if model and vocab_dict:
        test_basic_prediction(model, config, vocab_dict)
    
    # Test 4: Augmented prediction
    if model and vocab_dict:
        test_augmented_prediction(model, config, vocab_dict)
    
    # Test 5: Batch prediction
    if model and vocab_dict:
        test_batch_prediction(model, config, vocab_dict)
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
