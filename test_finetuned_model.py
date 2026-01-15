#!/usr/bin/env python3
"""
Test fine-tuned model vs original model

Compares predictions and metrics between original and fine-tuned models.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add SPT-MLX to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spt_mlx import load_model, load_vocabulary, predict_batch, predict_with_augmentation


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'n_samples': 0,
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'mean_error': np.nan,
            'std_error': np.nan
        }
    
    errors = y_pred_clean - y_true_clean
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # R²
    ss_res = np.sum((y_true_clean - y_pred_clean)**2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'n_samples': len(y_true_clean),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mean_error': float(mean_error),
        'std_error': float(std_error)
    }


def evaluate_model(model, config, vocab_dict, df_test, model_name, use_augmentation=False):
    """Evaluate model on test data"""
    print(f"\nEvaluating {model_name}...")
    if use_augmentation:
        print("  Using 3×3 augmentation...")
    
    # Prepare data
    solute_list = df_test['SMILES0'].tolist()
    solvent_list = df_test['SMILES1'].tolist()
    temperatures = df_test['T'].tolist()
    true_values = df_test['y0'].values
    
    # Predict
    print(f"  Running inference on {len(df_test)} samples...")
    
    if use_augmentation:
        # Use augmentation for each sample
        predictions = []
        for i, (solute, solvent, temp) in enumerate(zip(solute_list, solvent_list, temperatures)):
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(df_test)} samples...")
            pred, _ = predict_with_augmentation(
                model, config, vocab_dict,
                solute, solvent, temp,
                n_aug_per_smiles=10, n_unique=3
            )
            predictions.append(pred)
        predictions = np.array(predictions)
    else:
        predictions = predict_batch(
            model, config, vocab_dict,
            solute_list, solvent_list, temperatures,
            batch_size=512
        )
        predictions = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(true_values, predictions)
    
    print(f"  Results:")
    print(f"    MAE:  {metrics['mae']:.6f}")
    print(f"    RMSE: {metrics['rmse']:.6f}")
    print(f"    R²:   {metrics['r2']:.6f}")
    print(f"    Mean Error: {metrics['mean_error']:.6f} ± {metrics['std_error']:.6f}")
    
    return predictions, metrics


def main():
    print("="*80)
    print("Fine-tuned Model Evaluation")
    print("="*80)
    
    # Paths
    model_path = "../Models"
    vocab_path = "../vocab/vocab_dict_aug.csv"
    test_data_path = "../raw_data/brouwer_cleaned/brouwer_cleaned.csv"
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab_dict = load_vocabulary(vocab_path)
    print("✓ Vocabulary loaded")
    
    # Load test data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default=test_data_path,
                        help='Path to test data CSV')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use 3×3 augmentation for inference')
    args_test = parser.parse_args()
    
    print(f"\nLoading test data from: {args_test.test_data}")
    df_test = pd.read_csv(args_test.test_data)
    print(f"✓ Loaded {len(df_test)} test samples")
    
    # Convert temperature from Celsius to Kelvin if needed
    if df_test['T'].max() < 300:
        print(f"  Converting temperature from Celsius to Kelvin...")
        print(f"    Original range: {df_test['T'].min():.2f} to {df_test['T'].max():.2f} °C")
        df_test['T'] = df_test['T'] + 273.15
        print(f"    Converted range: {df_test['T'].min():.2f} to {df_test['T'].max():.2f} K")
    
    # Load original model
    print(f"\nLoading original model: model_512_brouwer")
    model_orig, config_orig = load_model(model_path, "model_512_brouwer")
    print("✓ Original model loaded")
    
    # Load fine-tuned model
    print(f"\nLoading fine-tuned model: model_512_brouwer_ft_spt")
    model_ft, config_ft = load_model(model_path, "model_512_brouwer_ft_spt")
    print("✓ Fine-tuned model loaded")
    
    # Evaluate both models
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    pred_orig, metrics_orig = evaluate_model(
        model_orig, config_orig, vocab_dict, df_test, "Original Model",
        use_augmentation=args_test.use_augmentation
    )
    
    pred_ft, metrics_ft = evaluate_model(
        model_ft, config_ft, vocab_dict, df_test, "Fine-tuned Model",
        use_augmentation=args_test.use_augmentation
    )
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"\n{'Metric':<15} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 60)
    
    mae_improvement = ((metrics_orig['mae'] - metrics_ft['mae']) / metrics_orig['mae'] * 100)
    rmse_improvement = ((metrics_orig['rmse'] - metrics_ft['rmse']) / metrics_orig['rmse'] * 100)
    r2_improvement = ((metrics_ft['r2'] - metrics_orig['r2']) / abs(metrics_orig['r2']) * 100) if metrics_orig['r2'] != 0 else 0
    
    print(f"{'MAE':<15} {metrics_orig['mae']:<15.6f} {metrics_ft['mae']:<15.6f} {mae_improvement:>14.2f}%")
    print(f"{'RMSE':<15} {metrics_orig['rmse']:<15.6f} {metrics_ft['rmse']:<15.6f} {rmse_improvement:>14.2f}%")
    print(f"{'R²':<15} {metrics_orig['r2']:<15.6f} {metrics_ft['r2']:<15.6f} {r2_improvement:>14.2f}%")
    
    print("\n" + "="*80)
    if mae_improvement > 0:
        print("✓ Fine-tuned model is BETTER than original!")
    else:
        print("⚠ Fine-tuned model needs more training or different hyperparameters")
    print("="*80)


if __name__ == "__main__":
    main()
