#!/usr/bin/env python3
"""
Fine-tune SPT-MLX model for one epoch

Uses the SPT-MLX package for fine-tuning with:
- Batch processing
- MSE loss
- Adam optimizer
- MLX GPU support
"""

import sys
import os
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from typing import List, Tuple

# Add SPT-MLX to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spt_mlx import load_model, load_vocabulary, predict_batch
from spt_mlx.tokenizer import smiles_to_tokens, create_sequence, preprocess_xT
from spt_mlx.model import GPT, ModelConfig


def create_training_data(df: pd.DataFrame, vocab_dict: dict, config: ModelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training data from DataFrame.
    
    Args:
        df: DataFrame with SMILES0, SMILES1, T, y0 columns
        vocab_dict: Vocabulary dictionary
        config: Model configuration
    
    Returns:
        (sequences, xts, targets) as numpy arrays
    """
    sequences = []
    xts = []
    targets = []
    
    print(f"Creating training data from {len(df)} samples...")
    
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
        
        # Get SMILES and temperature
        solute = str(row['SMILES0'])
        solvent = str(row['SMILES1'])
        temp = float(row['T'])
        target = float(row['y0'])
        
        # Tokenize and create sequence
        solute_tokens = smiles_to_tokens(solute, vocab_dict)
        solvent_tokens = smiles_to_tokens(solvent, vocab_dict)
        seq = create_sequence(solute_tokens, solvent_tokens, config.block_size)
        
        # Preprocess xT (composition = 0.0 for infinite dilution)
        xt = preprocess_xT(0.0, temp)
        
        sequences.append(seq)
        xts.append(xt)
        targets.append(target)
    
    return np.array(sequences), np.array(xts), np.array(targets)


def train_one_epoch(
    model: GPT,
    sequences: np.ndarray,
    xts: np.ndarray,
    targets: np.ndarray,
    optimizer: optim.Optimizer,
    batch_size: int = 512
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Returns:
        (average_loss, average_mae)
    """
    n_samples = len(sequences)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    total_loss = 0.0
    total_mae = 0.0
    
    # Loss function - takes model and data, returns scalar loss
    def loss_fn(seq_batch, xt_batch, target_batch):
        predictions = model(seq_batch, xt_batch, training=True)
        loss = nn.losses.mse_loss(predictions, target_batch)
        return loss
    
    # Gradient function - MLX requires model as first argument
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    print(f"\nTraining for {n_batches} batches (batch_size={batch_size})...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Get batch - MLX automatically uses default device (GPU if available)
        seq_batch = mx.array(sequences[start_idx:end_idx])
        xt_batch = mx.array(xts[start_idx:end_idx])
        target_batch = mx.array(targets[start_idx:end_idx])
        
        # Forward and backward pass
        loss, grads = loss_and_grad_fn(seq_batch, xt_batch, target_batch)
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        
        # Calculate metrics
        predictions = model(seq_batch, xt_batch, training=False)
        batch_loss = float(loss)
        batch_mae = float(mx.mean(mx.abs(predictions - target_batch)))
        
        total_loss += batch_loss
        total_mae += batch_mae
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches}: Loss={batch_loss:.6f}, MAE={batch_mae:.6f}")
    
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    
    return avg_loss, avg_mae


def evaluate_model(
    model: GPT,
    sequences: np.ndarray,
    xts: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 512
) -> Tuple[float, float, float]:
    """
    Evaluate model on data.
    
    Returns:
        (mse, mae, rmse)
    """
    n_samples = len(sequences)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_targets = []
    
    print(f"\nEvaluating on {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        seq_batch = mx.array(sequences[start_idx:end_idx])
        xt_batch = mx.array(xts[start_idx:end_idx])
        target_batch = targets[start_idx:end_idx]
        
        # MLX doesn't need no_grad - just set training=False
        predictions = model(seq_batch, xt_batch, training=False)
        all_predictions.extend([float(p) for p in predictions])
        all_targets.extend(target_batch.tolist())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{n_batches} batches...")
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse


def save_model_mlx(model: GPT, config: ModelConfig, output_path: Path, model_name: str):
    """Save MLX model to .npz file"""
    model_path = output_path / f"{model_name}.npz"
    config_path = output_path / f"{model_name}.pkl"
    
    print(f"\nSaving model to {model_path}...")
    
    # Flatten parameters for saving
    params_dict = dict(model.parameters())
    
    def flatten_params(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_params(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Handle list/tuple (like blocks)
                for i, item in enumerate(v):
                    list_key = f"{new_key}.{i}"
                    if isinstance(item, dict):
                        items.extend(flatten_params(item, list_key, sep=sep).items())
                    else:
                        # Try to convert to numpy
                        try:
                            arr = np.array(item)
                            items.append((list_key, arr))
                        except:
                            pass
            else:
                # Try to convert to numpy (handles MLX arrays)
                try:
                    import mlx.core as mx
                    if isinstance(v, mx.array):
                        arr = np.array(v)
                        items.append((new_key, arr))
                    elif hasattr(v, '__array__'):
                        arr = np.array(v)
                        items.append((new_key, arr))
                except:
                    pass
        return dict(items)
    
    flattened_params = flatten_params(params_dict)
    np.savez(str(model_path), **flattened_params)
    
    # Save config
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SPT-MLX model')
    parser.add_argument('--model_path', type=str, default='../Models',
                        help='Path to model directory')
    parser.add_argument('--model_name', type=str, default='model_512_brouwer',
                        help='Model name (without extension)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data CSV (with SMILES0, SMILES1, T, y0 columns)')
    parser.add_argument('--vocab_path', type=str, default='../vocab/vocab_dict_aug.csv',
                        help='Path to vocabulary file')
    parser.add_argument('--output_path', type=str, default='../Models',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--output_name', type=str, default='model_512_brouwer_ft',
                        help='Output model name')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate (default: 1e-6)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use MLX GPU if available')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SPT-MLX Fine-tuning")
    print("="*80)
    
    # Check MLX device
    if args.use_gpu and mx.metal.is_available():
        device = mx.Device(mx.gpu, 0)
        print(f"\n✓ Using MLX GPU: {device}")
    else:
        device = None
        print(f"\nUsing CPU")
    
    # Load model and vocabulary
    print(f"\nLoading model: {args.model_name}")
    model, config = load_model(args.model_path, args.model_name)
    vocab_dict = load_vocabulary(args.vocab_path)
    print("✓ Model and vocabulary loaded")
    
    # Load training data
    print(f"\nLoading training data from: {args.data_path}")
    df_train = pd.read_csv(args.data_path)
    print(f"✓ Loaded {len(df_train)} training samples")
    print(f"  Columns: {df_train.columns.tolist()}")
    
    # Verify required columns
    required_cols = ['SMILES0', 'SMILES1', 'T', 'y0']
    missing_cols = [col for col in required_cols if col not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert temperature from Celsius to Kelvin if needed
    if df_train['T'].max() < 300:
        print(f"  Converting temperature from Celsius to Kelvin...")
        print(f"    Original range: {df_train['T'].min():.2f} to {df_train['T'].max():.2f} °C")
        df_train['T'] = df_train['T'] + 273.15
        print(f"    Converted range: {df_train['T'].min():.2f} to {df_train['T'].max():.2f} K")
    
    # Create training data
    sequences, xts, targets = create_training_data(df_train, vocab_dict, config)
    print(f"✓ Created training data: {len(sequences)} samples")
    
    # Evaluate original model
    print("\n" + "="*80)
    print("Evaluating Original Model")
    print("="*80)
    orig_mse, orig_mae, orig_rmse = evaluate_model(model, sequences, xts, targets, batch_size=args.batch_size)
    print(f"Original Model - MSE: {orig_mse:.6f}, MAE: {orig_mae:.6f}, RMSE: {orig_rmse:.6f}")
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=args.lr)
    
    # Fine-tune for specified epochs
    print("\n" + "="*80)
    print(f"Fine-tuning for {args.epochs} epoch(s)")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        start_time = time.time()
        
        # Train
        avg_loss, avg_mae = train_one_epoch(
            model, sequences, xts, targets,
            optimizer, batch_size=args.batch_size
        )
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  Time: {elapsed:.2f}s")
        
        # Evaluate after epoch
        mse, mae, rmse = evaluate_model(model, sequences, xts, targets, batch_size=args.batch_size)
        print(f"\n  Validation - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    final_mse, final_mae, final_rmse = evaluate_model(model, sequences, xts, targets, batch_size=args.batch_size)
    print(f"Fine-tuned Model - MSE: {final_mse:.6f}, MAE: {final_mae:.6f}, RMSE: {final_rmse:.6f}")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    print(f"Original Model - MSE: {orig_mse:.6f}, MAE: {orig_mae:.6f}, RMSE: {orig_rmse:.6f}")
    print(f"Fine-tuned Model - MSE: {final_mse:.6f}, MAE: {final_mae:.6f}, RMSE: {final_rmse:.6f}")
    print(f"\nImprovement:")
    print(f"  MSE: {((orig_mse - final_mse) / orig_mse * 100):.2f}%")
    print(f"  MAE: {((orig_mae - final_mae) / orig_mae * 100):.2f}%")
    print(f"  RMSE: {((orig_rmse - final_rmse) / orig_rmse * 100):.2f}%")
    
    # Save model
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_model_mlx(model, config, output_path, args.output_name)
    
    print("\n" + "="*80)
    print("Fine-tuning Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
