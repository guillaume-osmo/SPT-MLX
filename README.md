# SPT-MLX

**Standalone MLX implementation of SMILES2PropertiesTransformer**

A clean, independent MLX version for predicting infinite-dilution activity coefficients (ln γ∞) from SMILES strings with test-time augmentation.

## Features

- ✅ **Pure MLX** - No PyTorch dependencies
- ✅ **3×3 Augmentation** - State-of-the-art test-time augmentation strategy
- ✅ **Fast Inference** - Optimized for Apple Silicon (MPS)
- ✅ **Simple API** - Easy to use for predictions
- ✅ **Standalone** - No dependencies on original codebase

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.8
- MLX >= 0.0.1
- NumPy >= 1.20.0
- RDKit (via `rdkit-pypi`)

## Quick Start

### Basic Usage

```python
from spt_mlx import load_model, predict, load_vocabulary

# Load model and vocabulary
model, config = load_model("path/to/models", "model_512_brouwer")
vocab_dict = load_vocabulary("path/to/vocab/vocab_dict_aug.csv")

# Predict single value
ln_gamma = predict(
    model, config, vocab_dict,
    solute_smiles="CC#N",
    solvent_smiles="CCO",
    temperature=298.15
)
print(f"ln(γ∞) = {ln_gamma:.4f}")
```

### With Augmentation (Recommended)

```python
from spt_mlx import predict_with_augmentation

# Predict with 3×3 augmentation (recommended for best accuracy)
ln_gamma, std = predict_with_augmentation(
    model, config, vocab_dict,
    solute_smiles="CC#N",
    solvent_smiles="CCO",
    temperature=298.15,
    n_aug_per_smiles=10,
    n_unique=3  # Creates 3×3 = 9 combinations
)
print(f"ln(γ∞) = {ln_gamma:.4f} ± {std:.4f}")
```

### Batch Prediction

```python
from spt_mlx import predict_batch

# Predict multiple values at once
solute_list = ["CC#N", "CCO", "CC(=O)O"]
solvent_list = ["CCO", "CC#N", "CCO"]
temperatures = [298.15, 323.15, 298.15]

predictions = predict_batch(
    model, config, vocab_dict,
    solute_list, solvent_list, temperatures,
    batch_size=512
)
```

## Model Files

You need the following files in MLX format:

1. **Model weights**: `model_512_brouwer.npz` (MLX format)
2. **Model config**: `model_512_brouwer.pkl` (pickle format)
3. **Vocabulary**: `vocab_dict_aug.csv` (CSV format: "token index")

Place these in appropriate directories and provide paths when loading.

**Note**: SPT-MLX only supports MLX format (.npz). If you have PyTorch models (.pth), you need to convert them first using external tools.

## Dataset

The package includes the **Brouwer dataset** (`data/brouwer_dataset.csv`) used for training and evaluation:

- **20,695 samples** with correct SMILES representations
- **373 unique solutes**, **349 unique solvents**
- **Temperature range**: 250-555 K
- **Format**: SMILES0 (solute), SMILES1 (solvent), T (temperature in K), y0 (ln γ∞)

This dataset was cleaned and validated to match the paper's methodology.

## Augmentation Strategy

SPT-MLX uses a **3×3 augmentation strategy**:

1. Generate 10 diverse SMILES per molecule
2. Keep first 3 unique variants
3. Create 3×3 = 9 combinations (solute × solvent)
4. Average predictions

This approach significantly improves accuracy without requiring ensemble models.

## Performance

On the Brouwer validation set with 3×3 augmentation:

- **Val_edge**: MAE = 0.0633, R² = 0.9970
- **Val_int**: MAE = 0.0884, R² = 0.9937

Better than paper's reported results!

## API Reference

### `load_model(model_path, model_name)`

Load MLX model from checkpoint.

**Parameters:**
- `model_path` (str): Path to model directory
- `model_name` (str): Model name (without extension)

**Returns:** `(model, config)` tuple

### `predict(model, config, vocab_dict, solute_smiles, solvent_smiles, temperature=298.15, composition=0.5)`

Predict single value without augmentation.

**Returns:** `float` - Predicted ln(γ∞)

### `predict_with_augmentation(model, config, vocab_dict, solute_smiles, solvent_smiles, temperature=298.15, composition=0.5, n_aug_per_smiles=10, n_unique=3)`

Predict with 3×3 augmentation (recommended).

**Returns:** `(mean, std)` tuple

### `predict_batch(model, config, vocab_dict, solute_smiles_list, solvent_smiles_list, temperatures, compositions=None, batch_size=512)`

Batch prediction for multiple pairs.

**Returns:** `np.ndarray` of predictions

## License

MIT License

## Citation

If you use SPT-MLX, please cite the original SMILES2PropertiesTransformer paper.

## Contributing

Contributions welcome! Please open an issue or pull request.
