# SPT-MLX Release Notes

## Version 1.0.0

### Features
- Standalone MLX implementation (no PyTorch dependencies)
- 3×3 SMILES augmentation strategy
- Fast inference optimized for Apple Silicon
- Simple, clean API
- Batch prediction support

### Model Files Required
- `model_512_brouwer.npz` - MLX model weights
- `model_512_brouwer.pkl` - Model configuration
- `vocab_dict_aug.csv` - Vocabulary dictionary

### Performance
- Val_edge: MAE = 0.0633, R² = 0.9970 (with 3×3 augmentation)
- Val_int: MAE = 0.0884, R² = 0.9937 (with 3×3 augmentation)

### Usage
See README.md and example.py for usage examples.
