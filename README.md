# TabKD: Tabular Knowledge Distillation through Interaction Diversity of Learned Feature Bins

A framework for data-free knowledge distillation that systematically covers feature interactions using combinatorial testing principles.

## Overview

TabKD extracts knowledge from tabular models (Neural Networks, XGBoost, Random Forest, TabTransformer) without access to the original training data. It uses:

- **Dynamic Bin Learning**: Learns feature discretization aligned with teacher decision boundaries
- **Interaction Diversity Loss**: Ensures generated samples cover pairwise feature combinations
- **Two-Phase Training**: Phase 1 learns bins + warms up generator; Phase 2 performs knowledge distillation

## Requirements

```bash
pip install torch numpy scikit-learn xgboost
```

## Project Structure

```
├── KD_Div_W3.py          # Main training script
├── models.py             # Model architectures (Teacher, Student, Generator, BinLearner)
├── losses.py             # Loss functions (Interaction Diversity, Variance-based Bin Loss)
├── datasets.py           # Dataset loading utilities
├── visualization.py      # Report generation
├── ablation.py           # Ablation study script
└── reports/              # Output directory for logs and reports
    ├── neural/
    ├── xgboost/
    ├── randomforest/
    └── tabtransformer/
```

## Usage

### Basic Command

```bash
python KD_Div_W3.py --teacher <TEACHER_TYPE> --dataset <DATASET_NAME>
```

### Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--teacher` | `neural`, `xgboost`, `randomforest`, `tabtransformer` | `neural` | Teacher model type |
| `--dataset` | `adult`, `credit`, `breast_cancer`, `mushroom` | `adult` | Dataset to use |

### Examples

```bash
# Neural network teacher on Adult dataset
python KD_Div_W3.py --teacher neural --dataset adult

# XGBoost teacher on Credit dataset
python KD_Div_W3.py --teacher xgboost --dataset credit

# Random Forest teacher on Breast Cancer dataset
python KD_Div_W3.py --teacher randomforest --dataset breast_cancer

# TabTransformer teacher on Mushroom dataset
python KD_Div_W3.py --teacher tabtransformer --dataset mushroom
```

### Run All Configurations

```bash
for teacher in neural xgboost randomforest tabtransformer; do
    for dataset in adult credit breast_cancer mushroom; do
        python KD_Div_W3.py --teacher $teacher --dataset $dataset
    done
done
```

## Output

Each run produces:

1. **Training Log**: `reports/<teacher>/training_log_<dataset>_<timestamp>.txt`
2. **Loss Report**: `reports/<teacher>/loss_report_<dataset>.txt`
3. **Bin Boundaries**: `reports/<teacher>/bin_boundaries_<dataset>.txt`
4. **Variance Evolution**: `reports/<teacher>/variance_evolution_<dataset>.txt`
5. **Model Checkpoint**: `training_history_<dataset>.pt`

## Key Hyperparameters

Located at the top of `KD_Div_W3.py`:

```python
BATCH_SIZE = 128
EPOCHS = 400           # Phase 2 epochs
NUM_BINS = 8           # Bins per feature
LATENT_DIM = 32        # Generator noise dimension
WARMUP_EPOCHS = 30     # Student warmup epochs

# Loss weights
LAMBDA_COV = 3.0       # Interaction diversity weight
LAMBDA_HARD = 8.0      # Adversarial hardness weight
```

## Ablation Study

Run ablation experiments:

```bash
python ablation.py
```

## Notes

- **GPU**: Automatically uses CUDA if available
- **TabTransformer**: Trains on CPU to avoid GPU OOM, then moves to GPU for inference
- **Temperature Schedules**: Automatically adjusted per teacher type (XGBoost uses softer temperatures)