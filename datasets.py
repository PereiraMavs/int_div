"""
Dataset loading and preprocessing functionality for ABLE.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import rdt
import torch

###############################################################################
# DATASET CONFIGURATION
###############################################################################
DATASET_CONFIG = {
    'credit': {
        'id': 350,
        'target': 'default',
        'preprocess': lambda y: y,  # If already 0/1
        'num_classes': 2
    },
    'adult': {
        'id': 2,
        'target': 'income',
        'preprocess': lambda y: np.where(
            y['income'].str.startswith('>50K') | y['income'].str.startswith('>=50K'),
            1, 0
        ),
        'num_classes': 2
    },
    'breast_cancer': {
        'id': 17,
        'target': 'Diagnosis',
        'preprocess': lambda y: np.where(
            y['Diagnosis'].str.startswith('M') | y['Diagnosis'].str.startswith('m'),
            1, 0
        ),
        'num_classes': 2
    },
    'mushroom': {
        'id': 73,
        'target': 'poisonous',
        'preprocess': lambda y: np.where(
            y['poisonous'].str.startswith('p') | y['poisonous'].str.startswith('P'),
            1, 0
        ),
        'num_classes': 2
    },
    'car': {
        'id': 19,
        'target': 'class',
        'preprocess': lambda y: pd.Categorical(y['class'], 
            categories=['unacc', 'acc', 'good', 'vgood']).codes,
        'num_classes': 4
    },
    'covertype': {
        'id': 31,
        'target': 'Cover_Type',
        'preprocess': lambda y: y - 1,  # Convert from 1-7 to 0-6
        'num_classes': 7
    }
}

def load_dataset(dataset_name, test_size=0.2, random_state=42):
    """Load and preprocess dataset using UCI ML Repo + rdt"""
    config = DATASET_CONFIG[dataset_name]
    ds = fetch_ucirepo(id=config['id'])
    X = ds.data.features
    y_raw = ds.data.targets
    y_processed = config['preprocess'](y_raw)
    y_processed = np.array(y_processed).astype(int).ravel()

    # Preprocess features
    ht = rdt.HyperTransformer()
    ht.detect_initial_config(data=X)
    X_transformed = ht.fit_transform(X)
    col_names = X_transformed.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    X_min = torch.tensor(X_scaled.min(axis=0), dtype=torch.float32)
    X_max = torch.tensor(X_scaled.max(axis=0), dtype=torch.float32)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_processed,
        test_size=test_size,
        stratify=y_processed,
        random_state=random_state
    )
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"Dataset {dataset_name}: {config['num_classes']} classes, {len(np.unique(y_train))} unique labels")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Input dimensions: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, X_min, X_max, col_names, scaler, ht, config['num_classes'] 