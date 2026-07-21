"""Data utilities for loading, preprocessing, and splitting datasets."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def find_label_column(columns):
    """Find a column that looks like a label column."""
    for col in columns:
        if 'label' in col.lower().strip():
            return col
    return None

def check_dataset_exists(dataset_path):
    """Check if the dataset exists and print helpful message if not."""
    if not os.path.exists(dataset_path):
        print("\n" + "="*60)
        print("❌ ERROR: Dataset not found!")
        print("="*60)
        print(f"\nExpected path: {dataset_path}\n")
        print("To generate the dataset, run:")
        print("  python -m scripts.run_combiner\n")
        print("="*60 + "\n")
        return False
    return True


def load_and_preprocess(dataset_path, scaler=None, sample_size=None):
    """
    Load and preprocess the dataset.

    Args:
        dataset_path : Path to CSV file
        scaler       : Optional pre-fitted StandardScaler (for inference)
        sample_size  : Optional row limit for quick testing

    Returns:
        tuple: (X_scaled, y, scaler, feature_names)
    """
    print(f"\n📂 Loading: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"   Shape: {df.shape}")

    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"   Sampled: {df.shape[0]} rows")

    df.dropna(inplace=True)

    label_col = find_label_column(df.columns)
    if not label_col:
        raise ValueError("❌ No Label column found. Run run_combiner first!")

    # Binary encode: BENIGN=0, ATTACK=1
    df["binary_label"] = df[label_col].str.strip().str.upper().apply(
        lambda x: 0 if x == "BENIGN" else 1
    )
    print(f"   BENIGN : {(df['binary_label']==0).sum():,}")
    print(f"   ATTACK : {(df['binary_label']==1).sum():,}")

    X = df.drop(columns=[label_col, "binary_label"], errors="ignore")

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=[np.number])
    y = df["binary_label"].values

    # Clean inf and NaN
    print(f"   Cleaning data...")
    X_array = np.array(X, dtype=np.float64)
    X_array[np.isinf(X_array)] = np.nan

    for col_idx in range(X_array.shape[1]):
        col_data = X_array[:, col_idx]
        col_mean = np.nanmean(col_data)
        if np.isnan(col_mean):
            col_mean = 0.0
        X_array[np.isnan(col_data), col_idx] = col_mean

    for col_idx in range(X_array.shape[1]):
        col_data  = X_array[:, col_idx]
        large_mask = np.abs(col_data) > 1e10
        if np.any(large_mask):
            col_mean = np.mean(col_data[~large_mask]) if np.any(~large_mask) else 0.0
            X_array[large_mask, col_idx] = col_mean

    X            = pd.DataFrame(X_array, columns=X.columns)
    feature_names = list(X.columns)

    if scaler is None:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    print(f"   Features: {X_scaled.shape[1]}")
    return X_scaled, y, scaler, feature_names


class DataSplitter:
    """
    Splits data across federated clients.
    Supports both IID and non-IID (Dirichlet) partitioning.
    """

    @staticmethod
    def split_for_clients(X, y, num_clients, alpha=0.5, random_state=42):
        """
        Non-IID Dirichlet partition across federated clients.

        Each class is distributed across clients using a Dirichlet
        distribution parameterized by alpha:
            alpha=0.1  → very non-IID (highly skewed per client)
            alpha=0.5  → moderately non-IID (recommended)
            alpha=10.0 → nearly IID

        This simulates realistic FL deployments where each network
        node observes a different mix of benign and attack traffic.

        Each client chunk is further split 80/20 into local
        train and validation subsets.

        Args:
            X           : Feature matrix
            y           : Binary labels (0=BENIGN, 1=ATTACK)
            num_clients : Number of federated clients
            alpha       : Dirichlet concentration parameter
            random_state: Reproducibility seed

        Returns:
            list of (client_id, X_train, y_train, X_val, y_val)
        """
        print(f"\n👥 Non-IID Dirichlet split "
              f"(alpha={alpha}) across {num_clients} clients...")

        np.random.seed(random_state)

        # Separate indices by class
        class_indices = {}
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            np.random.shuffle(idx)
            class_indices[cls] = idx

        # Assign samples to clients via Dirichlet proportions
        client_indices = [[] for _ in range(num_clients)]

        for cls, indices in class_indices.items():
            proportions = np.random.dirichlet([alpha] * num_clients)
            splits      = (proportions * len(indices)).astype(int)
            # Fix rounding so all samples are assigned
            splits[-1]  = len(indices) - splits[:-1].sum()

            current = 0
            for i, count in enumerate(splits):
                client_indices[i].extend(
                    indices[current:current + count].tolist()
                )
                current += count

        # Build per-client train/val splits
        client_splits = []
        for i, indices in enumerate(client_indices):
            indices = np.array(indices)
            np.random.shuffle(indices)

            X_c, y_c = X[indices], y[indices]

            # Each client: 80% local train / 20% local validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_c, y_c, test_size=0.2, random_state=random_state,
                stratify=y_c if len(np.unique(y_c)) > 1 else None
            )

            benign = (y_tr == 0).sum()
            attack = (y_tr == 1).sum()
            ratio  = attack / len(y_tr) if len(y_tr) > 0 else 0

            print(f"   Client {i+1}: {len(X_tr):>6} train | "
                  f"{len(X_val):>5} val | "
                  f"BENIGN:{benign:>6} ATTACK:{attack:>6} | "
                  f"attack_ratio:{ratio:.2f}")

            client_splits.append((i+1, X_tr, y_tr, X_val, y_val))

        # Report heterogeneity
        ratios = [(y_tr == 1).mean() for _, _, y_tr, _, _ in client_splits]
        std    = np.std(ratios)
        label  = "non-IID ✅" if std > 0.05 else "nearly IID ⚠️"
        print(f"\n   Attack ratios : {[round(r, 2) for r in ratios]}")
        print(f"   Std deviation : {std:.4f} → {label}")

        return client_splits
