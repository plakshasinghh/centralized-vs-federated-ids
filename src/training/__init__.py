"""
Training and evaluation utilities.
OPTIMIZED VERSION — key changes from original:
  - train_one_epoch accepts pos_weight for per-client class imbalance correction
  - Uses BCEWithLogitsLoss (numerically stable) instead of BCELoss + Sigmoid
  - evaluate_model applies sigmoid to raw logits before thresholding
  - Added per-round FPR and DR metrics for IDS-specific evaluation
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score
)


def compute_pos_weight(y):
    """
    Compute pos_weight for BCEWithLogitsLoss from a label array.

    BCEWithLogitsLoss(pos_weight=w) scales the loss for positive (attack)
    samples by w = n_negatives / n_positives. This corrects for the
    per-client class imbalance introduced by Dirichlet splitting.

    Args:
        y: numpy array of binary labels (0=BENIGN, 1=ATTACK)
    Returns:
        torch.Tensor scalar (safe: returns 1.0 if all one class)
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def make_dataloader(X, y, batch_size=64, shuffle=True):
    """
    Create a PyTorch DataLoader from numpy arrays.

    Args:
        X         : Feature matrix (numpy array)
        y         : Labels (numpy array)
        batch_size: Batch size
        shuffle   : Whether to shuffle data
    Returns:
        DataLoader
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle
    )


def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    """
    Train model for one epoch.

    Args:
        model    : PyTorch model (outputs raw logits)
        loader   : DataLoader for training
        optimizer: Optimizer
        criterion: Loss function (BCEWithLogitsLoss recommended)
        device   : torch device
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze()

        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, device="cpu"):
    """
    Evaluate model on a dataset.
    Applies sigmoid to raw logits before thresholding at 0.5.

    Args:
        model : PyTorch model (outputs raw logits)
        loader: DataLoader for evaluation
        device: torch device
    Returns:
        tuple: (accuracy, auc, f1, y_true_array, y_pred_array)
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch).squeeze().cpu()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            # Apply sigmoid to convert logits → probabilities
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    return acc, auc, f1, all_labels, all_preds


def evaluate_model_full(model, loader, device="cpu"):
    """
    Full evaluation returning IDS-specific metrics.
    Includes FPR and Detection Rate (DR/Recall) alongside standard metrics.

    Returns:
        dict with keys: accuracy, auc, f1, precision, recall,
                        fpr, dr, y_true, y_pred, y_prob
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch).squeeze().cpu()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.4).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)  # same as DR

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate

    return {
        "accuracy" : acc,
        "auc"      : auc,
        "f1"       : f1,
        "precision": prec,
        "recall"   : rec,
        "fpr"      : fpr,
        "dr"       : dr,
        "y_true"   : y_true,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
    }


def print_summary_table(acc, auc, f1, cm, fpr=None, dr=None):
    """Print a formatted summary table of model performance."""
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "="*58)
    print("         FINAL MODEL PERFORMANCE SUMMARY")
    print("="*58)
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    if dr  is not None: print(f"  Detection Rate  : {dr:.4f}  (recall on attacks)")
    if fpr is not None: print(f"  False Pos. Rate : {fpr:.4f}  (benign flagged)")
    print(f"  True Positives  : {tp:>7}   (attacks correctly caught)")
    print(f"  True Negatives  : {tn:>7}   (benign correctly allowed)")
    print(f"  False Positives : {fp:>7}   (benign flagged as attack)")
    print(f"  False Negatives : {fn:>7}   (attacks missed ⚠️)")
    print("="*58)
