"""Plotting utilities for FL experiment results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Colab
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Plot saved: {path}")


def plot_confusion_matrix(cm, model_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    _save(fig, save_path)


def plot_roc_curve(y_true, y_probs, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0,1], [0,1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    _save(fig, save_path)


def plot_pr_curve(y_true, y_probs, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2,
            label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="lower left")
    _save(fig, save_path)


def plot_training_progress(history, model_name, save_path):
    """Plot accuracy, AUC, F1, and loss across FL rounds."""
    rounds = list(range(1, len(history["accuracy"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"FL Training Progress — {model_name}", fontsize=14)

    metrics = [
        ("accuracy", "Accuracy",  "steelblue"),
        ("auc",      "ROC-AUC",   "darkorange"),
        ("f1",       "F1 Score",  "green"),
        ("loss",     "BCE Loss",  "red"),
    ]

    for ax, (key, label, color) in zip(axes.flat, metrics):
        if history.get(key):
            ax.plot(rounds, history[key], marker="o",
                    color=color, linewidth=2)
            ax.set_title(label)
            ax.set_xlabel("Round")
            ax.set_ylabel(label)
            ax.set_xticks(rounds)
            ax.grid(True, alpha=0.3)
            # Annotate final value
            ax.annotate(
                f"{history[key][-1]:.4f}",
                xy=(rounds[-1], history[key][-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=9, color=color
            )

    plt.tight_layout()
    _save(fig, save_path)


def plot_client_distribution(client_splits, model_name, save_path):
    """Bar chart showing BENIGN vs ATTACK per client."""
    client_ids, benign_counts, attack_counts = [], [], []

    for cid, _, y_tr, _, _ in client_splits:
        client_ids.append(f"Client {cid}")
        benign_counts.append((y_tr == 0).sum())
        attack_counts.append((y_tr == 1).sum())

    x   = np.arange(len(client_ids))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - w/2, benign_counts, w, label="BENIGN", color="steelblue")
    ax.bar(x + w/2, attack_counts, w, label="ATTACK", color="darkorange")

    ax.set_xlabel("Client")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Non-IID Client Distribution — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, save_path)
