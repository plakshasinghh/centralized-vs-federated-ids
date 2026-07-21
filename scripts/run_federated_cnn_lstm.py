"""
CLI script to run federated CNN+LSTM training.

Usage:
    python -m scripts.run_federated_cnn_lstm
"""
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_lstm_ids import CNN_LSTM_IDS
from src.federated import FederatedSimulator
from src.training import (
    evaluate_model,
    evaluate_model_full,
    make_dataloader,
    print_summary_table
)

from src.data      import check_dataset_exists, load_and_preprocess, DataSplitter
from src.reporting import ExperimentResult, ResultWriter
from src.reporting.plots import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_training_progress, plot_client_distribution
)


# ── Config ────────────────────────────────────────────────────────────────────

def get_config():
    return {
        "paths": {
            "dataset"      : "C:/Users/yashd/Downloads/IDS/final_balanced_dataset.csv",
            "artifacts_dir": "artifacts",
            "reports_dir"  : "reports"
        },
        "federated": {
            "num_clients"  : 5,
            "num_rounds"   : 20,
            "local_epochs" : 5,       
            "batch_size"   : 512,     # larger for GPU parallelism
            "learning_rate": 0.001,
            "proximal_mu"  : 0.05,    # loosened — correct loss scaling
            "alpha"        : 1.0,
            "test_split"   : 0.20,
            "random_state" : 42
        },
        "plots": {
            "training_progress"  : True,
            "confusion_matrix"   : True,
            "roc_curve"          : True,
            "pr_curve"           : True,
            "client_distribution": True
        }
    }


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    log_dir  = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fl_gpu_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return run_id


# ── GPU-specific DataLoader override ─────────────────────────────────────────
# The base make_dataloader in training.py doesn't set pin_memory.
# We override here for GPU performance.

from torch.utils.data import DataLoader, TensorDataset

def make_dataloader_gpu(X, y, batch_size=512, shuffle=True):
    """
    GPU-optimized DataLoader.
    pin_memory=True speeds up CPU→GPU tensor transfers significantly.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size  = batch_size,
        shuffle     = shuffle,
        pin_memory  = True,    # page-locked memory for faster GPU transfer
        num_workers = 0        # 0 = main process only (safe on Windows/Colab)
    )


# ── Monkey-patch GPU DataLoader into FederatedSimulator ──────────────────────
# We subclass FederatedSimulator and override create_clients to inject
# GPU DataLoaders into each IDSClient after creation.



class GPUFederatedSimulator(FederatedSimulator):
    """FederatedSimulator with GPU-optimized DataLoaders."""

    def create_clients(self, client_splits, input_dim):
        """Create clients then swap their DataLoaders for GPU-pinned ones."""
        clients = super().create_clients(client_splits, input_dim)
        for client, (_, X_tr, y_tr, X_val, y_val) in zip(clients, client_splits):
            client.train_loader = make_dataloader_gpu(
                X_tr, y_tr, batch_size=self.batch_size, shuffle=True
            )
            client.val_loader = make_dataloader_gpu(
                X_val, y_val, batch_size=self.batch_size, shuffle=False
            )
        return clients


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_federated_learning():
    print("\n" + "="*70)
    print("🚀 FEDERATED LEARNING CNN+LSTM IDS  [GPU VERSION — OPTIMIZED]")
    print("="*70)

    config       = get_config()
    run_id       = setup_logging(config["paths"]["reports_dir"] + "/logs")
    cfg_fl       = config["federated"]
    random_state = cfg_fl["random_state"]

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # ── Device selection ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(random_state)
        # Reproducible but still fast
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True   # auto-tune for fixed input size
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n⚙️  Device : {device} — {gpu_name}")
        print(f"⚙️  VRAM   : {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  CUDA not available — falling back to CPU")
        print(f"   Consider using run_federated_cpu.py for CPU-tuned settings.")

    paths         = config["paths"]
    artifacts_dir = Path(paths["artifacts_dir"])
    reports_dir   = Path(paths["reports_dir"])
    figures_dir   = reports_dir / "figures" / "cnn_lstm_gpu"
    results_dir   = reports_dir / "results"  / "cnn_lstm_gpu"

    for d in [artifacts_dir, results_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Dataset ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("📂 DATASET LOADING")
    print("="*70)

    if not check_dataset_exists(paths["dataset"]):
        print("\n   Run: python -m scripts.run_combiner")
        return False

    X, y, scaler, feature_names = load_and_preprocess(paths["dataset"])
    input_dim = X.shape[1]
    print(f"   Input dim: {input_dim} features")

    # ── STEP 2: Train/test split ──────────────────────────────────────
    print("\n" + "="*70)
    print("📊 DATA SPLITTING  (80/20 train/test)")
    print("="*70)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y,
        test_size    = cfg_fl["test_split"],
        random_state = random_state,
        stratify     = y
    )
    print(f"   Train : {len(X_train_all):,} samples")
    print(f"   Test  : {len(X_test):,} samples")

    # ── STEP 3: Non-IID client split ─────────────────────────────────
    print("\n" + "="*70)
    print("👥 CLIENT DATA DISTRIBUTION  (Non-IID Dirichlet)")
    print("="*70)

    client_splits = DataSplitter.split_for_clients(
        X_train_all, y_train_all,
        num_clients  = cfg_fl["num_clients"],
        alpha        = cfg_fl["alpha"],
        random_state = random_state
    )

    if config["plots"]["client_distribution"]:
        plot_client_distribution(
            client_splits, "CNN+LSTM FL (GPU)",
            figures_dir / "client_distribution.png"
        )

    # ── STEP 4: Federated learning ────────────────────────────────────
    print("\n" + "="*70)
    print("🌐 FEDERATED LEARNING SIMULATION  (FedProx — Optimized)")
    print("="*70)

    simulator = GPUFederatedSimulator(
        num_clients   = cfg_fl["num_clients"],
        num_rounds    = cfg_fl["num_rounds"],
        local_epochs  = cfg_fl["local_epochs"],
        batch_size    = cfg_fl["batch_size"],
        learning_rate = cfg_fl["learning_rate"],
        proximal_mu   = cfg_fl["proximal_mu"],
        random_state  = random_state,
        device        = str(device)
    )

    clients = simulator.create_clients(client_splits, input_dim)

    t0 = time.time()
    final_model_state, history = simulator.run(clients, input_dim)
    elapsed = time.time() - t0

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n⚙️  Peak GPU memory: {peak_mb:.0f} MB")
    print(f"⏱️  Total training time: {elapsed/60:.1f} min")

    # ── STEP 5: Training progress plot ───────────────────────────────
    if config["plots"]["training_progress"] and history["accuracy"]:
        plot_training_progress(
            history, "CNN+LSTM FL (GPU)",
            figures_dir / "training_progress.png"
        )

    # ── STEP 6: Final evaluation ──────────────────────────────────────
    print("\n" + "="*70)
    print("🧪 FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*70)

    final_model = CNN_LSTM_IDS(input_dim).to(device)
    final_model.load_state_dict(final_model_state)

    # Use GPU-optimized DataLoader for test set too
    test_loader = make_dataloader_gpu(
        X_test, y_test,
        batch_size = cfg_fl["batch_size"],
        shuffle    = False
    )

    metrics = evaluate_model_full(final_model, test_loader, device)
    cm      = confusion_matrix(metrics["y_true"], metrics["y_pred"])
    print_summary_table(
        metrics["accuracy"], metrics["auc"], metrics["f1"], cm,
        fpr=metrics["fpr"], dr=metrics["dr"]
    )
    print(f"\n{classification_report(metrics['y_true'], metrics['y_pred'], target_names=['BENIGN','ATTACK'])}")

    # ── STEP 7: Plots ─────────────────────────────────────────────────
    if config["plots"]["confusion_matrix"]:
        plot_confusion_matrix(cm, "CNN+LSTM FL (GPU)", figures_dir / "confusion_matrix.png")

    all_probs = metrics["y_prob"]

    if config["plots"]["roc_curve"]:
        plot_roc_curve(metrics["y_true"], all_probs, "CNN+LSTM FL (GPU)", figures_dir / "roc_curve.png")

    if config["plots"]["pr_curve"]:
        plot_pr_curve(metrics["y_true"], all_probs, "CNN+LSTM FL (GPU)", figures_dir / "pr_curve.png")

    # ── STEP 8: Save artifacts ────────────────────────────────────────
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    model_path = artifacts_dir / "fl_cnn_lstm_gpu_model.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"   ✅ Model  : {model_path}")

    import joblib
    scaler_path = artifacts_dir / "fl_cnn_lstm_gpu_scaler.pkl"
    joblib.dump({"scaler": scaler, "features": feature_names, "input_dim": input_dim}, scaler_path)
    print(f"   ✅ Scaler : {scaler_path}")

    result = ExperimentResult("CNN+LSTM FL GPU", "federated", run_id)
    result.set_params(
        num_clients  = cfg_fl["num_clients"],
        num_rounds   = cfg_fl["num_rounds"],
        local_epochs = cfg_fl["local_epochs"],
        batch_size   = cfg_fl["batch_size"],
        proximal_mu  = cfg_fl["proximal_mu"],
        alpha        = cfg_fl["alpha"]
    )
    result.set_metrics(
        metrics["accuracy"], metrics["auc"], metrics["f1"],
        metrics["precision"], metrics["recall"]
    )
    tn, fp, fn, tp = cm.ravel()
    result.set_confusion_matrix(tn, fp, fn, tp)
    for i, acc_val in enumerate(history["accuracy"]):
        result.add_round_metrics(i+1, acc_val, history["auc"][i], history["f1"][i])

    ResultWriter.save_metrics_json(result, results_dir / "metrics_summary.json")
    ResultWriter.save_round_metrics_csv(result, results_dir / "round_metrics.csv")

    import csv
    ext_csv = results_dir / "round_metrics_extended.csv"
    with open(ext_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round","accuracy","auc","f1","fpr","dr","loss"])
        w.writeheader()
        for i in range(len(history["accuracy"])):
            w.writerow({
                "round"   : i+1,
                "accuracy": history["accuracy"][i],
                "auc"     : history["auc"][i],
                "f1"      : history["f1"][i],
                "fpr"     : history["fpr"][i],
                "dr"      : history["dr"][i],
                "loss"    : history["loss"][i],
            })
    print(f"   ✅ Extended metrics: {ext_csv}")

    print("\n" + "="*70)
    print("✨ GPU TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Artifacts : {artifacts_dir}")
    print(f"📁 Plots      : {figures_dir}")
    print(f"📁 Metrics    : {results_dir}")
    print(f"⏱️  Wall time  : {elapsed/60:.1f} min\n")

    return True


if __name__ == "__main__":
    success = run_federated_learning()
    sys.exit(0 if success else 1)



 
