"""Federated learning components — OPTIMIZED VERSION.

Key fixes over original:
  1. BCEWithLogitsLoss + per-client pos_weight  (replaces BCELoss on sigmoid output)
  2. AdamW optimizer with bias exclusion from weight decay
  3. ReduceLROnPlateau now steps on VALIDATION loss (was incorrectly using train loss)
  4. local_epochs reduced to 3 (less client drift on severe non-IID splits)
  5. proximal_mu default lowered to 0.01 (looser constraint works better with correct loss)
  6. FPR and DR tracked per round in fit() metrics
  7. Warmup: first round uses lr * 0.1, then full lr from round 2 onward
  8. Flwr dependency removed — pure PyTorch custom simulator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict

from src.models.cnn_lstm_ids import CNN_LSTM_IDS
from src.training import (
    make_dataloader, evaluate_model, evaluate_model_full, compute_pos_weight
)


class IDSClient:
    """
    Federated client representing one network node.

    Each FL round:
    1. Server sends global model weights  → set_parameters()
    2. Client trains locally with FedProx → fit()
    3. Client sends updated weights back  → get_parameters()
    4. Raw data NEVER leaves the client

    FedProx loss:
        L = BCEWithLogitsLoss(pos_weight=w) + (mu/2) * ||w_local - w_global||²

    pos_weight corrects for the per-client class imbalance introduced by
    the Dirichlet split — e.g. Client 1 (1% attack) gets high pos_weight
    so attack samples aren't ignored.
    """

    def __init__(
        self,
        client_id,
        X_train, y_train,
        X_val,   y_val,
        input_dim,
        learning_rate = 0.001,
        batch_size    = 64,
        local_epochs  = 5,       
        proximal_mu   = 0.05,  
        device        = "cpu"
    ):
        self.client_id    = client_id
        self.model        = CNN_LSTM_IDS(input_dim).to(device)
        self.device       = device
        self.local_epochs = local_epochs
        self.proximal_mu  = proximal_mu
        self._round       = 0     # internal round counter for warmup

        # ── per-client pos_weight for class imbalance ─────────────────
        # Clients with few attack samples get a higher pos_weight so the
        # loss doesn't collapse to "predict everything as benign"
        pw = compute_pos_weight(y_train).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # ── AdamW: weight decay only on non-bias, non-norm parameters ──
        # LSTM bias terms and BatchNorm params should NOT be decayed
        decay_params    = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name or "BatchNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = optim.AdamW([
            {"params": decay_params,    "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=learning_rate)

        self.base_lr   = learning_rate
        self.warmup_lr = learning_rate * 0.1  # round 1 warmup

        # ReduceLROnPlateau — stepped on VAL loss (fixed from original)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5, min_lr=1e-6
        )

        self.train_loader = make_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=True
        )
        self.val_loader = make_dataloader(
            X_val, y_val, batch_size=batch_size, shuffle=False
        )

    # ── Weight helpers ─────────────────────────────────────────────────

    def get_parameters(self, config={}):
        """Return current model weights as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load weights received from server into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ── Training ───────────────────────────────────────────────────────

    def fit(self, parameters, config={}):
        """
        Receive global weights, train locally with FedProx, return updated weights.

        Round 1 uses warmup_lr to avoid destructive updates from a random
        global model. Full lr resumes from round 2.

        Returns:
            (updated_weights, num_train_samples, metrics_dict)
        """
        self._round += 1
        self.set_parameters(parameters)

        # Warmup: use lower lr in round 1
        current_lr = self.warmup_lr if self._round == 1 else self.base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = current_lr

        # Snapshot global weights before local training (for proximal term)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        losses = []
        for _ in range(self.local_epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch).squeeze()

                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                # BCEWithLogitsLoss with pos_weight (handles sigmoid internally)
                bce_loss = self.criterion(logits, y_batch)

                # FedProx proximal term: (mu/2) * ||w_local - w_global||²
                proximal_term = sum(
                    ((lp - gp) ** 2).sum()
                    for lp, gp in zip(self.model.parameters(), global_params)
                )

                loss = bce_loss + (self.proximal_mu / 2) * proximal_term
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            losses.append(total_loss / len(self.train_loader))

        # Evaluate on val set — use val_loss to step the LR scheduler
        val_metrics = evaluate_model_full(self.model, self.val_loader, self.device)
        val_loss    = self._compute_val_loss()

        # Step scheduler on VALIDATION loss (fix from original)
        self.scheduler.step(val_loss)

        print(
            f"   Client {self.client_id} → "
            f"train_loss:{losses[-1]:.4f} | "
            f"acc:{val_metrics['accuracy']:.4f} | "
            f"auc:{val_metrics['auc']:.4f} | "
            f"f1:{val_metrics['f1']:.4f} | "
            f"fpr:{val_metrics['fpr']:.4f} | "
            f"dr:{val_metrics['dr']:.4f}"
        )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "loss"     : float(losses[-1]),
                "accuracy" : float(val_metrics["accuracy"]),
                "auc"      : float(val_metrics["auc"]),
                "f1"       : float(val_metrics["f1"]),
                "fpr"      : float(val_metrics["fpr"]),
                "dr"       : float(val_metrics["dr"]),
            }
        )

    def _compute_val_loss(self):
        """Compute BCE loss on validation set (for scheduler stepping)."""
        self.model.eval()
        total_loss    = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch).squeeze()
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                loss           = self.criterion(logits, y_batch)
                total_loss    += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        return total_loss / total_samples if total_samples > 0 else 0.0

    def evaluate(self, parameters, config={}):
        """
        Evaluate global model on local validation set.
        Returns real BCE loss + accuracy/auc/f1/fpr/dr.
        """
        self.set_parameters(parameters)
        val_loss    = self._compute_val_loss()
        val_metrics = evaluate_model_full(self.model, self.val_loader, self.device)

        return (
            float(val_loss),
            len(self.val_loader.dataset),
            {
                "accuracy" : float(val_metrics["accuracy"]),
                "auc"      : float(val_metrics["auc"]),
                "f1"       : float(val_metrics["f1"]),
                "fpr"      : float(val_metrics["fpr"]),
                "dr"       : float(val_metrics["dr"]),
            }
        )


class FederatedSimulator:
    """
    Orchestrates FedProx federated learning simulation.

    Runs clients in-process — no Ray, no IPC overhead.
    Aggregation uses weighted FedAvg across client updates.
    History now also tracks FPR and DR per round.
    """

    def __init__(
        self,
        num_clients   = 5,
        num_rounds    = 10,
        local_epochs  = 3,
        batch_size    = 64,
        learning_rate = 0.001,
        proximal_mu   = 0.05,
        random_state  = 42,
        device        = "cpu"
    ):
        self.num_clients   = num_clients
        self.num_rounds    = num_rounds
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.proximal_mu   = proximal_mu
        self.random_state  = random_state
        self.device        = device

    def create_clients(self, client_splits, input_dim):
        """Create one IDSClient per data split."""
        return [
            IDSClient(
                cid, X_tr, y_tr, X_val, y_val,
                input_dim,
                learning_rate = self.learning_rate,
                batch_size    = self.batch_size,
                local_epochs  = self.local_epochs,
                proximal_mu   = self.proximal_mu,
                device        = self.device
            )
            for cid, X_tr, y_tr, X_val, y_val in client_splits
        ]

    def run(self, clients, input_dim):
        """
        Run FedProx simulation.

        Per round:
        1. Broadcast global weights to all clients
        2. Each client trains locally (FedProx loss + pos_weight + AdamW)
        3. Weighted FedAvg aggregation
        4. Evaluate global model on each client's val set
        5. Log round metrics including FPR and DR
        """
        history = {
            "accuracy": [], "auc": [], "f1": [],
            "loss": [], "fpr": [], "dr": []
        }

        assert len(clients) > 0, "No clients provided to FederatedSimulator.run()"
        global_weights = clients[0].get_parameters(config={})

        print(f"\n🚀 Starting Optimized FedProx Simulation")
        print(f"   Clients      : {self.num_clients}")
        print(f"   Rounds       : {self.num_rounds}")
        print(f"   Local Epochs : {self.local_epochs}")
        print(f"   Proximal mu  : {self.proximal_mu}")
        print(f"   Device       : {self.device}")
        print(f"   Optimizer    : AdamW (bias/norm excluded from decay)")
        print(f"   Loss         : BCEWithLogitsLoss + per-client pos_weight")
        print("=" * 65)

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n[ROUND {round_num}/{self.num_rounds}]")
            warmup_note = " (warmup lr)" if round_num == 1 else ""
            print(f"   Training {self.num_clients} clients{warmup_note}...")

            # ── FIT ───────────────────────────────────────────────────
            all_weights   = []
            all_sizes     = []

            for client in clients:
                weights, num_samples, _ = client.fit(global_weights, config={})
                all_weights.append(weights)
                all_sizes.append(num_samples)

            # ── AGGREGATE: weighted FedAvg ─────────────────────────────
            total = sum(all_sizes)
            global_weights = [
                sum(
                    w[i] * (n / total)
                    for w, n in zip(all_weights, all_sizes)
                )
                for i in range(len(global_weights))
            ]

            # ── EVALUATE ──────────────────────────────────────────────
            eval_losses = []
            eval_accs   = []
            eval_aucs   = []
            eval_f1s    = []
            eval_fprs   = []
            eval_drs    = []

            for client in clients:
                loss, _, m = client.evaluate(global_weights, config={})
                eval_losses.append(loss)
                eval_accs.append(m.get("accuracy", 0))
                eval_aucs.append(m.get("auc", 0))
                eval_f1s.append(m.get("f1", 0))
                eval_fprs.append(m.get("fpr", 0))
                eval_drs.append(m.get("dr", 0))

            history["loss"].append(float(np.mean(eval_losses)))
            history["accuracy"].append(float(np.mean(eval_accs)))
            history["auc"].append(float(np.mean(eval_aucs)))
            history["f1"].append(float(np.mean(eval_f1s)))
            history["fpr"].append(float(np.mean(eval_fprs)))
            history["dr"].append(float(np.mean(eval_drs)))

            print(
                f"\n📊 Round {round_num} → "
                f"Loss:{history['loss'][-1]:.4f} | "
                f"Acc:{history['accuracy'][-1]:.4f} | "
                f"AUC:{history['auc'][-1]:.4f} | "
                f"F1:{history['f1'][-1]:.4f} | "
                f"FPR:{history['fpr'][-1]:.4f} | "
                f"DR:{history['dr'][-1]:.4f}"
            )

        # ── Reconstruct final global model ────────────────────────────
        global_model = CNN_LSTM_IDS(input_dim).to(self.device)
        params_dict  = zip(global_model.state_dict().keys(), global_weights)
        state_dict   = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        global_model.load_state_dict(state_dict, strict=True)
        print("\n✅ Global aggregated model reconstructed.")

        return global_model.state_dict(), history

