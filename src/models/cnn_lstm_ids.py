"""
CNN + LSTM + MLP Hybrid Model for Intrusion Detection.
OPTIMIZED VERSION — key changes from original:
  - Sigmoid REMOVED from output layer (use BCEWithLogitsLoss for numerical stability)
  - Raw logits are returned; caller applies threshold or sigmoid as needed
"""

import torch
import torch.nn as nn


class CNN_LSTM_IDS(nn.Module):
    """
    Hybrid CNN+LSTM+MLP model for binary network intrusion detection.

    Architecture:
        Input (batch, features)
            ↓
        unsqueeze → (batch, 1, features)
            ↓
        [Conv1D(32) → BatchNorm → ReLU]     CNN block 1
            ↓
        [Conv1D(64) → BatchNorm → ReLU]     CNN block 2
            ↓
        [MaxPool1D(2) → Dropout(0.3)]       halves feature dimension
            ↓
        permute → (batch, features//2, 64)  reformat for LSTM
            ↓
        [LSTM(128, layers=2)]               sequential modeling
            ↓
        last hidden state → (batch, 128)
            ↓
        [Dropout(0.3)]
            ↓
        [Linear(128→64) → ReLU → Dropout(0.2)]   MLP block 1
            ↓
        [Linear(64→32) → ReLU]                    MLP block 2
            ↓
        [Linear(32→1)]                            raw logit — NO sigmoid

    IMPORTANT: Output is a raw logit, not a probability.
        - During training : use BCEWithLogitsLoss (numerically stable)
        - During inference: apply torch.sigmoid(logit) to get probability
        - Threshold        : prob >= 0.5  →  ATTACK,  else BENIGN
    """

    def __init__(self, input_dim):
        super(CNN_LSTM_IDS, self).__init__()
        self.input_dim = input_dim

        # ── CNN Stage ──────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # ── LSTM Stage ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.lstm_dropout = nn.Dropout(0.3)

        # ── MLP Stage ──────────────────────────────────────────
        # No Sigmoid here — BCEWithLogitsLoss handles it internally
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
            # ← Sigmoid intentionally removed
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (batch_size, input_dim) — normalized feature vector
        Returns:
            (batch_size, 1) — raw logit (apply sigmoid for probability)
        """
        x = x.unsqueeze(1)           # (batch, 1, input_dim)
        x = self.cnn(x)              # (batch, 64, input_dim//2)

        x = x.permute(0, 2, 1)      # (batch, input_dim//2, 64)
        lstm_out, _ = self.lstm(x)   # (batch, input_dim//2, 128)

        x = lstm_out[:, -1, :]      # (batch, 128)
        x = self.lstm_dropout(x)

        x = self.mlp(x)             # (batch, 1) — raw logit
        return x


if __name__ == "__main__":
    model = CNN_LSTM_IDS(input_dim=78)
    dummy = torch.randn(4, 78)
    logits = model(dummy)
    probs  = torch.sigmoid(logits)
    print(f"✅ Logit shape : {logits.shape}")
    print(f"✅ Prob range  : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"✅ Parameters  : {sum(p.numel() for p in model.parameters()):,}")
