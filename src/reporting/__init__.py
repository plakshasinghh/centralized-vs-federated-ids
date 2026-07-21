"""Result schema and metrics tracking for standardized evaluation."""

import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np


class ExperimentResult:
    """Standardized result container for any model run."""

    def __init__(self, model_name, training_mode, run_id=None, timestamp=None):
        self.model_name    = model_name
        self.training_mode = training_mode
        self.run_id        = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp     = timestamp or datetime.now().isoformat()
        self.params        = {}
        self.accuracy      = None
        self.auc           = None
        self.f1            = None
        self.precision     = None
        self.recall        = None
        self.tn = self.fp = self.fn = self.tp = None
        self.round_metrics = []

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def set_metrics(self, accuracy, auc, f1, precision, recall):
        self.accuracy  = float(accuracy)
        self.auc       = float(auc)
        self.f1        = float(f1)
        self.precision = float(precision)
        self.recall    = float(recall)
        return self

    def set_confusion_matrix(self, tn, fp, fn, tp):
        self.tn = int(tn)
        self.fp = int(fp)
        self.fn = int(fn)
        self.tp = int(tp)
        return self

    def add_round_metrics(self, round_num, accuracy, auc, f1):
        self.round_metrics.append({
            "round":    int(round_num),
            "accuracy": float(accuracy),
            "auc":      float(auc),
            "f1":       float(f1)
        })
        return self

    def to_dict(self):
        return {
            "model_name":    self.model_name,
            "run_id":        self.run_id,
            "timestamp":     self.timestamp,
            "training_mode": self.training_mode,
            "params":        self.params,
            "final_metrics": {
                "accuracy":  self.accuracy,
                "auc":       self.auc,
                "f1":        self.f1,
                "precision": self.precision,
                "recall":    self.recall
            },
            "confusion_matrix": {
                "tn": self.tn, "fp": self.fp,
                "fn": self.fn, "tp": self.tp
            }
        }

    def to_dict_with_rounds(self):
        data = self.to_dict()
        data["round_metrics"] = self.round_metrics
        return data


class ResultWriter:
    """Writes experiment results to disk."""

    @staticmethod
    def save_metrics_json(result, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict_with_rounds(), f, indent=2)
        print(f"   ✅ Metrics saved: {output_path}")

    @staticmethod
    def save_round_metrics_csv(result, output_path):
        if not result.round_metrics:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "accuracy", "auc", "f1"]
            )
            writer.writeheader()
            writer.writerows(result.round_metrics)
        print(f"   ✅ Round metrics saved: {output_path}")
