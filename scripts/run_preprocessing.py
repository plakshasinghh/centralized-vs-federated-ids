"""
CLI script to preprocess the combined/balanced dataset and save the
transformed result to disk.

This is the step that was previously "invisible" — `src.data.load_and_preprocess()`
does the actual cleaning/scaling, but until now it only ran in-memory inside
`run_federated_cnn_lstm.py`, so there was no file you could open and inspect
after preprocessing. This script runs that same transformation and writes
the result out.

What it does to the raw combined CSV:
  1. Strip column names, find the label column
  2. Binary-encode the label (BENIGN=0, everything else=1)
  3. Label-encode any leftover categorical (non-numeric) feature columns
  4. Replace inf with NaN, then fill NaN with the column mean
  5. Clip absurdly large outlier values (> 1e10) to the column mean
  6. Fit a StandardScaler and scale all features
  7. Save: the scaled dataset as CSV, a small preview as .xlsx, and the
     fitted scaler + feature list (needed to preprocess new data the same
     way at inference time) as a .pkl

Usage:
    python -m scripts.run_preprocessing
    python -m scripts.run_preprocessing --input final_balanced_dataset.csv --output data/preprocessed_dataset.csv
    python -m scripts.run_preprocessing --preview-rows 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd

from src.data import load_and_preprocess, check_dataset_exists


def run_preprocessing(
    input_path="final_balanced_dataset.csv",
    output_csv="data/preprocessed_dataset.csv",
    scaler_path="artifacts/preprocessing_scaler.pkl",
    preview_path="data/sample/preprocessed_preview.xlsx",
    preview_rows=10,
):
    print("\n" + "=" * 60)
    print("🧹 DATASET PREPROCESSING")
    print("=" * 60)

    if not check_dataset_exists(input_path):
        return False

    X_scaled, y, scaler, feature_names = load_and_preprocess(input_path)

    out_df = pd.DataFrame(X_scaled, columns=feature_names)
    out_df["Label"] = y  # 0 = BENIGN, 1 = ATTACK

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅ Preprocessed dataset saved: {output_csv}  (shape={out_df.shape})")

    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "features": feature_names}, scaler_path)
    print(f"✅ Fitted scaler saved: {scaler_path}")
    print("   (reuse this to preprocess new/live data the exact same way)")

    if preview_rows > 0:
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.head(preview_rows).to_excel(preview_path, index=False)
        print(f"✅ Preview ({preview_rows} rows) saved: {preview_path}")

    print("\n" + "=" * 60)
    print("✨ Preprocessing complete!\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, encode, and scale the combined dataset")
    parser.add_argument("--input", default="final_balanced_dataset.csv", help="Combined dataset CSV (output of run_combiner)")
    parser.add_argument("--output", default="data/preprocessed_dataset.csv", help="Where to save the preprocessed CSV")
    parser.add_argument("--scaler-output", default="artifacts/preprocessing_scaler.pkl", help="Where to save the fitted scaler")
    parser.add_argument("--preview-rows", type=int, default=10, help="Rows to save as a quick-look .xlsx preview (0 to skip)")
    args = parser.parse_args()

    success = run_preprocessing(
        input_path=args.input,
        output_csv=args.output,
        scaler_path=args.scaler_output,
        preview_rows=args.preview_rows,
    )
    sys.exit(0 if success else 1)
