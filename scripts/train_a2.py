#!/usr/bin/env python3
"""
Train A2 Data-Scaling Ablation Models.

Trains Hybrid H2 (LSTM encoder) on chronological training-prefix fractions
(50% and 75%) across random seeds (42, 43, 44) using the paper recipe:
  - AdamW optimizer, MSE loss, standardised target
  - Early stopping on validation loss
  - Saved to results/predictions/A2_H2_frac{0.50,0.75}_*.csv

Usage:
    PYTHONPATH=src python3 scripts/train_a2.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import pandas as pd
import yaml

# OpenMP / threading safety
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.data.dl_dataset import DriftDataset
from vio_drift.models.dl.hybrid import build_hybrid
from vio_drift.models.dl.hybrid_train import train_hybrid, predict_hybrid
from vio_drift.evaluation.metrics import score


def train_a2() -> None:
    config_file = ROOT / "configs" / "dl_config.yaml"
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    processed_dir = ROOT / "data" / "processed"
    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    fractions = [
        (0.50, "train_050", "A2_H2_frac0.50"),
        (0.75, "train_075", "A2_H2_frac0.75"),
    ]
    seeds = [42, 43, 44]
    folds = ["leave_MH_01_easy", "leave_MH_02_easy"]

    for frac, sub_dir, prefix in fractions:
        artifact_dir = processed_dir / "ablations_dl" / sub_dir
        print(f"\n--- Training A2: Fraction {frac*100:.0f}% ({sub_dir}) ---")

        for seed in seeds:
            for fold in folds:
                suffix = f"_seed{seed}" if seed != 42 else ""
                out_path = pred_dir / f"{prefix}{suffix}_{fold}.csv"

                print(f"  [Seed {seed}] Training on {fold}...")
                train_ds = DriftDataset(
                    processed_dir=processed_dir,
                    artifact_dir=artifact_dir,
                    fold_id=fold,
                    split="train",
                    seed=seed,
                )
                val_ds = DriftDataset(
                    processed_dir=processed_dir,
                    artifact_dir=artifact_dir,
                    fold_id=fold,
                    split="validation",
                    seed=seed,
                )
                test_ds = DriftDataset(
                    processed_dir=processed_dir,
                    artifact_dir=artifact_dir,
                    fold_id=fold,
                    split="test",
                    seed=seed,
                )

                model = build_hybrid("H2", imu_encoder="lstm", dropout=cfg.get("dropout", 0.3))
                res = train_hybrid(
                    model,
                    train_ds,
                    val_ds,
                    epochs=cfg.get("epochs", 40),
                    batch_size=cfg.get("batch_size", 32),
                    learning_rate=cfg.get("learning_rate", 1e-3),
                    weight_decay=cfg.get("weight_decay", 1e-4),
                    seed=seed,
                    device="cpu",
                    num_workers=0,
                    early_stopping=True,
                    patience=15,
                    min_delta=1e-4,
                    grad_clip=1.0,
                    min_epochs=0,
                )

                preds = predict_hybrid(res.model, test_ds, batch_size=32, device="cpu")
                metrics = score(preds.y_true, preds.y_pred)
                print(f"    -> Test MAE: {metrics['mae']*100:.2f} cm, R2: {metrics['r2']:.4f}")

                df = pd.DataFrame({
                    "seq_id": preds.seq_id,
                    "window_id": preds.window_id,
                    "y_true": preds.y_true,
                    "y_pred": preds.y_pred,
                })
                df.to_csv(out_path, index=False)
                print(f"    Saved: {out_path.name}")


if __name__ == "__main__":
    train_a2()
