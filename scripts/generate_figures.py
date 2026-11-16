#!/usr/bin/env python3
"""Generate all R5 figures from authoritative ablation and trajectory CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DL = root / "figures" / "dl"
FIGURES_TRAJ = root / "figures" / "trajectories"
FIGURES_DL.mkdir(parents=True, exist_ok=True)
FIGURES_TRAJ.mkdir(parents=True, exist_ok=True)

RESULTS = root / "results"

# ── colour palette ──────────────────────────────────────────────────────────
COL = {
    "Phase 1 (ML / Physics)": "#2196F3",
    "DL (IMU-only)":          "#FF9800",
    "Hybrid":                  "#4CAF50",
    "DL (Visual-only)":       "#9C27B0",
}


def get_lasso_mae(df: pd.DataFrame) -> float:
    """Retrieve Phase 1 Lasso MAE dynamically from E0 row."""
    e0_row = df[df["experiment_id"] == "E0"]
    if not e0_row.empty:
        return float(e0_row["mae_mean_cm"].iloc[0])
    return 2.5044


# ── 1. E0-E7 bar chart ──────────────────────────────────────────────────────
def plot_e0_e7() -> None:
    df = pd.read_csv(RESULTS / "ablations" / "experiments_e0_e7.csv")
    lasso_mae = get_lasso_mae(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [COL.get(f, "#9E9E9E") for f in df["family"]]
    bars = ax.bar(
        df["experiment_id"],
        df["mae_mean_cm"],
        yerr=df["mae_std_cm"],
        capsize=4,
        color=colors,
        edgecolor="k",
        linewidth=0.6,
        zorder=3,
    )
    ax.axhline(
        lasso_mae,
        color="#E53935",
        linestyle="--",
        linewidth=1.4,
        label=f"Phase 1 Lasso {lasso_mae:.2f} cm",
    )
    for bar, mae, std in zip(bars, df["mae_mean_cm"], df["mae_std_cm"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.15,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    patches = [mpatches.Patch(color=v, label=k) for k, v in COL.items()]
    ax.legend(
        handles=patches + [plt.Line2D([0], [0], color="#E53935", linestyle="--", label=f"Lasso {lasso_mae:.2f} cm")],
        fontsize=8,
        loc="upper right",
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("E0–E7: Experiment Comparison (MAE ± std across seeds 42, 43, 44)")
    ax.set_ylim(0, df["mae_mean_cm"].max() * 1.25)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    out = FIGURES_DL / "ablation_table_e0_e7.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── 2. Seed spread A4 ───────────────────────────────────────────────────────
def plot_seed_spread() -> None:
    df = pd.read_csv(RESULTS / "ablations" / "seeds_a4.csv")
    e0_df = pd.read_csv(RESULTS / "ablations" / "experiments_e0_e7.csv")
    lasso_mae = get_lasso_mae(e0_df)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["mae_mean_cm"],
        yerr=df["mae_std_cm"],
        capsize=5,
        color="#5C6BC0",
        edgecolor="k",
        linewidth=0.6,
        zorder=3,
    )
    ax.axhline(lasso_mae, color="#E53935", linestyle="--", linewidth=1.4, label=f"Lasso {lasso_mae:.2f} cm")
    for bar, mae, std in zip(bars, df["mae_mean_cm"], df["mae_std_cm"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.1,
            f"{mae:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_id"], rotation=15, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("A4: Seed Spread — MAE Mean ± Std (seeds 42, 43, 44)")
    ax.set_ylim(0, df["mae_mean_cm"].max() * 1.25)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGURES_DL / "seed_spread_a4.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── 3. Regularization A1 ────────────────────────────────────────────────────
def plot_regularization() -> None:
    df = pd.read_csv(RESULTS / "ablations" / "regularization_a1.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))

    if "variation" in df.columns:
        labels = [str(v).replace("_", "\n") for v in df["variation"]]
    else:
        labels = [f"d={r.dropout}\nwd={r.weight_decay}" for _, r in df.iterrows()]

    bars = ax.bar(
        x,
        df["mae_mean_cm"],
        yerr=df.get("mae_std_cm", 0.0),
        capsize=4,
        color="#26A69A",
        edgecolor="k",
        linewidth=0.6,
        zorder=3,
    )
    for bar, v in zip(bars, df["mae_mean_cm"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Regularisation variant")
    ax.set_ylabel("Validation MAE (cm)")
    ax.set_title("A1: Regularisation Ablation (CNN1D / E1)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    out = FIGURES_DL / "regularization_a1.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── 4. Data scaling A2 ──────────────────────────────────────────────────────
def plot_data_scaling() -> None:
    df = pd.read_csv(RESULTS / "ablations" / "data_scaling_a2.csv")
    e0_df = pd.read_csv(RESULTS / "ablations" / "experiments_e0_e7.csv")
    lasso_mae = get_lasso_mae(e0_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(
        df["fraction"] * 100,
        df["mae_mean_cm"],
        yerr=df["mae_std_cm"],
        marker="o",
        color="#1976D2",
        linewidth=1.8,
        capsize=4,
    )
    ax1.axhline(lasso_mae, color="#E53935", linestyle="--", linewidth=1.2, label=f"Lasso {lasso_mae:.2f}")
    ax1.set_xlabel("Training data prefix (%)")
    ax1.set_ylabel("MAE (cm)")
    ax1.set_title("A2: MAE vs Training Data Amount")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.4)

    ax2.plot(df["fraction"] * 100, df["r2_mean"], marker="s", color="#388E3C", linewidth=1.8)
    ax2.set_xlabel("Training data prefix (%)")
    ax2.set_ylabel("R²")
    ax2.set_title("A2: R² vs Training Data Amount")
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    out = FIGURES_DL / "data_scaling_a2.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── 5. Segments A5 ──────────────────────────────────────────────────────────
def plot_segments() -> None:
    df = pd.read_csv(RESULTS / "ablations" / "segments_a5.csv")
    seqs = df["flight"].unique()

    fig, axes = plt.subplots(1, len(seqs), figsize=(12, 5), sharey=False)
    if len(seqs) == 1:
        axes = [axes]

    for ax, seq in zip(axes, seqs):
        sub = df[df["flight"] == seq]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(
            x - w / 2,
            sub["takeoff_mae_cm"],
            w,
            label="Take-off (first 20%)",
            color="#EF5350",
            edgecolor="k",
            linewidth=0.5,
        )
        ax.bar(
            x + w / 2,
            sub["steady_mae_cm"],
            w,
            label="Steady (rest 80%)",
            color="#42A5F5",
            edgecolor="k",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=20, ha="right", fontsize=8)
        ax.set_title(seq)
        ax.set_ylabel("MAE (cm)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("A5: Take-off vs Steady-state MAE Decomposition", fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DL / "segments_a5.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── 6. Trajectory plots ─────────────────────────────────────────────────────
def plot_trajectory(seq_name: str) -> None:
    csv = RESULTS / "trajectories" / f"trajectory_{seq_name}.csv"
    if not csv.exists():
        print(f"Skipping {seq_name}: {csv} not found")
        return

    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XY path
    ax = axes[0]
    ax.plot(df["true_x"], df["true_y"], "g-", linewidth=2.0, label="Ground truth", zorder=4)
    ax.plot(df["raw_x"], df["raw_y"], "r--", linewidth=1.2, alpha=0.7, label="Dead-reckoned (raw)", zorder=3)
    ax.plot(df["corr_x"], df["corr_y"], "b-", linewidth=1.5, label="Corrected (E4vec)", zorder=4)
    ax.scatter([df["true_x"].iloc[0]], [df["true_y"].iloc[0]], c="g", s=80, marker="^", label="Start", zorder=5)
    ax.scatter([df["true_x"].iloc[-1]], [df["true_y"].iloc[-1]], c="g", s=80, marker="s", label="End", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{seq_name}: Reconstructed XY Path")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    ax.set_aspect("equal")

    # ATE over steps
    ax2 = axes[1]
    ax2.plot(df["step"], df["ate_raw"], "r-", linewidth=1.5, label="Raw Dead-Reckoning", alpha=0.8)
    ax2.plot(df["step"], df["ate_corr"], "b-", linewidth=1.5, label="Corrected Path", alpha=0.8)
    ax2.set_xlabel("Window step index (2.0 s stride)")
    ax2.set_ylabel("ATE (m)")
    ax2.set_title(f"{seq_name}: Cumulative Absolute Trajectory Error")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.4)

    final_raw = df["ate_raw"].iloc[-1]
    final_corr = df["ate_corr"].iloc[-1]
    ax2.text(
        0.98,
        0.95,
        f"Final ATE raw={final_raw:.2f} m\nFinal ATE corr={final_corr:.2f} m",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    plt.suptitle(f"Trajectory Correction — {seq_name}", fontweight="bold")
    plt.tight_layout()
    out = FIGURES_TRAJ / f"trajectory_{seq_name}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_ate_combined() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors_raw = ["#E53935", "#FB8C00"]
    colors_corr = ["#1E88E5", "#43A047"]

    for i, seq in enumerate(["MH_01_easy", "MH_02_easy"]):
        csv = RESULTS / "trajectories" / f"trajectory_{seq}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            ax.plot(
                df["step"],
                df["ate_raw"],
                color=colors_raw[i],
                linewidth=1.4,
                linestyle="--",
                label=f"{seq} Raw Dead-Reckoning",
            )
            ax.plot(
                df["step"],
                df["ate_corr"],
                color=colors_corr[i],
                linewidth=1.8,
                label=f"{seq} Corrected (E4vec)",
            )

    ax.set_xlabel("Window step index (2.0 s non-overlapping stride)")
    ax.set_ylabel("ATE (m)")
    ax.set_title("Absolute Trajectory Error over Time — Before vs After Drift Correction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    out = FIGURES_TRAJ / "ate_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating R5 figures from authoritative results...")
    plot_e0_e7()
    plot_seed_spread()
    plot_regularization()
    plot_data_scaling()
    plot_segments()
    plot_trajectory("MH_01_easy")
    plot_trajectory("MH_02_easy")
    plot_ate_combined()
    print("All figures successfully generated.")
