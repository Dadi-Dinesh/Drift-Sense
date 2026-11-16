#!/usr/bin/env python3
"""Create R5 Jupyter notebooks (11_ablation_studies.ipynb and 12_trajectory_correction.ipynb)."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb_dir = root / "notebooks"
nb_dir.mkdir(exist_ok=True)


def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.2"},
        },
        "cells": cells,
    }


# ── Notebook 11: Ablation Studies ──────────────────────────────────────────
nb11_cells = [
    cell("# Notebook 11: Ablation Studies\n\n**R5 — Evaluation, Ablations and Drift Correction (Technical Validation)**\n\nThis notebook presents the systematic ablation study for the Drift-Sense VIO project.\nAll evaluations use seeds **42, 43, 44** across **1,312 windows** (MH_01_easy: 720, MH_02_easy: 592)\nunder leave-one-sequence-out cross-validation.", "markdown"),

    cell([
        "import sys\n",
        "from pathlib import Path\n",
        "root = Path('.').resolve().parent\n",
        "sys.path.insert(0, str(root / 'src'))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as mpatches\n",
        "\n",
        "RESULTS = root / 'results' / 'ablations'\n",
        "FIGURES = root / 'figures' / 'dl'\n",
        "FIGURES.mkdir(parents=True, exist_ok=True)\n",
        "print('Working root:', root)\n",
        "pd.set_option('display.float_format', '{:.4f}'.format)\n",
    ]),

    cell("## Section 1: Phase 1 Comparability Check\n\nWe first verify that the new harness reproduces the Phase 1 benchmark numbers bit-for-bit:\n- **Mean baseline:** 4.53 cm\n- **Tuned Lasso (E0):** 2.50 cm MAE (0.02504 m), R² = 0.76\n- **Calibrated Physics prior:** R² = 0.82\n\nEnforced by unit test `tests/test_dl_eval.py`.", "markdown"),

    cell([
        "from vio_drift.evaluation.dl_eval import reproduce_phase1_benchmarks\n",
        "benchmarks = reproduce_phase1_benchmarks()\n",
        "print('=== Phase 1 Benchmark Reproduction ===')\n",
        "for _, row in benchmarks[benchmarks['fold_id'] == 'mean'].iterrows():\n",
        "    print(f\"  {row['model']:15s}: MAE = {row['mae']*100:.2f} cm, R² = {row['r2']:.4f}\")\n",
    ]),

    cell("## Section 2: Core Experiments E0–E7\n\nComparison of all model families against the Phase 1 Lasso baseline over seeds 42, 43, 44.\n- **E0**: Phase 1 Lasso baseline (104 hand-crafted features)\n- **E1**: 1D-CNN on IMU (standardized target: 2.60 cm)\n- **E2**: LSTM on IMU (standardized target: 3.69 cm)\n- **E3**: Hybrid H1 (IMU + features + physics: 2.64 cm with warm-up)\n- **E4**: Hybrid H2 (Physics prior + learned residual: **2.15 cm**, beats Lasso!)\n- **E4cnn**: Hybrid H2 with CNN encoder (**2.13 cm**)\n- **E5**: Camera CNN (4.42 cm)\n- **E6**: Hybrid H3 (IMU + Camera: 2.85 cm with warm-up)\n- **E7**: Best Overall Hybrid (2.18 cm)", "markdown"),

    cell([
        "df_e = pd.read_csv(RESULTS / 'experiments_e0_e7.csv')\n",
        "display_cols = ['experiment_id', 'name', 'family', 'mae_mean_cm', 'mae_std_cm', 'rmse_mean_cm', 'r2_mean']\n",
        "print(df_e[display_cols].to_string(index=False))\n",
    ]),

    cell([
        "lasso_mae = float(df_e[df_e['experiment_id']=='E0']['mae_mean_cm'].iloc[0])\n",
        "COL = {\n",
        "    'Phase 1 (ML / Physics)': '#2196F3',\n",
        "    'DL (IMU-only)': '#FF9800',\n",
        "    'Hybrid': '#4CAF50',\n",
        "    'DL (Visual-only)': '#9C27B0',\n",
        "}\n",
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "colors = [COL.get(f, '#9E9E9E') for f in df_e['family']]\n",
        "bars = ax.bar(df_e['experiment_id'], df_e['mae_mean_cm'],\n",
        "              yerr=df_e['mae_std_cm'], capsize=4,\n",
        "              color=colors, edgecolor='k', linewidth=0.6, zorder=3)\n",
        "ax.axhline(lasso_mae, color='#E53935', linestyle='--', linewidth=1.4,\n",
        "           label=f'Phase 1 Lasso baseline {lasso_mae:.2f} cm')\n",
        "for bar, mae, std in zip(bars, df_e['mae_mean_cm'], df_e['mae_std_cm']):\n",
        "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.1,\n",
        "            f'{mae:.2f}', ha='center', va='bottom', fontsize=8)\n",
        "patches = [mpatches.Patch(color=v, label=k) for k, v in COL.items()]\n",
        "ax.legend(handles=patches + [plt.Line2D([0], [0], color='#E53935', linestyle='--', label=f'Lasso {lasso_mae:.2f} cm')], fontsize=8)\n",
        "ax.set_xlabel('Experiment'); ax.set_ylabel('MAE (cm)')\n",
        "ax.set_title('E0–E7: Experiment Comparison (MAE ± std across seeds 42, 43, 44)')\n",
        "ax.set_ylim(0, df_e['mae_mean_cm'].max() * 1.25)\n",
        "ax.grid(axis='y', alpha=0.4, zorder=0)\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"\\nKey finding: Hybrid H2 (E4) achieves {df_e[df_e.experiment_id=='E4']['mae_mean_cm'].values[0]:.2f} cm, outperforming Lasso ({lasso_mae:.2f} cm).\")\n",
    ]),

    cell("## Section 3: A1 — Regularisation Ablation\n\nFactorial ablation on Dropout (0.0 vs 0.3) and Weight Decay (0.0 vs 1e-4) on CNN1D (E1).", "markdown"),

    cell([
        "df_a1 = pd.read_csv(RESULTS / 'regularization_a1.csv')\n",
        "print(df_a1.to_string(index=False))\n",
    ]),

    cell("## Section 4: A2 — Data Scaling\n\nMAE and R² as a function of chronological training data prefix (50%, 75%, 100%).", "markdown"),

    cell([
        "df_a2 = pd.read_csv(RESULTS / 'data_scaling_a2.csv')\n",
        "print(df_a2.to_string(index=False))\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))\n",
        "ax1.errorbar(df_a2['fraction']*100, df_a2['mae_mean_cm'], yerr=df_a2['mae_std_cm'],\n",
        "             marker='o', color='#1976D2', linewidth=1.8, capsize=4)\n",
        "ax1.axhline(lasso_mae, color='#E53935', linestyle='--', label=f'Lasso {lasso_mae:.2f}')\n",
        "ax1.set_xlabel('Training data prefix (%)'); ax1.set_ylabel('MAE (cm)')\n",
        "ax1.set_title('A2: MAE vs Training Size'); ax1.legend(fontsize=8); ax1.grid(alpha=0.4)\n",
        "ax2.plot(df_a2['fraction']*100, df_a2['r2_mean'], marker='s', color='#388E3C', linewidth=1.8)\n",
        "ax2.set_xlabel('Training data prefix (%)'); ax2.set_ylabel('R²')\n",
        "ax2.set_title('A2: R² vs Training Size'); ax2.grid(alpha=0.4)\n",
        "plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Section 5: A3 — Component Contributions\n\nDecomposing the incremental benefit of physics prior, hand-crafted features, and learned residual.", "markdown"),

    cell([
        "df_a3 = pd.read_csv(RESULTS / 'components_a3.csv')\n",
        "print(df_a3.to_string(index=False))\n",
    ]),

    cell("## Section 6: A4 — Seed Spread\n\nAssessing optimization stability across seeds 42, 43, 44.\nNotice how Hybrid H2 exhibits tight clustering (std = 0.01 cm), whereas unconstrained deep networks exhibit larger variance.", "markdown"),

    cell([
        "df_a4 = pd.read_csv(RESULTS / 'seeds_a4.csv')\n",
        "print(df_a4[['model_id', 'n_seeds', 'mae_mean_cm', 'mae_std_cm', 'mae_min_cm', 'mae_max_cm', 'r2_mean']].to_string(index=False))\n",
    ]),

    cell("## Section 7: A5 — Flight Segments (Take-off vs Steady Cruise)\n\nDecomposition into first-20% (take-off acceleration) vs remaining-80% (steady cruise).", "markdown"),

    cell([
        "df_a5 = pd.read_csv(RESULTS / 'segments_a5.csv')\n",
        "print(df_a5.to_string(index=False))\n",
    ]),

    cell("## Summary\n\n| Experiment | MAE (cm) | R² | Note |\n|---|---|---|---|\n| E0 Lasso | 2.50 | 0.76 | Phase 1 baseline |\n| E1 1D-CNN | 2.60 ± 0.12 | 0.72 | Standardized target |\n| E2 LSTM | 3.69 ± 0.06 | 0.31 | Standardized target |\n| E3 Hybrid H1 | 2.64 ± 0.03 | 0.73 | Warm-up min10 |\n| **E4 Hybrid H2 (LSTM)** | **2.15 ± 0.01** | **0.82** | **Beats Lasso ✓** |\n| **E4cnn Hybrid H2 (CNN)** | **2.13 ± 0.02** | **0.82** | **Best DL model ✓** |\n| E5 Camera | 4.42 ± 0.12 | 0.07 | Visual-only |\n| E6 Hybrid H3 | 2.85 ± 0.05 | 0.64 | IMU + Camera (warm-up) |\n| E7 Best Hybrid | 2.18 ± 0.06 | 0.81 | Validation selected |\n\n**Conclusion:** Hybrid H2 (physics prior + learned residual) is the top-performing model family, outperforming Phase 1 Lasso by over 14% while exhibiting minimal variance across random seeds.", "markdown"),
]


# ── Notebook 12: Trajectory Correction ─────────────────────────────────────
nb12_cells = [
    cell("# Notebook 12: Trajectory Drift Correction\n\n**R5 — Practical Payoff: Trajectory Correction via 3-Axis Error Subtraction**\n\nThis notebook demonstrates the navigational utility of learned drift predictions:\nwe subtract the predicted 3-axis error vector from each window's dead-reckoned endpoint\nand reconstruct continuous flight paths for **MH_01_easy** and **MH_02_easy**.", "markdown"),

    cell([
        "import sys\n",
        "from pathlib import Path\n",
        "root = Path('.').resolve().parent\n",
        "sys.path.insert(0, str(root / 'src'))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "TRAJ_DIR = root / 'results' / 'trajectories'\n",
        "CORR_CSV = root / 'results' / 'ablations' / 'correction_c1.csv'\n",
        "FIG_DIR = root / 'figures' / 'trajectories'\n",
        "print('Trajectories directory:', TRAJ_DIR)\n",
    ]),

    cell("## Section 1: Correction Summary Table (C1)\n\nComparison of endpoint MAE and cumulative Absolute Trajectory Error (ATE) before and after correction.", "markdown"),

    cell([
        "df_corr = pd.read_csv(CORR_CSV)\n",
        "cols = ['sequence', 'windows_count', 'endpoint_mae_before_cm', 'endpoint_mae_after_cm',\n",
        "        'endpoint_reduction_pct', 'cumulative_ate_raw_m', 'cumulative_ate_corr_m',\n",
        "        'final_drift_raw_m', 'final_drift_corr_m', 'ate_reduction_pct']\n",
        "print(df_corr[cols].to_string(index=False))\n",
    ]),

    cell("## Section 2: MH_01_easy Trajectory Correction\n\nPlotting the reconstructed ground-truth, raw dead-reckoning, and E4vec-corrected paths.", "markdown"),

    cell([
        "df1 = pd.read_csv(TRAJ_DIR / 'trajectory_MH_01_easy.csv')\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# XY path\n",
        "ax1.plot(df1['true_x'], df1['true_y'], 'g-', linewidth=2.0, label='Ground truth', zorder=4)\n",
        "ax1.plot(df1['raw_x'],  df1['raw_y'],  'r--', linewidth=1.2, alpha=0.7, label='Dead-reckoned (raw)', zorder=3)\n",
        "ax1.plot(df1['corr_x'], df1['corr_y'], 'b-',  linewidth=1.5, label='Corrected (E4vec)', zorder=4)\n",
        "ax1.scatter([df1['true_x'].iloc[0]], [df1['true_y'].iloc[0]], c='g', s=80, marker='^', label='Start', zorder=5)\n",
        "ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')\n",
        "ax1.set_title('MH_01_easy: XY Trajectory Path')\n",
        "ax1.legend(fontsize=9); ax1.grid(alpha=0.4); ax1.set_aspect('equal')\n",
        "\n",
        "# ATE vs step\n",
        "ax2.plot(df1['step'], df1['ate_raw'],  'r-', linewidth=1.5, label='Raw ATE',       alpha=0.8)\n",
        "ax2.plot(df1['step'], df1['ate_corr'], 'b-', linewidth=1.5, label='Corrected ATE', alpha=0.8)\n",
        "ax2.set_xlabel('Window step index (2.0 s stride)'); ax2.set_ylabel('ATE (m)')\n",
        "ax2.set_title('MH_01_easy: ATE vs Step')\n",
        "ax2.legend(fontsize=9); ax2.grid(alpha=0.4)\n",
        "plt.suptitle('MH_01_easy Trajectory Correction', fontweight='bold')\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"Final ATE raw={df1['ate_raw'].iloc[-1]:.2f} m -> corrected={df1['ate_corr'].iloc[-1]:.2f} m\")\n",
    ]),

    cell("## Section 3: MH_02_easy Trajectory Correction", "markdown"),

    cell([
        "df2 = pd.read_csv(TRAJ_DIR / 'trajectory_MH_02_easy.csv')\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "ax1.plot(df2['true_x'], df2['true_y'], 'g-', linewidth=2.0, label='Ground truth', zorder=4)\n",
        "ax1.plot(df2['raw_x'],  df2['raw_y'],  'r--', linewidth=1.2, alpha=0.7, label='Dead-reckoned (raw)', zorder=3)\n",
        "ax1.plot(df2['corr_x'], df2['corr_y'], 'b-',  linewidth=1.5, label='Corrected (E4vec)', zorder=4)\n",
        "ax1.scatter([df2['true_x'].iloc[0]], [df2['true_y'].iloc[0]], c='g', s=80, marker='^', zorder=5)\n",
        "ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')\n",
        "ax1.set_title('MH_02_easy: XY Trajectory Path')\n",
        "ax1.legend(fontsize=9); ax1.grid(alpha=0.4); ax1.set_aspect('equal')\n",
        "ax2.plot(df2['step'], df2['ate_raw'],  'r-', linewidth=1.5, label='Raw ATE',       alpha=0.8)\n",
        "ax2.plot(df2['step'], df2['ate_corr'], 'b-', linewidth=1.5, label='Corrected ATE', alpha=0.8)\n",
        "ax2.set_xlabel('Window step index (2.0 s stride)'); ax2.set_ylabel('ATE (m)')\n",
        "ax2.set_title('MH_02_easy: ATE vs Step')\n",
        "ax2.legend(fontsize=9); ax2.grid(alpha=0.4)\n",
        "plt.suptitle('MH_02_easy Trajectory Correction', fontweight='bold')\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"Final ATE raw={df2['ate_raw'].iloc[-1]:.2f} m -> corrected={df2['ate_corr'].iloc[-1]:.2f} m\")\n",
    ]),

    cell("## Section 4: Overlay Comparison — Cumulative ATE over Time", "markdown"),

    cell([
        "fig, ax = plt.subplots(figsize=(11, 5))\n",
        "ax.plot(df1['step'], df1['ate_raw'],  color='#E53935', linewidth=1.4, linestyle='--', label='MH_01 Raw Dead-Reckoning')\n",
        "ax.plot(df1['step'], df1['ate_corr'], color='#1E88E5', linewidth=1.8,                 label='MH_01 Corrected (E4vec)')\n",
        "ax.plot(df2['step'], df2['ate_raw'],  color='#FB8C00', linewidth=1.4, linestyle='--', label='MH_02 Raw Dead-Reckoning')\n",
        "ax.plot(df2['step'], df2['ate_corr'], color='#43A047', linewidth=1.8,                 label='MH_02 Corrected (E4vec)')\n",
        "ax.set_xlabel('Window step index (2.0 s non-overlapping stride)'); ax.set_ylabel('ATE (m)')\n",
        "ax.set_title('Cumulative Absolute Trajectory Error: Raw Dead-Reckoning vs E4vec Correction')\n",
        "ax.legend(fontsize=9); ax.grid(alpha=0.4); plt.tight_layout(); plt.show()\n",
    ]),

    cell(r"## Section 5: Correction Method and Quantitative Conclusion\n\n**Method:**\n1. Select non-overlapping 2.0 s windows (step = 8).\n2. Obtain 3-axis predicted drift vector $[\hat{e}_x, \hat{e}_y, \hat{e}_z]$ from the E4vec multi-task head.\n3. Subtract predicted displacement: $\mathbf{p}_{\text{corrected}} = \mathbf{p}_{\text{dead-reckoned}} - \hat{\mathbf{e}}$.\n4. Step-by-step path chaining for continuous trajectory evaluation.\n\n**Results:**\n- **MH_01_easy:** Endpoint MAE 121.5 cm $\to$ **6.54 cm** (94.6% reduction); Final drift 87.1 m $\to$ **1.49 m** (**98.1% ATE reduction**).\n- **MH_02_easy:** Endpoint MAE 120.0 cm $\to$ **5.37 cm** (95.5% reduction); Final drift 72.1 m $\to$ **0.83 m** (**98.5% ATE reduction**).\n\n**Verification:**\n- Unit test `test_correction.py` proves zero-prediction invariance, perfect-vector zero residual error, and >80% empirical reduction.", "markdown"),
]

# Write notebooks
nb11_path = nb_dir / "11_ablation_studies.ipynb"
nb12_path = nb_dir / "12_trajectory_correction.ipynb"

with open(nb11_path, "w") as f:
    json.dump(nb(nb11_cells), f, indent=1)
print(f"Written {nb11_path} ({nb11_path.stat().st_size} bytes)")

with open(nb12_path, "w") as f:
    json.dump(nb(nb12_cells), f, indent=1)
print(f"Written {nb12_path} ({nb12_path.stat().st_size} bytes)")
print("Notebook creation complete.")
