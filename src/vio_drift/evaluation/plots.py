"""
Figures for Part D. Every function takes the long predictions table written
by ``scripts/evaluate.py`` (one row per fold, window and model, with
``y_true`` / ``y_pred``) or an importance table, and returns a matplotlib
Figure. Saving is the caller's job, so notebooks can show the same figures.

Colours: the first three slots of a colour-blind-validated categorical
palette. In scatter/residual plots they mark the test *sequence*; in the
time plots they mark the *model*, with the mean baseline drawn as a grey
dashed reference and the ground truth in near-black ink.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BASELINE_GREY = "#8a8985"

MODEL_LABELS = {
    "mean": "Mean baseline",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "svr": "SVR (RBF)",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _seq_colors(predictions: pd.DataFrame) -> dict[str, str]:
    seqs = list(dict.fromkeys(predictions["seq_id"]))
    return {s: SERIES[i % len(SERIES)] for i, s in enumerate(seqs)}


def _model_colors(models: list[str]) -> dict[str, str]:
    colors, i = {}, 0
    for m in models:
        if m == "mean":
            colors[m] = BASELINE_GREY
        else:
            colors[m] = SERIES[i % len(SERIES)]
            i += 1
    return colors


def plot_pred_vs_actual(predictions: pd.DataFrame, target: str) -> Figure:
    """One panel per model: predicted vs true label, coloured by test sequence."""
    _style()
    models = list(dict.fromkeys(predictions["model"]))
    seq_colors = _seq_colors(predictions)
    lo = min(predictions["y_true"].min(), predictions["y_pred"].min())
    hi = max(predictions["y_true"].max(), predictions["y_pred"].max())
    pad = 0.05 * (hi - lo)

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4),
                             sharex=True, sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        g = predictions[predictions["model"] == model]
        for seq, color in seq_colors.items():
            s = g[g["seq_id"] == seq]
            ax.scatter(s["y_true"], s["y_pred"], s=10, alpha=0.5, color=color,
                       edgecolors="none", label=seq)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_MUTED,
                lw=1, ls="--", label="perfect")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel(f"true {target} (m)")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    np.atleast_1d(axes)[0].set_ylabel(f"predicted {target} (m)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), markerscale=2)
    fig.suptitle(f"Predicted vs actual — held-out sequence, {target}", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_error_over_time(predictions: pd.DataFrame, target: str) -> Figure:
    """One row per held-out sequence: true label and each model over flight time."""
    _style()
    seqs = list(dict.fromkeys(predictions["seq_id"]))
    models = list(dict.fromkeys(predictions["model"]))
    colors = _model_colors(models)

    fig, axes = plt.subplots(len(seqs), 1, figsize=(10, 2.4 * len(seqs)), sharex=False)
    for ax, seq in zip(np.atleast_1d(axes), seqs):
        g = predictions[predictions["seq_id"] == seq]
        truth = g[g["model"] == models[0]].sort_values("t_start_ns")
        t = (truth["t_start_ns"] - truth["t_start_ns"].iloc[0]) / 1e9
        ax.plot(t, truth["y_true"], color=INK, lw=1.2, label="true")
        for model in models:
            m = g[g["model"] == model].sort_values("t_start_ns")
            ax.plot(t, m["y_pred"].to_numpy(), color=colors[model],
                    lw=2 if model != "mean" else 1.2,
                    ls="--" if model == "mean" else "-",
                    label=MODEL_LABELS.get(model, model))
        ax.set_title(f"{seq} (held out)")
        ax.set_ylabel(f"{target} (m)")
    np.atleast_1d(axes)[-1].set_xlabel("window start (s)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_residuals(predictions: pd.DataFrame, target: str) -> Figure:
    """One panel per model: residual (true - predicted) vs prediction."""
    _style()
    models = list(dict.fromkeys(predictions["model"]))
    seq_colors = _seq_colors(predictions)

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2),
                             sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        g = predictions[predictions["model"] == model]
        for seq, color in seq_colors.items():
            s = g[g["seq_id"] == seq]
            ax.scatter(s["y_pred"], s["y_true"] - s["y_pred"], s=10, alpha=0.5,
                       color=color, edgecolors="none", label=seq)
        ax.axhline(0, color=INK_MUTED, lw=1, ls="--")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel(f"predicted {target} (m)")
    np.atleast_1d(axes)[0].set_ylabel("residual: true − predicted (m)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), markerscale=2)
    fig.suptitle(f"Residuals — held-out sequence, {target}", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_feature_importance(
    importance: pd.Series,
    title: str,
    xlabel: str,
    top_n: int = 20,
    error: pd.Series | None = None,
) -> Figure:
    """Horizontal bars for the ``top_n`` largest values in ``importance``."""
    _style()
    top = importance.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(top) + 1.2))
    xerr = error.reindex(top.index).to_numpy() if error is not None else None
    ax.barh(top.index, top.to_numpy(), color=SERIES[0], height=0.7,
            xerr=xerr, error_kw={"ecolor": INK_MUTED, "lw": 0.8, "capsize": 0})
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
