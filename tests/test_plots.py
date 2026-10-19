"""
Smoke tests: every Part D plot renders from a toy predictions table.

Run with: pytest tests/test_plots.py -v
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation import plots


def _toy_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for seq in ["A", "B"]:
        y = rng.normal(0.18, 0.01, 20)
        for model in ["mean", "ridge", "random_forest", "svr"]:
            rows.append(pd.DataFrame({
                "fold_id": f"leave_{seq}", "seq_id": seq, "window_id": range(20),
                "t_start_ns": np.arange(20) * 250_000_000, "model": model,
                "y_true": y, "y_pred": y + rng.normal(0, 0.005, 20),
            }))
    return pd.concat(rows, ignore_index=True)


def test_prediction_plots_render():
    preds = _toy_predictions()
    figs = [plots.plot_error_over_time(preds, "err_mag_m", seq) for seq in ["A", "B"]]
    for model in ["mean", "ridge", "random_forest", "svr"]:
        figs.append(plots.plot_pred_vs_actual(preds, "err_mag_m", model))
        figs.append(plots.plot_residuals(preds, "err_mag_m", model))
    for fig in figs:
        assert len(fig.axes) == 1   # one plot per figure
        plt.close(fig)


def test_feature_importance_plot_keeps_top_n():
    imp = pd.Series(np.arange(30, dtype=float), index=[f"f{i}" for i in range(30)])
    fig = plots.plot_feature_importance(imp, "t", "x", top_n=10)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 10 and labels[-1] == "f29"
    plt.close(fig)


def test_every_model_gets_its_own_colour():
    models = ["mean", "linear", "ridge", "random_forest", "svr"]
    colors = plots._model_colors(models)
    assert len(set(colors.values())) == len(models)
