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
    for fn in (plots.plot_pred_vs_actual, plots.plot_error_over_time, plots.plot_residuals):
        fig = fn(preds, "err_mag_m")
        assert fig.axes
        plt.close(fig)


def test_feature_importance_plot_keeps_top_n():
    imp = pd.Series(np.arange(30, dtype=float), index=[f"f{i}" for i in range(30)])
    fig = plots.plot_feature_importance(imp, "t", "x", top_n=10)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 10 and labels[-1] == "f29"
    plt.close(fig)
