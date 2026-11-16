#!/usr/bin/env bash
# run_phase2.sh — Reproduce all Phase 2 (R5) results in one command.
# Usage: bash run_phase2.sh [--skip-tests]
#
# Outputs:
#   results/ablations/      — all ablation CSV files
#   results/trajectories/   — trajectory CSVs
#   figures/dl/             — ablation plots (PNG)
#   figures/trajectories/   — trajectory correction plots (PNG)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python3
SKIP_TESTS=0

for arg in "$@"; do
  [[ "$arg" == "--skip-tests" ]] && SKIP_TESTS=1
done

echo "============================================================"
echo "  Drift-Sense Phase 2 Reproduction Script"
echo "  Working directory: $(pwd)"
echo "  Python: $($PYTHON --version)"
echo "============================================================"

# ── Step 0: Sanity check ───────────────────────────────────────────────────
echo ""
echo "[Step 0] Checking data files..."
for f in data/processed/splits.json data/processed/physics_leave_MH_01_easy.csv data/processed/physics_leave_MH_02_easy.csv; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file $f not found. Run Phase 1 pipeline first." >&2
    exit 1
  fi
done
echo "  Data OK (1,312 windows across MH_01_easy and MH_02_easy)."

# ── Step 1: Run all experiments and ablations ──────────────────────────────
echo ""
echo "[Step 1] Running ablation experiments and validation checks..."
$PYTHON scripts/collect_final_results.py --check
echo "  Ablations done. CSVs written to results/ablations/"

# ── Step 2: Run trajectory correction ─────────────────────────────────────
echo ""
echo "[Step 2] Verifying trajectory dead-reckoning correction (E4vec)..."
PYTHONPATH=src $PYTHON -c "
from vio_drift.correction.correct_trajectory import run_sequence_correction
for seq in ['MH_01_easy', 'MH_02_easy']:
    comp, traj_df = run_sequence_correction(seq)
    print(f'  {seq}: Endpoint MAE {comp[\"mae_before\"]*100:.2f} -> {comp[\"mae_after\"]*100:.2f} cm ({comp[\"drift_reduction_pct\"]:.1f}% reduction), ATE reduction: {comp[\"ate_reduction_pct\"]:.1f}%')
"
echo "  Trajectories verified in results/trajectories/"

# ── Step 3: Generate figures ───────────────────────────────────────────────
echo ""
echo "[Step 3] Generating figures..."
$PYTHON scripts/generate_figures.py
echo "  Figures written to figures/dl/ and figures/trajectories/"

# ── Step 4: Run test suite ─────────────────────────────────────────────────
if [[ $SKIP_TESTS -eq 0 ]]; then
  echo ""
  echo "[Step 4] Running R5 test suite..."
  $PYTHON -m pytest tests/test_dl_eval.py tests/test_correction.py -v --tb=short
  echo "  Tests passed."
else
  echo ""
  echo "[Step 4] Skipping tests (--skip-tests flag set)."
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2 reproduction complete."
echo ""
echo "  Key results:"
$PYTHON - <<'PYEOF'
import pandas as pd
from pathlib import Path
try:
    df = pd.read_csv('results/ablations/experiments_e0_e7.csv')
    best = df.loc[df['mae_mean_cm'].idxmin()]
    lasso = df[df['experiment_id']=='E0'].iloc[0]
    print(f"    Phase 1 Lasso (E0):  MAE = {lasso['mae_mean_cm']:.2f} cm, R² = {lasso['r2_mean']:.4f}")
    print(f"    Best DL ({best['experiment_id']}):        MAE = {best['mae_mean_cm']:.2f} cm ± {best['mae_std_cm']:.2f} cm, R² = {best['r2_mean']:.4f}  ← {best['name']}")
except Exception as e:
    print(f"    (Could not load results: {e})")
try:
    dc = pd.read_csv('results/ablations/correction_c1.csv')
    for _, r in dc.iterrows():
        print(f"    {r['sequence']}: Endpoint {r['endpoint_mae_before_cm']:.1f} -> {r['endpoint_mae_after_cm']:.1f} cm ({r['endpoint_reduction_pct']:.1f}%), Final ATE {r['final_drift_raw_m']:.2f}m -> {r['final_drift_corr_m']:.2f}m ({r['ate_reduction_pct']:.1f}% reduction)")
except Exception as e:
    print(f"    (Correction CSV not found: {e})")
PYEOF
echo ""
echo "  Notebooks: notebooks/11_ablation_studies.ipynb"
echo "             notebooks/12_trajectory_correction.ipynb"
echo "  Decisions: docs/decisions_log.md"
echo "============================================================"
