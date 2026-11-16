# R5 slides: Technical Validation, Ablations and Drift Correction (3 slides)

## Slide 1: Comparability & Benchmark Anchoring
- Figure: `figures/dl/ablation_table_e0_e7.png`
- **Rule 1:** Reproduce Phase 1 benchmarks before evaluating any deep network
  - Mean baseline: 4.53 cm ($R^2 = -0.06$)
  - Tuned Lasso (E0): **2.50 cm MAE**, $R^2 = 0.76$
  - Calibrated physics prior: $R^2 = 0.82$
- Enforced by automated regression tests (`test_dl_eval.py`)
- All models evaluated across 3 fixed random seeds (42, 43, 44) under identical leave-one-sequence-out folds (1,312 windows)

## Slide 2: Ablation Findings & Best Architecture
- Figure: `figures/dl/seed_spread_a4.png` and `figures/dl/data_scaling_a2.png`

| Model | Inputs | MAE (cm) | $R^2$ | Note |
|---|---|---|---|---|
| E0 Lasso | 104 Hand-crafted | 2.50 ± 0.00 | 0.76 | Phase 1 baseline |
| E1 1D-CNN (raw target) | Raw IMU (unscaled) | 13.87 | -8.55 | Severe underfit on offset |
| IMUcnn (std target) | Raw IMU (scaled) | 2.60 ± 0.12 | 0.72 | Target rescaling cures failure |
| E2 LSTM (raw target) | Raw IMU (unscaled) | 3.52 | 0.50 | Beats mean baseline (4.53 cm) |
| IMUlstm (std target) | Raw IMU (scaled) | 3.69 ± 0.06 | 0.31 | Lower variance, stable |
| E3_min10 Hybrid H1 | IMU + Feat + Phys | 2.64 ± 0.03 | 0.73 | Needs 10-epoch warm-up |
| **E4 Hybrid H2 (LSTM)** | **Phys + Residual** | **2.15 ± 0.01** | **0.82** | **Beats Lasso by 14% ✓** |
| **E4cnn Hybrid H2 (CNN)**| **Phys + Residual** | **2.13 ± 0.02** | **0.82** | **Best overall deep model ✓** |
| E5 Camera CNN | $3 \times 64 \times 64$ frames | 4.42 ± 0.12 | 0.07 | Visual-only lacks metric scale |
| E6_min10 Hybrid H3 | IMU + Camera | 2.85 ± 0.05 | 0.64 | Multi-modal fusion |
| E7 Best Hybrid | Multi-modal | 2.18 ± 0.06 | 0.81 | Validation-selected ensemble |

- **Key Insight:** Inductive bias is critical on small datasets. Pure deep learning struggles; grounding the network in analytical physics (H2 residual) achieves the lowest error with minimal seed variance (std 0.012 cm).

## Slide 3: Practical Payoff — Trajectory Drift Correction
- Figure: `figures/trajectories/trajectory_MH_01_easy.png` and `figures/trajectories/ate_over_time.png`
- Reconstruct continuous paths using non-overlapping 2.0 s windows (every 8th window, stride 2.0 s)
- Subtract predicted 3-axis drift vector from dead-reckoned endpoints: $\mathbf{p}_{\text{corr}} = \mathbf{p}_{\text{DR}} - \hat{\mathbf{e}}$

| Sequence | Method | Steps | Endpoint MAE (cm) | Final ATE (m) | ATE Reduction |
|---|---|---|---|---|---|
| MH_01_easy | Dead-Reckoning (Raw) | 90 | 121.5 | 87.09 | Baseline |
| MH_01_easy | Physics-only | 90 | 6.90 (-94.3%) | 1.66 | 98.0% |
| MH_01_easy | **E4vec (seed 42)** | 90 | **6.54 (-94.6%)** | **1.49** | **98.1%** |
| MH_01_easy | E4vec (3-seed mean) | 90 | 6.73 ± 0.14 | 1.51 ± 0.02 | 98.1% |
| MH_02_easy | Dead-Reckoning (Raw) | 74 | 120.0 | 72.09 | Baseline |
| MH_02_easy | **Physics-only** | 74 | **5.29 (-95.6%)** | **0.57** | **99.0%** |
| MH_02_easy | E4vec (seed 42) | 74 | 5.37 (-95.5%) | 0.83 | 98.5% |
| MH_02_easy | E4vec (3-seed mean) | 74 | 5.43 ± 0.05 | 0.89 ± 0.04 | 98.6% |

- **Physics vs Learned:** Physics-only correction provides ~98–99% reduction alone (0.57 m on MH_02); E4vec further refines MH_01 (1.49 m vs 1.66 m).
- Unit tests (`test_correction.py`) prove zero-invariance, perfect-vector zero-error, and >80% empirical reduction.
- Single command reproduction: `bash run_phase2.sh`.

