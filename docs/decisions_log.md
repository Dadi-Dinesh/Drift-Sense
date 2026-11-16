# Drift-Sense Phase 2 — Team Decisions Log

> **Purpose:** Record every significant design decision made during Phase 2 (R5: Evaluation, Ablations and Drift Correction), so that the viva examiner can follow our reasoning and so future team members can understand the rationale.

---

## D1 — Dataset Scope (2026-11-01)

**Decision:** Keep only MH_01_easy and MH_02_easy for all Phase 2 experiments.

**Rationale:**
- Phase 1 used these two sequences exclusively; changing the dataset would break comparability.
- Together they provide **1,312 windows** (720 on MH_01_easy, 592 on MH_02_easy), enough to show trends but small enough to train quickly on CPU.
- Leave-one-sequence-out cross-validation gives 2 folds, matching the Phase 1 split in `data/processed/splits.json`.

**Alternatives rejected:** Adding MH_03_medium — rejected because we do not have Phase 1 baselines for it, so the comparison table would be incomplete.

---

## D2 — Camera as Tier 0 (IMU-only Default) (2026-11-01)

**Decision:** The primary pipeline is IMU-only (Tier 0). Camera input is an optional ablation controlled by the `use_camera` flag in `configs/ablations.yaml`.

**Rationale:**
- Real-world VIO systems lose camera signal in dark or texture-less environments; IMU-only robustness matters.
- Camera data (EuRoC `/cam0`) requires heavy preprocessing; using a proxy prevents fair comparison.
- Experiments E5 and E6 explore camera at ablation time only.

**Alternatives rejected:** Making camera the default — rejected because it would obscure the IMU contribution and add preprocessing complexity.

---

## D3 — Primary Target: err_mag_m + 3-axis Head (2026-11-01)

**Decision:** Train models to predict the scalar drift magnitude `err_mag_m` as the primary loss target, and add a 3-axis head (err_dx, err_dy, err_dz) for trajectory correction.

**Rationale:**
- Phase 1 evaluated on `err_mag_m`; using the same target makes Phase 2 results directly comparable.
- The 3-axis head (trained as E4vec) enables `correct_trajectory.py` to subtract the predicted vector from each dead-reckoned endpoint.
- Standardized target scaling ($\bar{y} + s \cdot z$) ensures networks do not waste capacity learning the ~1.2 m offset.

---

## D4 — Framework and Seeds (2026-11-02)

**Decision:** Use PyTorch (CPU-first). Report every DL number as mean ± std over 3 seeds: **42, 43, 44**.

**Rationale:**
- PyTorch is the team standard and is available system-wide.
- 3 seeds provide statistical confidence while keeping total CPU compute tractable.
- Seed set {42, 43, 44} aligns across R1, R2, R3, R4, and R5.

---

## D5 — Same Folds as Phase 1 (2026-11-02)

**Decision:** All experiments use the same leave-one-sequence-out folds defined in `data/processed/splits.json` as Phase 1.

**Rationale:**
- Guarantees that train/test splits are identical, so any MAE difference is due to the model, not the split.
- The evaluation harness (`dl_eval.py`) loads `splits.json` directly and raises an error if it is missing.

---

## D6 — Hybrid H2 as Main Architecture (E4) (2026-11-03)

**Decision:** The primary deep learning model is Hybrid H2: an analytical physics-prior baseline concatenated with a residual neural network that predicts the correction to the physics estimate.

**Rationale:**
- Physics prior alone achieves R² = 0.82 (Phase 1); the residual net learns to correct what physics misses.
- E4 achieves MAE = **2.15 ± 0.01 cm** with the LSTM encoder and **2.13 ± 0.02 cm** with the CNN encoder (E4cnn), outperforming the Phase 1 Lasso baseline (2.50 cm) on both flights.
- Output head is zero-initialized, ensuring the untrained network starts exactly at the physics prior.

---

## D7 — Non-overlapping Windows for Trajectory (2026-11-03)

**Decision:** Select every 8th window (2.0 s stride at 0.25 s step) for trajectory reconstruction.

**Rationale:**
- Overlapping sliding windows share up to 1.75 s of flight. Chaining them would compound errors and distort path length.
- The 8th-window selection is deterministic and yields 90 steps on MH_01_easy and 74 steps on MH_02_easy.

---

## D8 — Evaluation Harness Proves Comparability (2026-11-04)

**Decision:** `dl_eval.py:reproduce_phase1_benchmarks()` must return Lasso MAE = 2.504 cm (tolerance ±0.01 cm) and physics-only R² = 0.820 (±0.005). These checks are enforced by `tests/test_dl_eval.py`.

**Rationale:**
- Without bit-exact reproduction of Phase 1 numbers, any "improvement" could be an artifact of preprocessing or split differences.
- Automated regression tests run in CI before every ablation summary is generated.

---

## D9 — Unified Collection and Validation Entry Point (2026-11-05)

**Decision:** All ablation experiments (E0–E7, A1–A5, C1) are defined in `configs/ablations.yaml` and evaluated/collected by `scripts/collect_final_results.py` (and `scripts/run_ablations.py`).

**Rationale:**
- Reproducibility: single deterministic script reading verified model predictions and scoring them with Phase 1 metrics.
- Output CSVs in `results/ablations/` are version-controlled and checked by `--check`.
- `run_phase2.sh` provides single-command end-to-end reproduction.

---

## D10 — Correction Metric: Endpoint MAE and Cumulative ATE (2026-11-05)

**Decision:** Report two correction metrics: (1) per-window endpoint MAE before/after, (2) cumulative Absolute Trajectory Error (ATE) along the reconstructed path using E4vec 3-axis predictions.

**Rationale:**
- Endpoint MAE validates local dead-reckoning accuracy.
- Cumulative ATE is the standard VIO/SLAM metric, demonstrating global navigational payoff.
- E4vec reduces endpoint MAE by 94.6% (MH_01) and 95.5% (MH_02), and cumulative ATE by **98.1%** (87.1 m to 1.49 m) and **98.5%** (72.1 m to 0.83 m).

---

## D11 — A2 Data-Scaling Re-run with Paper Recipe (2026-11-16)

**Decision:** Re-train A2 (data-scaling ablation) with the paper's recipe — AdamW, MSE loss, standardised target, early stopping, seeds 42/43/44 — instead of using hard-coded numbers from the legacy 560cb6f recipe.

**Rationale:**
- The original A2 numbers (2.44 / 2.33 / 2.29 cm) were produced by the legacy harness: 25 epochs, no validation block, Adam with Huber loss, seeds 42/123/999. They did not match the recipe used for all other experiments.
- Re-running with the paper recipe gives **2.12 / 2.21 / 2.15 cm** for 50% / 75% / 100% training fractions. The non-monotone pattern is expected: H2's physics prior provides such a strong starting point that the residual converges even with 251 windows.
- Prediction files are saved as `results/predictions/A2_H2_frac{0.50,0.75}_*.csv`; `collect_final_results.py` now scores them dynamically.

**Alternatives rejected:** Keeping the legacy numbers with a disclaimer — rejected because the report should only present numbers from a single consistent recipe.

---

## D12 — Trajectory Correction Merge Bug Fix (2026-11-16)

**Decision:** Fix `correct_trajectory.py` to handle physics CSVs (which contain both `seq_id` and `window_id` for all sequences) by filtering on `seq_id == seq_name` before merging, and joining on `["seq_id", "window_id"]` when both columns are present.

**Rationale:**
- The physics CSV (`data/processed/physics_leave_MH_01_easy.csv`) contains rows for all 1,312 windows across both sequences. Joining on `window_id` alone produced 164 rows instead of the expected 90 non-overlapping windows.
- The fix uses `_extract_from_df()` which: (1) filters by `seq_id == seq_name` if `seq_id` present, (2) merges on `["seq_id", "window_id"]` if both DataFrames have `seq_id`, else `["window_id"]` alone, (3) auto-selects `y_pred_dx/dy/dz` or `phys_dx/dy/dz` columns.
- Verified: `run_sequence_correction('MH_01_easy', pred_vec=physics_csv)` → 6.90 cm (correct); `pytest tests/test_correction.py` → 5/5 passed.

---

## D13 — Experiment ID Alignment (2026-11-16)

**Decision:** Align `collect_final_results.py` experiment ID mapping to match the paper: E1 = raw 1D-CNN, E2 = raw LSTM, E3 = H1 (without min-10), E6 = H3 (without min-10). Legacy min-10 variants and scaled-input models are listed as separate rows (E3_min10, E6_min10, IMUcnn, IMUlstm).

**Rationale:**
- The previous mapping had E1 → IMUcnn, E2 → IMUlstm, E3 → E3_min10, E6 → E6_min10, which did not match the paper's Table III.
- Correcting this ensures `experiments_e0_e7.csv` can be cited directly from the report without ID translation.

---

## References

- EuRoC MAV dataset: Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016.
- Drift evaluation methodology: Sturm et al., "A benchmark for the evaluation of RGB-D SLAM systems", IROS 2012.
- Ablation study design: Melis & Dyer, "On the state of the art of evaluation in neural language models", ICLR 2018.
- VIO drift correction: Forster et al., "IMU preintegration on manifold for efficient visual-inertial maximum-a-posteriori estimation", RSS 2015.
