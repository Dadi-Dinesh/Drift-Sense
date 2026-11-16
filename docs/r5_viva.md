# R5 viva preparation: Technical Validation, Ablations and Drift Correction

1. **What is R5's core responsibility?**
   R5 owns technical validation: establishing cross-role comparability, enforcing identical evaluation metrics to Phase 1, running systematic ablations (E0–E7, A1–A5), and validating practical payoff via trajectory dead-reckoning drift correction (C1).

2. **Why was reproducing Phase 1 before anything else mandatory?**
   Without reproducing Phase 1's numbers bit-for-bit, any reported improvement could easily stem from differing evaluation windows, target scales, or data splits. Our evaluation harness (`dl_eval.py`) reproduces the exact Phase 1 benchmarks: Mean baseline at 4.53 cm, Lasso at 2.50 cm (MAE 0.02504 m, R² = 0.76), and Physics-only at R² = 0.82. This is guarded by unit tests (`test_dl_eval.py`).

3. **What are the primary metrics used across all models?**
   MAE (cm), RMSE (cm), R² (coefficient of determination), and Mean Signed Bias (cm). MAE on the 2.0 s endpoint position error magnitude (`err_mag_m`) is the primary comparison metric, identical to Phase 1.

4. **Why evaluate over 3 seeds (42, 43, 44)?**
   Deep learning models with small datasets (1,312 total windows across two flights) exhibit optimization stochasticity from parameter initialisation and batch shuffling. Reporting `mean ± std` over seeds 42, 43, 44 ensures that gains are statistically robust and not cherry-picked random seeds.

5. **Why evaluate first-20% (take-off) vs remaining-80% (steady flight) separately?**
   Take-off dynamics involve high acceleration and angular rate changes, producing larger drift errors (e.g., Camera CNN E5 MAE ~8.6–12.2 cm during take-off vs ~2.7–3.3 cm during steady flight). Evaluating segments reveals whether models handle transient high-acceleration phases or only fit benign steady motion.

6. **What do the E0–E7 experiment families show?**
   - **E0 (Phase 1 Lasso):** 2.50 cm MAE, R² = 0.76 (deterministic benchmark).
   - **E1 (1D-CNN, IMUcnn):** 2.60 ± 0.12 cm MAE, R² = 0.72 with standardized target.
   - **E2 (LSTM, IMUlstm):** 3.69 ± 0.06 cm MAE, R² = 0.31.
   - **E3 (Hybrid H1 - concatenation):** 2.64 ± 0.03 cm MAE, R² = 0.73 with 10-epoch warm-up.
   - **E4 (Hybrid H2 - residual physics, LSTM):** **2.15 ± 0.01 cm MAE, R² = 0.82**, outperforming Lasso by 14%.
   - **E4cnn (Hybrid H2 with CNN):** **2.13 ± 0.02 cm MAE, R² = 0.82**, the best overall deep model.
   - **E5 (Camera CNN):** 4.42 ± 0.12 cm MAE, R² = 0.07 (visual-only lacks metric scale without stereo/IMU).
   - **E6 (Hybrid H3 - IMU + Camera):** 2.85 ± 0.05 cm MAE, R² = 0.64.
   - **E7 (Best ensemble/hybrid variant):** 2.18 ± 0.06 cm MAE, R² = 0.81.

7. **Why does Hybrid H2 (E4) succeed where unconstrained models struggle?**
   Because of its inductive bias: physics integration provides an analytical prior ($R^2 = 0.82$), and the neural network only learns the residual deviation. Its output head is zero-initialised, guaranteeing that the model starts at the physics solution and can never perform worse than physics on uninformative data.

8. **How does trajectory drift correction work (C1)?**
   For each non-overlapping 2.0 s window (selected every 8th window with 0.25 s step):
   - The E4vec model predicts the 3D drift displacement vector $[\Delta x, \Delta y, \Delta z]$.
   - The corrected endpoint is obtained by subtracting the predicted drift from the dead-reckoned endpoint: $\mathbf{p}_{\text{corrected}} = \mathbf{p}_{\text{dead-reckoned}} - \hat{\mathbf{e}}$.
   - A continuous trajectory is reconstructed by chaining corrected window vectors.

9. **Why use non-overlapping windows for trajectory reconstruction?**
   Because sliding windows overlap by 1.75 s. Integrating overlapping predictions would count the same physical interval multiple times, severely distorting the cumulative trajectory length and error.

10. **What are the practical trajectory correction results?**
    - **MH_01_easy:** Endpoint MAE reduced from 121.5 cm to 6.54 cm (94.6% reduction); final drift reduced from 87.1 m to 1.49 m (**98.1% ATE reduction**).
    - **MH_02_easy:** Endpoint MAE reduced from 120.0 cm to 5.37 cm (95.5% reduction); final drift reduced from 72.1 m to 0.83 m (**98.5% ATE reduction**).

11. **What unit tests guarantee technical validity?**
    `tests/test_dl_eval.py` (7 tests) guarantees Phase 1 metric reproduction, seed aggregation, and segment splitting. `tests/test_correction.py` (5 tests) proves zero-prediction invariance (zero error vector changes nothing), perfect-vector property (perfect prediction gives exactly 0 endpoint error), and >80% error reduction on real sequences.

12. **Can this pipeline be reproduced with a single command?**
    Yes, via `bash run_phase2.sh`, which runs all ablation sweeps, trajectory corrections, figure generation, and unit tests end-to-end.
