# Feature Dictionary (Part C)

Every feature in `data/processed/features_<seq>.csv`, what it means in plain
words, and why it might reveal drift. One window = 1.0 s of IMU data (200
samples × 6 channels). All features are computed per window by
`src/vio_drift/features/`; the full list is produced by
`build_features.feature_names()`.

**Total: 104 features** = 60 time-domain + 18 frequency-domain + 26 cross-axis.

## Channel naming

Window arrays store six channels in this fixed order (see `windowing.py`):

| Prefix | Sensor | Axis | Unit |
|--------|--------|------|------|
| `gx`, `gy`, `gz` | Gyroscope (angular rate) | x, y, z | rad/s |
| `ax`, `ay`, `az` | Accelerometer (specific force) | x, y, z | m/s² |

A feature name is `<channel>_<statistic>` for per-axis features (e.g.
`ax_rms`), or a descriptive name for the cross-axis features (e.g.
`acc_mag_mean`, `corr_gx_ay`).

**Why these should reveal drift.** An open-loop IMU integrator (the one that
builds our label) accumulates position error fastest when a window is
energetic, erratic, or hard to integrate between samples. So features that
measure *how much* and *how abruptly* the drone moved, and *how* its axes move
together, are natural predictors of the drift that accumulated over that
second.

---

## 1. Time-domain features — `src/vio_drift/features/time_domain.py`

Ten statistics on each of the 6 channels → **60 features**. For a channel `c`
in `{gx, gy, gz, ax, ay, az}`:

| Feature | Meaning (plain words) | Why it might reveal drift |
|---------|-----------------------|---------------------------|
| `c_mean` | Average reading over the window. | For accel axes this tracks the gravity direction / sustained acceleration; a biased mean is exactly what double-integration turns into position error. `ax_mean` is the single strongest linear correlate of the label. |
| `c_std` | Spread of the readings about their mean. | More variation = more motion and noise to integrate = more room for error to accumulate. |
| `c_var` | Variance (std squared). | Same signal as `std`, on a squared scale; emphasises high-energy windows. |
| `c_rms` | Root-mean-square: typical magnitude including any offset. | Combines "how big" and "how variable" into one energy-like number; `ax_rms` is the top RandomForest feature. |
| `c_min` | Smallest reading in the window. | Captures extremes (hard decelerations, sharp rotations) a mean would hide. |
| `c_max` | Largest reading in the window. | Same, for the positive extreme. |
| `c_ptp` | Peak-to-peak: `max − min`, the full swing. | A large swing within one second means violent motion that strapdown integration handles poorly. |
| `c_skew` | Skewness: asymmetry of the value distribution. | A lopsided window (e.g. one strong push) integrates differently than a symmetric wobble. 0 for a flat channel. |
| `c_kurt` | Excess kurtosis: how "peaky"/heavy-tailed the values are. | High kurtosis flags rare spikes (bumps, transients) that drive sudden error. 0 for a flat channel. |
| `c_zcr` | Zero-crossing rate about the window **mean**: fraction of adjacent samples that change sign after the mean is removed. | A cheap frequency proxy — how often the signal oscillates. Fast oscillation = vibration/shake, which the integrator smooths over and mis-tracks. Mean-removed so the accel gravity offset does not dominate. |

---

## 2. Frequency-domain features — `src/vio_drift/features/frequency_domain.py`

Three FFT statistics on each of the 6 channels → **18 features**. The
per-window mean is removed before the FFT so the accelerometer's gravity/DC
term is not read as a 0 Hz "signal". Sampling rate `fs = 200 Hz`.

| Feature | Meaning (plain words) | Why it might reveal drift |
|---------|-----------------------|---------------------------|
| `c_dom_freq` | Frequency (Hz) of the strongest spectral peak (DC excluded). | Where the motion's energy sits. Low-frequency = smooth manoeuvres; high-frequency = vibration/noise, which integrates poorly. 0 Hz for a flat channel. |
| `c_spec_energy` | Total AC power: summed power spectrum after mean removal. | A frequency-domain measure of how energetic the window is; complements time-domain `var`/`rms`. |
| `c_spec_entropy` | Shannon entropy of the power spectrum, normalised to [0, 1]. | *How spread out* the energy is: near 0 = one clean tone (coordinated motion), near 1 = broadband (noise-like). The accel/gyro spectral-entropy features are among the strongest label correlates, suggesting noise-like windows drift more. |

---

## 3. Cross-axis & magnitude features — `src/vio_drift/features/cross_axis.py`

Vector magnitudes, jerk, and channel correlations → **26 features**. These
capture coupling the per-axis features miss, and are orientation-independent.

### Vector magnitudes (6)

| Feature | Meaning | Why it might reveal drift |
|---------|---------|---------------------------|
| `acc_mag_mean` | Mean of \|accel\| = √(ax²+ay²+az²) over the window. | Heading-independent "amount of linear motion" (plus gravity); a compact overall energy measure. |
| `acc_mag_std` | Spread of \|accel\|. | Variability of total acceleration — bursty motion. |
| `acc_mag_max` | Peak \|accel\|. | The hardest instant of the window. |
| `gyro_mag_mean` | Mean of \|gyro\| = √(gx²+gy²+gz²). | How much the drone is rotating overall; rotation errors feed into position drift via the attitude estimate. |
| `gyro_mag_std` | Spread of \|gyro\|. | Variability of rotation rate. |
| `gyro_mag_max` | Peak \|gyro\|. | Sharpest turn in the window. |

### Jerk (5) — jerk = d(accel)/dt, scaled by `fs`

| Feature | Meaning | Why it might reveal drift |
|---------|---------|---------------------------|
| `jerk_ax_rms` | RMS jerk on the accel x-axis (m/s³). | Jerk is how fast acceleration changes. Big jerk = the signal moves a lot *between* samples, exactly where discrete strapdown integration loses accuracy. |
| `jerk_ay_rms` | RMS jerk, accel y-axis. | As above, y-axis. |
| `jerk_az_rms` | RMS jerk, accel z-axis. | As above, z-axis. |
| `jerk_mag_mean` | Mean magnitude of the 3-axis jerk vector. | Overall abruptness of the window. |
| `jerk_mag_max` | Peak jerk magnitude. | The single most abrupt change — a prime cause of a sudden error jump. |

### Channel correlations (15)

| Feature | Meaning | Why it might reveal drift |
|---------|---------|---------------------------|
| `corr_<a>_<b>` | Pearson correlation between channels `a` and `b`, for all 15 unordered pairs of `{gx, gy, gz, ax, ay, az}` (e.g. `corr_gx_gy`, `corr_gz_ax`, `corr_ax_ay`). | *How* the axes move together. Strong cross-axis correlation is the signature of coordinated rigid-body motion (banking turns, climbs) rather than uncorrelated sensor noise. Several accel–gyro cross terms (e.g. `corr_gz_ax`) rank high in RandomForest importance. 0 when either channel is constant (correlation undefined). |

---

## Scaling note

`features_<seq>.csv` is written **unscaled on purpose**. Because the project
splits by sequence (leave-one-sequence-out), any `StandardScaler` must be fit
on a fold's training sequences only and then applied to the held-out
sequence — fitting across all sequences first would leak test statistics.
`build_features.fit_scaler` / `apply_scaler` do this at modelling time
(Part D). See `src/vio_drift/features/build_features.py`.

## Redundancy / feature selection

Several features are highly collinear by construction — e.g. `c_var` is
`c_std²`, and `c_rms ≈ |c_mean|` when the mean dominates. Pooled across the
two easy sequences (`MH_01_easy`, `MH_02_easy`), 22 feature pairs have |correlation| > 0.95. This is expected
and not a bug: tree models are insensitive to it, and `notebook 04` runs
correlation and feature-selection experiments to show how a reduced feature
set performs. See that notebook for the current ranking.
