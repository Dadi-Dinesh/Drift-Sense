"""
Frequency-domain (FFT) features, computed per IMU channel on one window.

Time-domain statistics say *how much* a window moves; the spectrum says
*how* it moves. A smooth hover concentrates its energy in a few low
frequencies, while vibration, fast attitude changes or sensor noise
spread energy across higher frequencies. Both the location of the energy
(dominant frequency) and how spread-out it is (spectral entropy) plausibly
track how hard a window is to integrate cleanly, and therefore how much
drift accumulates.

We work on a single window of shape (n_samples, 6) with the project's
``gx, gy, gz, ax, ay, az`` column order, sampled at a fixed rate
``fs`` (200 Hz for EuRoC). The per-window mean is removed before the FFT
so the large accelerometer DC/gravity term does not masquerade as a
0 Hz "dominant frequency".
"""
from __future__ import annotations

import numpy as np

from .time_domain import AXES

_STATS: tuple[str, ...] = ("dom_freq", "spec_energy", "spec_entropy")


def feature_names() -> list[str]:
    """Column names this module produces, in output order."""
    return [f"{axis}_{stat}" for axis in AXES for stat in _STATS]


def _spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of a mean-removed signal and its freq axis."""
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs, spectrum


def extract(window: np.ndarray, fs: float = 200.0) -> dict[str, float]:
    """
    Frequency-domain features for one window of shape (n_samples, 6).

    For each channel:

    * ``dom_freq``     — frequency (Hz) of the largest spectral peak, DC
      excluded. A nearly flat channel has no peak and reports 0.0 Hz.
    * ``spec_energy``  — total AC power, i.e. the summed power spectrum
      after mean removal. Rises with how energetic the window is.
    * ``spec_entropy`` — Shannon entropy of the power spectrum normalised
      to [0, 1]. Near 0 means energy sits in one tone; near 1 means it is
      spread broadband (noise-like). Reported 0.0 for a silent channel.

    Returns an ordered mapping whose keys match :func:`feature_names`;
    all values are finite.
    """
    window = np.asarray(window, dtype=np.float64)
    out: dict[str, float] = {}

    for i, axis in enumerate(AXES):
        freqs, spectrum = _spectrum(window[:, i], fs)
        total = float(spectrum.sum())

        if total <= 0.0:
            # Constant channel: no spectral content at all.
            out[f"{axis}_dom_freq"] = 0.0
            out[f"{axis}_spec_energy"] = 0.0
            out[f"{axis}_spec_entropy"] = 0.0
            continue

        # Dominant frequency: strongest bin, ignoring the DC bin (index 0,
        # already ~0 after mean removal but kept out on principle).
        if spectrum.size > 1:
            dom_idx = int(np.argmax(spectrum[1:])) + 1
        else:
            dom_idx = 0
        out[f"{axis}_dom_freq"] = float(freqs[dom_idx])
        out[f"{axis}_spec_energy"] = total

        # Normalised spectral entropy of the power distribution.
        p = spectrum / total
        nonzero = p[p > 0]
        entropy = -float(np.sum(nonzero * np.log2(nonzero)))
        max_entropy = np.log2(spectrum.size)
        out[f"{axis}_spec_entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0

    return out
