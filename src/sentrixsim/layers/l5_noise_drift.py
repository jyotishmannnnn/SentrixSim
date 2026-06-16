"""Layer 5 - Noise and drift.

Owns the seeded RNG and the thermal field. Provides Gaussian noise at the
datasheet RMS to Layers 3/4. Slow 1/f drift is OFF by default (magnitude
UNKNOWN); the thermal term is OFF by default (bmm.tco/tcs disabled).

Assumptions
-----------
* Noise is white and independent across sensors/axes (benign-EMI human glove,
  DERIV A6). Correlated EMI is not modelled.

Limitations
-----------
* Drift coefficients are unknown -> default = noise only (sim never falsely
  understates achievable quality).

Hardware-upgrade path
---------------------
* Fit drift/temperature coefficients from a thermal-chamber sweep and enable
  bmm.tco/tcs, lis.tcoff; add measured 1/f drift PSD.
"""
from __future__ import annotations

import numpy as np


class NoiseModel:
    """Per-sample Gaussian noise (rng) + optional per-episode static drift offset
    (drift_rng). Drift is a separate realization axis: a constant per-sensor bias
    drawn once per episode, standing in for the UNKNOWN per-unit/session offset
    spread (cal.bundle). It is OFF unless a drift_seed is supplied."""

    def __init__(self, seed: int, drift_seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.drift_rng = np.random.default_rng(drift_seed) if drift_seed is not None else None

    def gauss(self, shape, sigma) -> np.ndarray:
        """Gaussian noise; sigma may be scalar or broadcastable array."""
        return self.rng.normal(0.0, 1.0, size=shape) * np.asarray(sigma)

    def drift_offset(self, shape, sigma) -> np.ndarray:
        """Static per-episode offset (drift realization). Zeros if no drift_seed."""
        if self.drift_rng is None:
            return np.zeros(shape)
        return self.drift_rng.normal(0.0, 1.0, size=shape) * np.asarray(sigma)
