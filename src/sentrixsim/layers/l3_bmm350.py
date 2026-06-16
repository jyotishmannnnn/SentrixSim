"""Layer 3 - BMM350 magnetometer model.

Transforms true field B_true[T,21,3] (uT) into digital readings, applying the
datasheet-authoritative chain: calibration (identity by default), saturation at
+/-2000 uT, optional temperature drift (OFF), averaging bandwidth, datasheet
noise (190/450 nT), and 0.1 uT quantization (24-bit raw container).

Inputs:  B_true[T,21,3] uT, NoiseModel, params.
Outputs: B_read uT float[T,21,3], B_lsb int[T,21,3], sat_flag bool[T,21,3].

Equations
---------
m = S R B_true + b(T)            (cal; S=I, b=0 default)
m = clip(m, +/-range)            (saturation)
m += noise (190 nT xy, 450 nT z) (datasheet)
read = round(m / quant) * quant  (quantization, quant=0.1 uT)

Assumptions
-----------
* Per-unit calibration unknown -> identity (cal.bundle UNKNOWN).
* TCO/TCS/cross-axis/nonlinearity OFF (exact datasheet sub-values to-confirm).

Limitations
-----------
* Cannot reproduce a real per-unit calibration signature pre-hardware.

Hardware-upgrade path
---------------------
* Load S,b (and cross-axis) from the per-unit calibration bundle (BUILD Part 8);
  enable bmm.tco/tcs from a thermal sweep.
"""
from __future__ import annotations

import numpy as np

from ..params import ParameterRegistry
from .l5_noise_drift import NoiseModel


def run(B_true: np.ndarray, reg: ParameterRegistry, noise: NoiseModel,
        session_cal: dict | None = None, dropout_mask: np.ndarray | None = None) -> dict:
    """session_cal (hard mode #9): {'gain':(nb,3),'offset':(nb,3)} non-identity
    per-session calibration. dropout_mask (hard mode #5): (T,nb) bool, dropped
    samples freeze to the last good value (zero-order hold; never NaN)."""
    T, nb, _ = B_true.shape
    rng_uT = float(reg.get("bmm.range_uT"))
    quant = float(reg.get("bmm.quant_step_uT"))
    n_xy = float(reg.get("bmm.noise_xy_nT")) * 1e-3   # nT -> uT
    n_z = float(reg.get("bmm.noise_z_nT")) * 1e-3
    avg = max(1, int(reg.get("bmm.avg")))

    # Calibration (identity unless cal.bundle becomes KNOWN).
    if reg.allow_placeholders:
        _ = reg.get("cal.bundle")
    m = B_true.copy()

    # Optional temperature drift (OFF by default -> coefficients are 0).
    tco = float(reg.get("bmm.tco_nT_per_C")) * 1e-3 if reg.param("bmm.tco_nT_per_C").enabled else 0.0
    if tco:
        m = m + tco  # placeholder linear term once enabled with measured value

    # Per-episode static offset (drift realization). Zero unless a drift_seed was
    # supplied. Magnitude = bmm.offset_spread_uT (ESTIMATED stand-in for the
    # UNKNOWN per-unit zero-field offset spread).
    spread = float(reg.get("bmm.offset_spread_uT"))
    m = m + noise.drift_offset((nb, 3), spread)[None, :, :]

    # Per-session calibration (hard mode): non-identity gain + offset.
    if session_cal is not None:
        m = session_cal["gain"][None, :, :] * m + session_cal["offset"][None, :, :]

    # Saturation.
    sat = np.abs(m) >= rng_uT
    m = np.clip(m, -rng_uT, rng_uT)

    # Noise (averaging reduces RMS by 1/sqrt(N)).
    sigma = np.empty((1, 1, 3))
    sigma[..., 0] = n_xy
    sigma[..., 1] = n_xy
    sigma[..., 2] = n_z
    sigma = sigma / np.sqrt(avg)
    m = m + noise.gauss((T, nb, 3), sigma)

    # Re-clip post-noise, then quantize.
    m = np.clip(m, -rng_uT, rng_uT)
    lsb = np.round(m / quant).astype(np.int64)
    read = lsb.astype(float) * quant

    # Sensor dropouts (hard mode): vectorized zero-order hold of last good value.
    dropout = np.zeros((T, nb), bool)
    if dropout_mask is not None:
        dropout = dropout_mask.copy()
        dropout[0, :] = False  # seed sample is always valid
        valid_idx = np.where(dropout, 0, np.arange(T)[:, None])
        last = np.maximum.accumulate(valid_idx, axis=0)            # (T,nb)
        cols = np.arange(nb)[None, :]
        read = read[last, cols, :]
        lsb = lsb[last, cols, :]
        sat = sat[last, cols, :]

    return {"B_read_uT": read, "B_lsb": lsb, "sat_flag": sat, "dropout": dropout}
