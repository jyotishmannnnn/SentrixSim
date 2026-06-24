"""Layer 6 - Synchronization.

Builds the unified microsecond hub timebase and the per-stream sampling masks,
then assembles the master-grid aligned table via latest-at (zero-order hold) -
the fill_latest_at() semantics of ENGINE 3.1. BMM350 is sampled at the field
rate (<=400 Hz) and held to the 1600 Hz master grid; temperature at <=50 Hz.

Inputs:  bmm_out, lis_out (computed on the master grid), params.
Outputs: aligned dict + bmm_valid / temp_valid masks + t_master_us.

Assumptions
-----------
* Mark 2 transport is PIO-bit-banged I2C Fast+ (1 MHz) with lock-step PIO start +
  BMM350 forced-mode + a hardware strobe (architecture_freeze 3.4) -- NOT I3C
  broadcast. Skew budgets are TARGETS for that lock-step start spread, not
  measured residuals; intra-array skew is modelled as a small fixed per-sensor
  offset within the timestamp.
* Hub clock is ideal (alpha=1, beta=0) in v1.

Limitations
-----------
* Models the sync ARCHITECTURE, not real hardware jitter (labelled
  sync_quality = simulated-target).

Hardware-upgrade path
---------------------
* Replace target skews with measured genlock latency-probe residuals (Phase 4
  Y1-Y6); fit the affine clock model alpha,beta from PTP/event logs.
"""
from __future__ import annotations

import numpy as np

from ..params import ParameterRegistry


def _hold(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-order-hold: carry the last valid sample forward along axis 0."""
    out = values.copy()
    last = None
    for t in range(values.shape[0]):
        if valid[t]:
            last = values[t]
        elif last is not None:
            out[t] = last
    return out


def run(n: int, bmm_out: dict, lis_out: dict, reg: ParameterRegistry,
        jitter_us: np.ndarray | None = None) -> dict:
    fs = float(reg.get("sync.master_rate_hz"))
    field_rate = float(reg.get("sync.field_rate_hz"))
    temp_odr = float(reg.get("lis.temp_odr_hz"))
    ts_res = float(reg.get("sync.timestamp_res_us"))

    t_us = np.round(np.arange(n) / fs * 1e6 / ts_res) * ts_res
    t_us = t_us.astype(np.int64)
    # Timestamp jitter (hard mode #6): perturb the grid, keep strictly monotonic.
    if jitter_us is not None:
        t_us = t_us + np.round(jitter_us).astype(np.int64)
        # enforce strictly-increasing (min gap 1 us): cummax then +arange.
        t_us = np.maximum.accumulate(t_us) + np.arange(n, dtype=np.int64)
        if t_us[0] < 0:                       # keep timestamps non-negative
            t_us = t_us - t_us[0]

    field_ratio = max(1, int(round(fs / field_rate)))
    temp_ratio = max(1, int(round(fs / temp_odr)))
    bmm_valid = (np.arange(n) % field_ratio) == 0
    temp_valid = (np.arange(n) % temp_ratio) == 0

    B = _hold(bmm_out["B_read_uT"], bmm_valid)
    B_lsb = _hold(bmm_out["B_lsb"].astype(float), bmm_valid).astype(np.int64)
    sat = _hold(bmm_out["sat_flag"].astype(float), bmm_valid) > 0.5
    temp = _hold(lis_out["temp_read_c"], temp_valid)
    drop_in = bmm_out.get("dropout")
    dropout = (_hold(drop_in.astype(float), bmm_valid) > 0.5
               if drop_in is not None else np.zeros((n, B.shape[1]), bool))

    return {
        "t_master_us": t_us,
        "B_read_uT": B,
        "B_lsb": B_lsb,
        "sat_flag": sat,
        "dropout": dropout,
        "bmm_valid": bmm_valid,
        "accel_read_g": lis_out["accel_read_g"],
        "accel_lsb": lis_out["accel_lsb"],
        "temp_read_c": temp,
        "temp_valid": temp_valid,
    }
