"""Layer 4 - LIS2DTW12 accelerometer + temperature model.

Acceleration: full-scale clip, sensitivity quantization (0.244 mg/LSB at +/-2g,
14-bit; scaled per range), datasheet noise (90 ug/rtHz HP; 1.3 mg LP floor),
bandwidth ~ ODR/2.
Temperature: Gaussian about true T at 0.8 degC accuracy, coarse quantization.

Inputs:  accel_true_g {finger->(T,3)}, temp_true_c (T,), NoiseModel, params.
Outputs: accel_read_g[T,3,3], accel_lsb, temp_read_c[T,3].
(3 tripod sites, ordered thumb, index, middle.)

Assumptions
-----------
* Sites ordered [thumb, index, middle]. HP 14-bit by default.
* Temp LSB/degC UNKNOWN -> output modelled directly in degC with a coarse quant.

Limitations
-----------
* Slip->vibration coupling unknown (injected only if event enables it).
* TCoff OFF by default.

Hardware-upgrade path
---------------------
* Calibrate temp LSB/degC and zero-g TCoff from a thermal sweep; refine noise
  per measured ODR/mode table.
"""
from __future__ import annotations

import numpy as np

from ..params import ParameterRegistry
from .l5_noise_drift import NoiseModel

TRIPOD = ["thumb", "index", "middle"]


def run(accel_true_g: dict, temp_true_c: np.ndarray, reg: ParameterRegistry,
        noise: NoiseModel, scene: dict, dyn_fingers: list[str] | None = None) -> dict:
    # Dynamics sites from the topology descriptor; default to Layout-B tripod.
    fingers = dyn_fingers if dyn_fingers is not None else TRIPOD
    nsite = len(fingers)
    T = temp_true_c.shape[0]
    fs_g = float(reg.get("lis.fs_g"))
    bits = int(reg.get("lis.bits"))
    odr = float(reg.get("lis.odr_hz"))
    nd = float(reg.get("lis.noise_density_ug_rthz")) * 1e-6  # ug/rtHz -> g/rtHz
    floor_lp = float(reg.get("lis.noise_floor_lp_mg")) * 1e-3
    mode = scene.get("l4", {}).get("accel_mode", "high_performance")

    lsb_g = fs_g / (2 ** (bits - 1))                 # g per code at selected FS
    bw = odr / 2.0
    if mode == "high_performance":
        sigma_a = nd * np.sqrt(bw)
    else:
        sigma_a = floor_lp

    # Per-episode static zero-g offset (drift realization). ESTIMATED stand-in
    # for the UNKNOWN per-unit offset/TCoff spread; zero unless drift_seed set.
    off_spread = float(reg.get("lis.offset_spread_mg")) * 1e-3
    drift_off = noise.drift_offset((nsite, 3), off_spread)

    accel_read = np.zeros((T, nsite, 3))
    accel_lsb = np.zeros((T, nsite, 3), np.int64)
    for k, f in enumerate(fingers):
        a = accel_true_g.get(f, np.zeros((T, 3)))
        a = a + noise.gauss((T, 3), sigma_a) + drift_off[k][None, :]
        a = np.clip(a, -fs_g, fs_g)
        codes = np.round(a / lsb_g).astype(np.int64)
        accel_lsb[:, k, :] = codes
        accel_read[:, k, :] = codes.astype(float) * lsb_g

    # Temperature channel.
    t_acc = float(reg.get("lis.temp_acc_c"))
    t_quant = float(reg.get("lis.temp_quant_C"))
    temp_read = np.zeros((T, nsite))
    for k in range(nsite):
        tt = temp_true_c + noise.gauss((T,), t_acc)
        temp_read[:, k] = np.round(tt / t_quant) * t_quant
    return {"accel_read_g": accel_read, "accel_lsb": accel_lsb, "temp_read_c": temp_read}
