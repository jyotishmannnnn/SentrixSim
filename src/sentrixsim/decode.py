"""Decode demo (inverse path) - light contact detector per ENGINE 3.2.

NOT part of the forward model. Produces a *decoded estimate* of contact and
normal-force proxy from the simulated BMM350 field, so the dataset is
self-validating: ground-truth labels (source=ground_truth) sit next to decoded
estimates (source=simulated_estimate) and the gap is measurable.

Pipeline: per-cluster aggregate field magnitude -> zero-phase Butterworth
low-pass (filtfilt, no onset drift) -> Schmitt hysteresis contact detection.

Limitation: measures internal consistency, not accuracy against true force
(CTO 4.3). The force proxy is normalized field deflection, not Newtons.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .params import ParameterRegistry
from .topology import Topology


def decode_contacts(aligned: dict, topo: Topology, reg: ParameterRegistry,
                    scene: dict) -> dict:
    dec = scene.get("decode", {})
    if not dec.get("enabled", True):
        return {}
    fs = float(reg.get("sync.master_rate_hz"))
    fc = float(dec.get("lowpass_fc_hz", 30.0))
    B = aligned["B_read_uT"]                       # (T,21,3)
    B0 = np.asarray(reg.get("env.B0_uT"), float)
    dB = np.linalg.norm(B - B0, axis=-1)           # (T,21) deflection magnitude

    # group bmm indices by finger
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(topo.bmm_sites):
        groups.setdefault(s.finger, []).append(i)

    b, a = butter(4, min(fc / (fs / 2.0), 0.99), btype="low")
    out = {}
    for f, idxs in groups.items():
        agg = dB[:, idxs].mean(axis=1)
        if agg.shape[0] > 12:
            agg = filtfilt(b, a, agg)
        span = float(agg.max() - agg.min()) or 1.0
        hi = agg.min() + dec.get("hysteresis_hi", 0.30) * span
        lo = agg.min() + dec.get("hysteresis_lo", 0.15) * span
        state = 0
        contact = np.zeros_like(agg, bool)
        for t in range(agg.shape[0]):
            if state == 0 and agg[t] > hi:
                state = 1
            elif state == 1 and agg[t] < lo:
                state = 0
            contact[t] = state == 1
        # normalized force proxy in [0,1]
        force_proxy = (agg - agg.min()) / span
        out[f] = {"contact": contact, "force_proxy": force_proxy}
    return out
