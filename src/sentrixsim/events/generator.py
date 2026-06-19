"""Event generator + Layer 0 ground-truth interaction generator.

Reads a gesture YAML (idle/tap/press/hold/shear/slip/release/pinch/grasp),
builds piecewise-linear normalized force/shear profiles on the master grid, and
emits a ``GroundTruth`` object.

Assumptions
-----------
* Forces are NORMALIZED [0,1]; absolute Newtons are UNKNOWN (no scale invented).
* Slip onset is scripted ground truth; slip *velocity* is normalized only.
* Object pose is a coarse translation stub (vision is out of scope for the glove
  streams in v1).

Limitations
-----------
* Scripted profiles, not human-recorded motion; no inter-subject variability.

Hardware-upgrade path
---------------------
* Replace scripted profiles with motion captured from instrumented sessions; add
  a force_scale (N) once the 6-DoF calibration grid (BUILD Part 8) is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..core_types import PHASE_ID, GroundTruth
from ..params import ParameterRegistry


def _pw_linear(keyframes: list[list[float]], t_frac: np.ndarray) -> np.ndarray:
    kf = np.asarray(keyframes, float)
    return np.interp(t_frac, kf[:, 0], kf[:, 1])


def load_event(event_path: str | Path) -> dict:
    return yaml.safe_load(Path(event_path).read_text(encoding="utf-8"))


def generate_ground_truth(
    event: dict, reg: ParameterRegistry, gravity_axis: int = 2,
    dyn_fingers: list[str] | None = None,
) -> GroundTruth:
    # Dynamics (LIS) fingers come from the topology descriptor; default to the
    # Layout-B tripod for backward compatibility.
    if dyn_fingers is None:
        dyn_fingers = ["thumb", "index", "middle"]
    fs = float(reg.get("sync.master_rate_hz"))
    dur = float(event["duration_s"])
    n = max(2, int(round(dur * fs)))
    t = np.arange(n) / fs
    t_frac = t / dur

    fingers = list(event.get("fingers", []))
    prof = event["profile"]
    normal_n = _pw_linear(prof["normal"], t_frac)
    shear_n = _pw_linear(prof["shear"], t_frac)
    shear_dir = np.deg2rad(float(prof.get("shear_dir_deg", 0.0)))

    slip_cfg = event.get("slip", {"enabled": False})

    normal, shear_x, shear_y = {}, {}, {}
    contact, slip, slip_vel, loc = {}, {}, {}, {}
    for f in fingers:
        normal[f] = normal_n.copy()
        shear_x[f] = shear_n * np.cos(shear_dir)
        shear_y[f] = shear_n * np.sin(shear_dir)
        contact[f] = normal_n > 1e-3
        s = np.zeros(n, bool)
        sv = np.zeros(n, float)
        if slip_cfg.get("enabled"):
            onset = int(round(float(slip_cfg.get("onset_frac", 0.6)) * n))
            s[onset:] = contact[f][onset:]
            sv[onset:] = float(slip_cfg.get("slip_velocity_norm", 1.0))
        slip[f] = s
        slip_vel[f] = sv
        loc[f] = np.column_stack([shear_x[f], shear_y[f]])  # contact drifts with shear

    # Dynamics ground-truth acceleration on tripod LIS sites (gravity + optional
    # placeholder slip vibration). Gravity = 1 g along the chosen axis.
    accel = {}
    grav = np.zeros(3)
    grav[gravity_axis] = 1.0
    vib_on = bool(slip_cfg.get("vibration_enabled", False))
    for f in dyn_fingers:
        a = np.tile(grav, (n, 1))
        if f in fingers and slip_cfg.get("enabled") and vib_on:
            fv = float(slip_cfg.get("vibration_freq_hz", 250.0))
            amp = float(slip_cfg.get("vibration_amp_g", 0.05))
            burst = (slip[f].astype(float)) * amp * np.sin(2 * np.pi * fv * t)
            a[:, 0] += burst
        accel[f] = a

    temp = np.full(n, float(reg.get("env.temp_c")))

    phase_id = np.zeros(n, int)
    for tfrac, label in event.get("phases", [[0.0, "idle"]]):
        phase_id[t_frac >= float(tfrac)] = PHASE_ID[label]

    # Coarse object translation stub: object lifts during transport phase.
    obj = np.zeros((n, 3))
    transport = phase_id == PHASE_ID["transport"]
    obj[transport, gravity_axis] = np.linspace(0, 20.0, int(transport.sum())) if transport.any() else 0.0

    return GroundTruth(
        t_master_s=t,
        fingers=fingers,
        normal=normal,
        shear_x=shear_x,
        shear_y=shear_y,
        contact=contact,
        slip=slip,
        slip_vel=slip_vel,
        contact_loc=loc,
        accel_true_g=accel,
        temp_true_c=temp,
        phase_id=phase_id,
        object_pos_mm=obj,
    )
