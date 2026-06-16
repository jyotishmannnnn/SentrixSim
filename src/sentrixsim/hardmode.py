"""Hard Mode (Dataset v0.2): realistic-ambiguity augmentation.

Implements the 10 effects on top of the v0.1 forward chain. Each effect is an
ESTIMATED modelling choice configured in configs/scene_hard.yaml (no UNKNOWN
physical value is invented). Episodes are stamped physics_fidelity =
"relative+hardmode".

Effects: 1 multi-stage, 2 overlap (shared idle/reach/release context), 3 micro-
slip in holds, 4 partial contacts, 5 sensor dropouts, 6 timestamp jitter,
7 cross-finger coupling, 8 variable styles, 9 per-session calibration drift,
10 hard negatives (non-contact motion + field disturbance).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .core_types import PHASE_ID, PHASES, Episode, GroundTruth
from .decode import decode_contacts
from .events.generator import _pw_linear, load_event
from .layers import l1_contact, l3_bmm350, l4_lis2dtw12, l6_sync
from .layers.l2_field import FieldModel
from .layers.l5_noise_drift import NoiseModel
from .params import ParameterRegistry
from .topology import build_topology

TRIPOD = ["thumb", "index", "middle"]


def load_hard_cfg(config_dir) -> dict:
    p = Path(config_dir) / "scene_hard.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["hard_mode"]


def sample_session_cal(session_seed: int, n_bmm: int, hm: dict, temp_range=(22.0, 36.0)) -> dict:
    """Per-session non-identity calibration (effect #9)."""
    rng = np.random.default_rng(session_seed)
    s = hm["session"]
    gain = 1.0 + rng.normal(0, s["gain_spread"], size=(n_bmm, 3))
    offset = rng.normal(0, s["offset_spread_uT"], size=(n_bmm, 3))
    temp_c = float(rng.uniform(*temp_range))
    return {"gain": gain, "offset": offset, "temp_c": temp_c, "session_seed": session_seed}


def _dropout_mask(rng, n: int, nb: int, hm: dict) -> np.ndarray:
    """Per-episode sensor dropout spans (effect #5)."""
    mask = np.zeros((n, nb), bool)
    p = hm["dropout"]["prob_per_sensor"]
    lo, hi = hm["dropout"]["span_frac_range"]
    for j in range(nb):
        if rng.random() < p:
            span = int(rng.uniform(lo, hi) * n)
            start = int(rng.uniform(0, max(1, n - span)))
            mask[start:start + span, j] = True
    return mask


def _jitter(rng, n: int, hm: dict) -> np.ndarray:
    j = hm["jitter"]
    return np.clip(rng.normal(0, j["sigma_us"], n), -j["max_us"], j["max_us"])


def _event_path(config_dir, name):
    return Path(config_dir) / "events" / f"{name}.yaml"


def build_hard_episode(event_name, config_dir, *, noise_seed, drift_seed,
                       style_seed, dropout_seed, session_cal, duration_s,
                       is_hard_neg=False) -> Episode:
    config_dir = Path(config_dir)
    reg = ParameterRegistry.load(config_dir / "parameters.yaml")
    scene = yaml.safe_load((config_dir / "scene_default.yaml").read_text(encoding="utf-8"))
    hm = load_hard_cfg(config_dir)
    topo = build_topology(config_dir / "topology_layoutB.yaml", reg)
    reg.param("env.temp_c").value = session_cal["temp_c"]

    rng = np.random.default_rng(style_seed)
    fs = float(reg.get("sync.master_rate_hz"))
    n = max(2, int(round(duration_s * fs)))
    t = np.arange(n) / fs
    tfrac = t / duration_s

    core = load_event(_event_path(config_dir, event_name))
    prof = core["profile"]
    base_fingers = [] if is_hard_neg else list(core.get("fingers", []))

    # --- style (effect #8) ---
    st = hm["style"]
    amp = rng.uniform(*st["amp_range"])
    shamp = rng.uniform(*st["shear_amp_range"])
    warp = rng.uniform(*st["time_warp_range"])
    tremor_hz = rng.uniform(*st["tremor_hz_range"])
    shear_dir = np.deg2rad(float(prof.get("shear_dir_deg", 0.0))
                           + rng.uniform(-st["shear_dir_jitter_deg"], st["shear_dir_jitter_deg"]))

    # --- multi-stage embedding (effects #1, #2) ---
    ms = hm["multistage"]
    lead = rng.uniform(*ms["lead_idle_range"])
    span = rng.uniform(*ms["core_span_range"])
    s0 = min(lead, 0.4)
    s1 = min(s0 + span, 0.98)
    in_core = (tfrac >= s0) & (tfrac <= s1)
    u = np.clip((tfrac - s0) / max(s1 - s0, 1e-6), 0.0, 1.0)
    u_w = u ** warp

    normal_core = _pw_linear(prof["normal"], u_w) * amp
    shear_core = _pw_linear(prof["shear"], u_w) * shamp

    # --- partial contact (effect #4) ---
    partial = (not is_hard_neg) and base_fingers and (rng.random() < hm["partial_contact"]["prob"])
    decentre_norm = 0.0
    if partial:
        pc = hm["partial_contact"]
        normal_core *= rng.uniform(*pc["amp_scale_range"])
        shear_travel = float(scene["l1"]["shear_travel_mm"])
        decentre_norm = rng.uniform(*pc["decentre_mm_range"]) / shear_travel

    normal, shear_x, shear_y = {}, {}, {}
    contact, slip, slip_vel, loc = {}, {}, {}, {}
    for f in base_fingers:
        nrm = np.where(in_core, normal_core, 0.0).copy()
        # plateau tremor (effect #8)
        cmask = nrm > 0.02
        nrm = nrm + st["tremor_amp"] * np.sin(2 * np.pi * tremor_hz * t) * cmask
        nrm = np.clip(nrm, 0.0, None)
        sh = np.where(in_core, shear_core, 0.0)
        sx = sh * np.cos(shear_dir) + decentre_norm * cmask
        sy = sh * np.sin(shear_dir)
        normal[f] = nrm
        shear_x[f] = sx
        shear_y[f] = sy
        contact[f] = nrm > 0.02
        slip[f] = np.zeros(n, bool)
        slip_vel[f] = np.zeros(n, float)
        loc[f] = np.column_stack([sx, sy])

    # --- micro-slip during holds (effect #3) ---
    accel = {f: np.tile([0.0, 0.0, 1.0], (n, 1)) for f in TRIPOD}  # gravity
    if (not is_hard_neg) and event_name in hm["micro_slip"]["events"]:
        msc = hm["micro_slip"]
        for f in base_fingers:
            if rng.random() < msc["prob"] and contact[f].any():
                cidx = np.flatnonzero(contact[f])
                k = rng.integers(msc["count_range"][0], msc["count_range"][1] + 1)
                for _ in range(int(k)):
                    d = int(rng.uniform(*msc["dur_ms_range"]) * 1e-3 * fs)
                    c0 = int(rng.choice(cidx))
                    c1 = min(c0 + max(d, 1), n)
                    ex = rng.uniform(*msc["amp_range"])
                    shear_x[f][c0:c1] += ex * np.cos(shear_dir)
                    shear_y[f][c0:c1] += ex * np.sin(shear_dir)
                    if msc["label_as_slip"]:
                        slip[f][c0:c1] = True
                        slip_vel[f][c0:c1] = ex
                    if f in TRIPOD:  # faint vibration burst (the real, subtle cue)
                        seg = np.arange(c0, c1)
                        accel[f][c0:c1, 0] += 0.018 * np.sin(2 * np.pi * 220.0 * t[seg])

    # --- slip GESTURE: intrinsic slip after onset, distinguished from static
    #     shear only by a subtle >200 Hz vibration cue (shear magnitude now
    #     matches the shear gesture) - realistically hard (effect #3 core). ---
    if (not is_hard_neg) and event_name == "slip":
        for f in base_fingers:
            if contact[f].any():
                cidx = np.flatnonzero(contact[f])
                onset = cidx[int(0.4 * len(cidx))]
                slip[f][onset:] = contact[f][onset:]
                slip_vel[f][onset:] = 1.0
                if f in TRIPOD:
                    seg = np.arange(onset, n)
                    accel[f][onset:, 0] += 0.016 * np.sin(2 * np.pi * 240.0 * t[seg])

    # --- contact-transient ring at every contact ONSET (non-slip vibration):
    #     making/breaking contact rings, so vibration energy alone does NOT
    #     imply slip. This is the realistic confounder that makes slip hard. ---
    for f in base_fingers:
        if f in TRIPOD and contact[f].any():
            edges = np.flatnonzero((~contact[f][:-1]) & contact[f][1:]) + 1
            d = int(0.035 * fs)
            for e in edges:
                seg = np.arange(e, min(e + d, n))
                decay = np.exp(-(seg - e) / (0.012 * fs))
                accel[f][seg, 0] += 0.02 * np.sin(2 * np.pi * 260.0 * t[seg]) * decay

    # --- hard negative: non-contact motion + field disturbance (effect #10) ---
    field_bias = np.zeros((n, 3))
    hn = hm["hard_negative"]
    wander_amp = hn["field_disturb_uT"] * (1.0 if is_hard_neg else 0.25)
    wf = rng.uniform(0.5, 2.0)
    field_bias[:, 0] = wander_amp * np.sin(2 * np.pi * wf * t + rng.uniform(0, 6.28))
    field_bias[:, 2] = 0.6 * wander_amp * np.sin(2 * np.pi * (wf * 0.7) * t)
    if is_hard_neg:
        mhz = rng.uniform(*hn["motion_hz_range"])
        for f in TRIPOD:
            ph = rng.uniform(0, 6.28, 3)
            for ax in range(3):
                accel[f][:, ax] += hn["motion_accel_g"] * np.sin(2 * np.pi * mhz * t + ph[ax])

    # --- phases ---
    phase_id = np.zeros(n, int)
    if not is_hard_neg:
        for tf, label in core.get("phases", [[0.0, "idle"]]):
            phase_id[(u >= float(tf)) & in_core] = PHASE_ID[label]

    temp = np.full(n, float(reg.get("env.temp_c")))
    gt = GroundTruth(
        t_master_s=t, fingers=base_fingers, normal=normal, shear_x=shear_x,
        shear_y=shear_y, contact=contact, slip=slip, slip_vel=slip_vel,
        contact_loc=loc, accel_true_g=accel, temp_true_c=temp, phase_id=phase_id,
        object_pos_mm=np.zeros((n, 3)),
    )

    # ===== forward chain with hard-mode hooks =====
    deform = l1_contact.run(gt, reg, scene)
    field = FieldModel(topo, reg, scene)
    B_true = field.run(deform, n, coupling_gain=hm["coupling"]["neighbor_gain"])
    B_true = B_true + field_bias[:, None, :]
    noise = NoiseModel(noise_seed, drift_seed=drift_seed)
    drop_mask = _dropout_mask(np.random.default_rng(dropout_seed), n, topo.n_bmm, hm)
    bmm_out = l3_bmm350.run(B_true, reg, noise, session_cal=session_cal, dropout_mask=drop_mask)
    lis_out = l4_lis2dtw12.run(gt.accel_true_g, gt.temp_true_c, reg, noise, scene)
    jitter = _jitter(np.random.default_rng(noise_seed + 7), n, hm)
    aligned = l6_sync.run(n, bmm_out, lis_out, reg, jitter_us=jitter)

    ep = Episode(name=event_name, meta={}, t_master_us=aligned["t_master_us"])
    ep.aligned = {
        "B_read_uT": aligned["B_read_uT"], "B_lsb": aligned["B_lsb"],
        "sat_flag": aligned["sat_flag"], "dropout": aligned["dropout"],
        "bmm_valid": aligned["bmm_valid"], "accel_read_g": aligned["accel_read_g"],
        "accel_lsb": aligned["accel_lsb"], "temp_read_c": aligned["temp_read_c"],
        "temp_valid": aligned["temp_valid"], "phase_id": gt.phase_id,
    }

    _label = lambda name, arr, src, units, conf, tier: (
        ep.labels.__setitem__(name, np.asarray(arr)),
        ep.label_meta.__setitem__(name, {"source": src, "units": units,
                                         "confidence": conf, "tier": tier}))
    _label("phase", gt.phase_id, "ground_truth", "enum", 1.0, "KNOWN")
    for f in gt.fingers:
        _label(f"label.{f}.normal_force", normal[f], "ground_truth", "normalized", 1.0, "KNOWN")
        _label(f"label.{f}.shear_x", shear_x[f], "ground_truth", "normalized", 1.0, "KNOWN")
        _label(f"label.{f}.shear_y", shear_y[f], "ground_truth", "normalized", 1.0, "KNOWN")
        _label(f"label.{f}.contact", contact[f], "ground_truth", "bool", 1.0, "KNOWN")
        _label(f"label.{f}.slip", slip[f], "ground_truth", "bool", 1.0, "KNOWN")
        _label(f"label.{f}.slip_velocity", slip_vel[f], "ground_truth", "normalized", 0.3, "UNKNOWN")
    for f, d in decode_contacts(ep.aligned, topo, reg, scene).items():
        _label(f"est.{f}.contact", d["contact"], "simulated_estimate", "bool", 0.5, "ESTIMATED")
        _label(f"est.{f}.force_proxy", d["force_proxy"], "simulated_estimate", "normalized", 0.4, "ESTIMATED")

    ep.provenance = reg.provenance_table()
    ep.meta = {
        "sim_version": __version__, "event": event_name, "seed": noise_seed,
        "drift_seed": drift_seed, "style_seed": style_seed, "dropout_seed": dropout_seed,
        "session_seed": session_cal["session_seed"], "duration_s": float(duration_s),
        "n_samples": n, "master_rate_hz": fs, "field_rate_hz": float(reg.get("sync.field_rate_hz")),
        "n_bmm350": topo.n_bmm, "n_lis2dtw12": topo.n_lis, "layout": topo.name,
        "physics_fidelity": "relative+hardmode", "sync_quality": "simulated-jittered",
        "hard_mode": True, "is_hard_negative": is_hard_neg, "partial_contact": bool(partial),
        "param_counts": reg.counts(), "phase_labels": PHASES,
        "fingers_in_contact": gt.fingers,
        "units": {"B_read_uT": "uT", "accel_read_g": "g", "temp_read_c": "degC",
                  "t_master_us": "us", "forces": "normalized (absolute N UNKNOWN)"},
    }
    return ep
