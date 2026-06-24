"""Pipeline orchestration: L0 -> L1 -> L2 -> L3/L4 -> L6 -> decode -> Episode.

Assembles raw sensor streams, ground-truth labels, decoded-estimate labels, and
a full provenance/metadata block (parameter tiers, confidence, physics_fidelity,
seed, versions).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .core_types import PHASES, Episode
from .decode import decode_contacts
from .events import generate_ground_truth, load_event
from .layers import l1_contact, l3_bmm350, l4_lis2dtw12, l6_sync
from .layers.l2_field import FieldModel
from .layers.l5_noise_drift import NoiseModel
from .params import ParameterRegistry
from .topology import from_descriptor

from sentrix_contracts import bundled_descriptor_path, load_descriptor


def _label(ep: Episode, name: str, arr, source: str, units: str,
           confidence: float, tier: str):
    ep.labels[name] = np.asarray(arr)
    ep.label_meta[name] = {
        "source": source, "units": units, "confidence": confidence, "tier": tier
    }


def simulate(
    event_name: str,
    config_dir: str | Path,
    seed: int = 0,
    allow_placeholders: bool = False,
    scene_path: str | Path | None = None,
    drift_seed: int | None = None,
    duration_s: float | None = None,
    descriptor: str | Path | None = None,
) -> Episode:
    config_dir = Path(config_dir)
    reg = ParameterRegistry.load(
        config_dir / "parameters.yaml", allow_placeholders=allow_placeholders
    )
    scene = yaml.safe_load(
        Path(scene_path or config_dir / "scene_default.yaml").read_text(encoding="utf-8")
    )
    # scene environment overrides
    env = scene.get("environment", {})
    if "temp_c" in env:
        reg.param("env.temp_c").value = env["temp_c"]
    if "B0_uT" in env:
        reg.param("env.B0_uT").value = env["B0_uT"]

    # Topology is descriptor-driven (Migration Phase 1). Default = bundled
    # Mark2_v1; pass `descriptor` (a version name or a path) for any other layout.
    if descriptor is None:
        desc_path = bundled_descriptor_path("Mark2_v1")
    else:
        desc_path = Path(descriptor)
        if not desc_path.exists():
            desc_path = bundled_descriptor_path(str(descriptor))
    desc = load_descriptor(desc_path)
    topo = from_descriptor(desc)
    dyn_fingers = [s.finger for s in topo.lis_sites]
    event = load_event(config_dir / "events" / f"{event_name}.yaml")
    if duration_s is not None:
        event = {**event, "duration_s": float(duration_s)}

    # L0..L6
    gt = generate_ground_truth(event, reg, dyn_fingers=dyn_fingers)
    n = gt.t_master_s.shape[0]
    deform = l1_contact.run(gt, reg, scene)
    field = FieldModel(topo, reg, scene)
    B_true = field.run(deform, n)
    noise = NoiseModel(seed, drift_seed=drift_seed)
    bmm_out = l3_bmm350.run(B_true, reg, noise, local_frames=topo.bmm_local_frames)
    lis_out = l4_lis2dtw12.run(gt.accel_true_g, gt.temp_true_c, reg, noise, scene,
                               dyn_fingers=dyn_fingers)
    aligned = l6_sync.run(n, bmm_out, lis_out, reg)

    ep = Episode(name=event_name, meta={}, t_master_us=aligned["t_master_us"])
    ep.aligned = {
        "B_read_uT": aligned["B_read_uT"],
        "B_lsb": aligned["B_lsb"],
        "sat_flag": aligned["sat_flag"],
        "bmm_valid": aligned["bmm_valid"],
        "accel_read_g": aligned["accel_read_g"],
        "accel_lsb": aligned["accel_lsb"],
        "temp_read_c": aligned["temp_read_c"],
        "temp_valid": aligned["temp_valid"],
        "phase_id": gt.phase_id,
    }

    # ---- ground-truth labels ----
    _label(ep, "phase", gt.phase_id, "ground_truth", "enum", 1.0, "KNOWN")
    for f in gt.fingers:
        _label(ep, f"label.{f}.normal_force", gt.normal[f], "ground_truth",
                "normalized", 1.0, "KNOWN")
        _label(ep, f"label.{f}.shear_x", gt.shear_x[f], "ground_truth",
                "normalized", 1.0, "KNOWN")
        _label(ep, f"label.{f}.shear_y", gt.shear_y[f], "ground_truth",
                "normalized", 1.0, "KNOWN")
        _label(ep, f"label.{f}.contact", gt.contact[f], "ground_truth",
                "bool", 1.0, "KNOWN")
        _label(ep, f"label.{f}.slip", gt.slip[f], "ground_truth",
                "bool", 1.0, "KNOWN")
        _label(ep, f"label.{f}.slip_velocity", gt.slip_vel[f], "ground_truth",
                "normalized", 0.3, "UNKNOWN")  # absolute slip velocity unobservable

    # ---- decoded-estimate labels (inverse demo) ----
    decoded = decode_contacts(ep.aligned, topo, reg, scene)
    for f, d in decoded.items():
        _label(ep, f"est.{f}.contact", d["contact"], "simulated_estimate",
                "bool", 0.6, "ESTIMATED")
        _label(ep, f"est.{f}.force_proxy", d["force_proxy"], "simulated_estimate",
                "normalized", 0.5, "ESTIMATED")

    # ---- metadata + provenance ----
    ep.provenance = reg.provenance_table()
    ep.meta = {
        "sim_version": __version__,
        "event": event_name,
        "event_description": event.get("description", ""),
        "seed": seed,
        "drift_seed": drift_seed,
        "drift_applied": drift_seed is not None,
        "duration_s": float(event["duration_s"]),
        "n_samples": n,
        "master_rate_hz": float(reg.get("sync.master_rate_hz")),
        "field_rate_hz": float(reg.get("sync.field_rate_hz")),
        "n_bmm350": topo.n_bmm,
        "n_lis2dtw12": topo.n_lis,
        "layout": topo.name,
        "descriptor_version": desc.descriptor_version,
        "descriptor_hash": desc.descriptor_hash,
        "bmm_sensor_ids": [s.sensor_id for s in topo.bmm_sites],
        "lis_sensor_ids": [s.sensor_id for s in topo.lis_sites],
        "physics_fidelity": reg.physics_fidelity(),
        "sync_quality": "simulated-target",
        "allow_placeholders": allow_placeholders,
        "param_counts": reg.counts(),
        "phase_labels": PHASES,
        "fingers_in_contact": gt.fingers,
        "units": {
            "B_read_uT": "uT", "accel_read_g": "g", "temp_read_c": "degC",
            "t_master_us": "us", "forces": "normalized (absolute N UNKNOWN)",
        },
    }
    return ep
