"""Sentrix Dataset builder.

Plans and executes a balanced multi-event dataset using the SentrixSim pipeline,
writing Parquet (per-episode) + a consolidated LeRobot v3 dataset + MCAP logs,
then computes statistics, runs integrity validation, and emits a Markdown report.

Determinism: every episode's (event, duration_s, noise_seed, drift_seed) is
derived from a single master_seed + matrix indices and recorded in manifest.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from . import __version__
from .core_types import PHASES
from .hardmode import build_hard_episode, load_hard_cfg, sample_session_cal
from .layers.l7_export import mcap, parquet
from .layers.l7_export.lerobot_dataset import LeRobotDatasetWriter
from .params import ParameterRegistry
from .pipeline import simulate
from .topology import build_topology

EVENTS = ["idle", "tap", "press", "hold", "shear", "slip", "release", "pinch", "grasp"]
DURATION_MULTS = [0.75, 1.0, 1.25, 1.5, 2.0]
FIELD_RATIO = 4  # master 1600 / field 400


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_dataset(config_dir, events=EVENTS, duration_mults=DURATION_MULTS,
                 n_noise=5, n_drift=4, master_seed=20260601):
    config_dir = Path(config_dir)
    specs = []
    for e_idx, ev in enumerate(events):
        base = float(yaml.safe_load(
            (config_dir / "events" / f"{ev}.yaml").read_text(encoding="utf-8"))["duration_s"])
        for di, mult in enumerate(duration_mults):
            dur = round(base * mult, 4)
            for ni in range(n_noise):
                for ri in range(n_drift):
                    noise_seed = master_seed + e_idx * 100000 + di * 10000 + ni * 100 + ri
                    specs.append({
                        "event": ev,
                        "episode_id": f"{ev}__d{di}_n{ni}_r{ri}",
                        "duration_s": dur,
                        "duration_mult": mult,
                        "noise_seed": noise_seed,
                        "drift_seed": noise_seed + 50_000_000,
                        "idx": {"dur": di, "noise": ni, "drift": ri},
                    })
    return specs


# --------------------------------------------------------------------------- #
# Running statistics
# --------------------------------------------------------------------------- #
class _Acc:
    """Streaming min/max/sum/sumsq."""
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.lo = np.inf
        self.hi = -np.inf

    def update(self, x: np.ndarray):
        x = np.asarray(x, float).ravel()
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float((x * x).sum())
        if x.size:
            self.lo = min(self.lo, float(x.min()))
            self.hi = max(self.hi, float(x.max()))

    def summary(self):
        if self.n == 0:
            return {"n": 0}
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        return {"n": self.n, "min": self.lo, "max": self.hi,
                "mean": mean, "std": var ** 0.5}


def _f(x):
    """JSON-safe scalar."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_dataset(config_dir, out_root, version="0.1", events=EVENTS,
                  duration_mults=DURATION_MULTS, n_noise=5, n_drift=4,
                  master_seed=20260601, formats=("parquet", "mcap", "lerobot"),
                  mcap_stride=1, hard_mode=False):
    config_dir = Path(config_dir)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parquet").mkdir(exist_ok=True)
    (out / "mcap").mkdir(exist_ok=True)

    specs = plan_dataset(config_dir, events, duration_mults, n_noise, n_drift, master_seed)
    t0 = time.time()

    # ---- hard-mode setup: sessions, calibration, hard-negative designation ----
    hm = session_cals = hardneg_set = None
    n_sessions = 0
    if hard_mode:
        hm = load_hard_cfg(config_dir)
        reg0 = ParameterRegistry.load(config_dir / "parameters.yaml")
        topo0 = build_topology(config_dir / "topology_layoutB.yaml", reg0)
        nb = topo0.n_bmm
        n_sessions = int(hm["session"]["n_sessions"])
        session_cals = {sid: sample_session_cal(master_seed + 1000 + sid, nb, hm)
                        for sid in range(n_sessions)}
        idle_idx = [i for i, s in enumerate(specs) if s["event"] == "idle"]
        n_hn = int(len(idle_idx) * hm["hard_negative"]["idle_fraction"])
        hardneg_set = set(idle_idx[:n_hn])

    lr = LeRobotDatasetWriter(out / "lerobot", fps=1600.0) if "lerobot" in formats else None

    # accumulators
    counts = {ev: 0 for ev in events}
    bacc = {ax: _Acc() for ax in ("bx", "by", "bz")}
    aacc = {ax: _Acc() for ax in ("ax", "ay", "az")}
    tacc = _Acc()
    sat_total = sat_count = 0
    drop_total = drop_count = 0
    phase_hist = np.zeros(len(PHASES), int)
    per_event_phase = {ev: np.zeros(len(PHASES), int) for ev in events}
    contact_frac = {ev: [] for ev in events}
    slip_frac = {ev: [] for ev in events}
    durations_s = []
    dt_lo, dt_hi = np.inf, -np.inf

    manifest = []
    val = {"missing_channels": [], "nan_episodes": [], "sync_failures": [],
           "export_failures": [], "n_checked": 0}

    # Canonical sensor_id-keyed columns, derived from the topology descriptor
    # (SIM-3 retired the legacy tactile.bNN / dyn.<finger> shim). Count-agnostic.
    from sentrix_contracts import bundled_descriptor_path, load_descriptor

    from .layers.l7_export.schema import accel_columns, tactile_columns, temp_columns
    _desc = load_descriptor(bundled_descriptor_path("Mark2_v1"))
    _bmm = [s.sensor_id for s in _desc.sensors.values() if s.modality == "magnetic"]
    _lis = [s.sensor_id for s in _desc.sensors.values() if s.modality == "dynamics"]
    expected_sensor_cols = (
        {"t_master_us", "bmm_valid", "temp_valid", "sat_any", "phase_id"}
        | set(tactile_columns(_bmm)) | set(accel_columns(_lis)) | set(temp_columns(_lis))
    )

    n_hardneg_done = 0
    for i, spec in enumerate(specs):
        ev = spec["event"]
        if hard_mode:
            is_hn = i in hardneg_set
            n_hardneg_done += int(is_hn)
            ep = build_hard_episode(
                ev, config_dir, noise_seed=spec["noise_seed"],
                drift_seed=spec["drift_seed"], style_seed=spec["noise_seed"] + 900000,
                dropout_seed=spec["noise_seed"] + 1900000,
                session_cal=session_cals[i % n_sessions],
                duration_s=spec["duration_s"], is_hard_neg=is_hn)
        else:
            ep = simulate(ev, config_dir, seed=spec["noise_seed"],
                          drift_seed=spec["drift_seed"], duration_s=spec["duration_s"])
        ep.name = spec["episode_id"]

        # ---- stats (in memory) ----
        counts[ev] += 1
        B = ep.aligned["B_read_uT"]
        A = ep.aligned["accel_read_g"]
        T = ep.aligned["temp_read_c"]
        bacc["bx"].update(B[:, :, 0]); bacc["by"].update(B[:, :, 1]); bacc["bz"].update(B[:, :, 2])
        aacc["ax"].update(A[:, :, 0]); aacc["ay"].update(A[:, :, 1]); aacc["az"].update(A[:, :, 2])
        tacc.update(T)
        sat_total += ep.aligned["sat_flag"].size
        sat_count += int(ep.aligned["sat_flag"].sum())
        if "dropout" in ep.aligned:
            drop_total += ep.aligned["dropout"].size
            drop_count += int(ep.aligned["dropout"].sum())
        ph = ep.aligned["phase_id"]
        hc = np.bincount(ph, minlength=len(PHASES))
        phase_hist += hc
        per_event_phase[ev] += hc
        for f in ep.meta["fingers_in_contact"]:
            contact_frac[ev].append(float(ep.labels[f"label.{f}.contact"].mean()))
            slip_frac[ev].append(float(ep.labels[f"label.{f}.slip"].mean()))
        durations_s.append(float(ep.t_master_us[-1] * 1e-6))

        # ---- validation ----
        val["n_checked"] += 1
        # NaNs
        if (np.isnan(B).any() or np.isnan(A).any() or np.isnan(T).any()):
            val["nan_episodes"].append(spec["episode_id"])
        # sync integrity
        tus = ep.t_master_us
        dt = np.diff(tus)
        ok_mono = bool(np.all(dt > 0))
        if hard_mode:
            # jitter-aware: monotonic + bounded dt (base 625 us +/- ~4x jitter max)
            ok_dt = bool(dt.size == 0 or np.all((dt >= 50) & (dt <= 1700)))
        else:
            ok_mono = ok_mono and int(tus[0]) == 0
            ok_dt = bool(np.all(dt == dt[0])) if dt.size else True
        bvalid_idx = np.flatnonzero(ep.aligned["bmm_valid"])
        ok_bvalid = bool(np.array_equal(bvalid_idx, np.arange(0, ep.n_samples, FIELD_RATIO)))
        if not (ok_mono and ok_dt and ok_bvalid):
            val["sync_failures"].append(spec["episode_id"])
        if dt.size:
            dt_lo = min(dt_lo, float(dt.min())); dt_hi = max(dt_hi, float(dt.max()))

        # ---- exports ----
        if "parquet" in formats:
            ppath = parquet.write(ep, out / "parquet" / ev)
            cols = set(pq.ParquetFile(ppath).schema.names)
            missing = expected_sensor_cols - cols
            if missing:
                val["missing_channels"].append({"episode": spec["episode_id"],
                                                "missing": sorted(missing)})
            if pq.ParquetFile(ppath).metadata.num_rows != ep.n_samples:
                val["export_failures"].append(f"{spec['episode_id']}:parquet_rowcount")
        if lr is not None:
            lr.append(ep)
        if "mcap" in formats and (i % mcap_stride == 0):
            mpath = mcap.write(ep, out / "mcap" / ev)
            if not (mpath.exists() and mpath.stat().st_size > 0):
                val["export_failures"].append(f"{spec['episode_id']}:mcap")

        manifest.append({
            "episode_id": spec["episode_id"], "event": ev,
            "duration_s": spec["duration_s"], "duration_mult": spec["duration_mult"],
            "noise_seed": spec["noise_seed"], "drift_seed": spec["drift_seed"],
            "n_samples": ep.n_samples,
            "physics_fidelity": ep.meta["physics_fidelity"],
            "fingers_in_contact": ep.meta["fingers_in_contact"],
            "session": (i % n_sessions) if hard_mode else None,
            "is_hard_negative": bool(ep.meta.get("is_hard_negative", False)),
            "partial_contact": bool(ep.meta.get("partial_contact", False)),
        })

    lr_root = lr.finalize(extra_meta={"dataset_version": version}) if lr is not None else None
    elapsed = time.time() - t0

    # ---- storage ----
    def dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
    storage = {
        "parquet_bytes": dir_size(out / "parquet"),
        "mcap_bytes": dir_size(out / "mcap"),
        "lerobot_bytes": dir_size(out / "lerobot"),
    }
    storage["total_bytes"] = sum(storage.values())

    # ---- lerobot frame-total integrity ----
    if lr_root is not None:
        info = json.loads((lr_root / "meta" / "info.json").read_text(encoding="utf-8"))
        total_frames_manifest = sum(m["n_samples"] for m in manifest)
        if info["total_frames"] != total_frames_manifest:
            val["export_failures"].append("lerobot_total_frames_mismatch")

    stats = {
        "episode_counts": counts,
        "total_episodes": len(specs),
        "class_balance": {ev: round(counts[ev] / len(specs), 4) for ev in events},
        "channel_stats": {
            "B_uT": {k: {kk: _f(vv) for kk, vv in v.summary().items()} for k, v in bacc.items()},
            "accel_g": {k: {kk: _f(vv) for kk, vv in v.summary().items()} for k, v in aacc.items()},
            "temp_degC": {kk: _f(vv) for kk, vv in tacc.summary().items()},
        },
        "sensor_range": {
            "B_within_pm2000uT": bool(bacc["bx"].lo >= -2000 and bacc["bx"].hi <= 2000
                                      and bacc["by"].lo >= -2000 and bacc["by"].hi <= 2000
                                      and bacc["bz"].lo >= -2000 and bacc["bz"].hi <= 2000),
            "accel_within_pm16g": bool(min(aacc[a].lo for a in aacc) >= -16
                                       and max(aacc[a].hi for a in aacc) <= 16),
            "temp_min": _f(tacc.lo), "temp_max": _f(tacc.hi),
            "saturated_fraction": round(sat_count / max(sat_total, 1), 8),
            "dropout_fraction": round(drop_count / max(drop_total, 1), 8),
        },
        "hard_mode": hard_mode,
        "n_hard_negatives": n_hardneg_done,
        "n_sessions": n_sessions,
        "timestamp_stats": {
            "expected_dt_us": 1e6 / 1600.0,
            "observed_dt_min_us": _f(dt_lo), "observed_dt_max_us": _f(dt_hi),
            "duration_s_min": round(min(durations_s), 4),
            "duration_s_max": round(max(durations_s), 4),
        },
        "label_distribution": {
            "phase_hist_global": {PHASES[i]: int(phase_hist[i]) for i in range(len(PHASES))},
            "phase_hist_per_event": {ev: {PHASES[i]: int(per_event_phase[ev][i])
                                          for i in range(len(PHASES))} for ev in events},
            "mean_contact_fraction": {ev: round(float(np.mean(contact_frac[ev])), 4)
                                      if contact_frac[ev] else 0.0 for ev in events},
            "mean_slip_fraction": {ev: round(float(np.mean(slip_frac[ev])), 4)
                                   if slip_frac[ev] else 0.0 for ev in events},
        },
        "storage": storage,
        "elapsed_s": round(elapsed, 1),
    }
    validation = {
        "n_checked": val["n_checked"],
        "missing_channels": val["missing_channels"],
        "nan_episodes": val["nan_episodes"],
        "sync_failures": val["sync_failures"],
        "export_failures": val["export_failures"],
        "all_passed": not any([val["missing_channels"], val["nan_episodes"],
                               val["sync_failures"], val["export_failures"]]),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    _write_report(out, version, events, duration_mults, n_noise, n_drift,
                  master_seed, formats, mcap_stride, stats, validation)
    return {"stats": stats, "validation": validation, "out": str(out)}


def _write_report(out, version, events, duration_mults, n_noise, n_drift,
                  master_seed, formats, mcap_stride, stats, validation):
    gb = stats["storage"]["total_bytes"] / 1e9
    lines = []
    A = lines.append
    A(f"# Sentrix Dataset v{version} - Report\n")
    A(f"Generated by SentrixSim v{__version__}. Physics fidelity: **relative / "
      f"shape-only** (absolute force/field magnitudes gated behind UNKNOWN "
      f"parameters; see SentrixSim README).\n")

    A("## 1. Generation settings\n")
    A(f"- Events: {', '.join(events)}")
    A(f"- Matrix per event: {len(duration_mults)} durations x {n_noise} noise "
      f"seeds x {n_drift} drift seeds = {len(duration_mults)*n_noise*n_drift}")
    A(f"- Total episodes: **{stats['total_episodes']}**")
    A(f"- Duration multipliers (x base): {duration_mults}")
    A(f"- Master seed: {master_seed} (per-episode noise_seed/drift_seed derived "
      f"deterministically; full list in manifest.json)")
    A(f"- Formats: {', '.join(formats)} (MCAP stride = {mcap_stride})")
    A(f"- Master rate 1600 Hz; BMM350 field rate 400 Hz; LIS2DTW12 1600 Hz / "
      f"temp 50 Hz")
    A(f"- Generation time: {stats['elapsed_s']} s\n")

    A("## 2. Episode counts & class balance\n")
    A("| Event | Episodes | Fraction |")
    A("|---|---|---|")
    for ev in events:
        A(f"| {ev} | {stats['episode_counts'][ev]} | "
          f"{stats['class_balance'][ev]:.3f} |")
    A("")

    A("## 3. Storage\n")
    s = stats["storage"]
    A(f"- Parquet: {s['parquet_bytes']/1e6:.1f} MB")
    A(f"- LeRobot: {s['lerobot_bytes']/1e6:.1f} MB")
    A(f"- MCAP: {s['mcap_bytes']/1e6:.1f} MB")
    A(f"- **Total: {gb:.2f} GB**\n")

    A("## 4. Channel statistics\n")
    A("| Channel | n | min | max | mean | std |")
    A("|---|---|---|---|---|---|")
    for ax, st in stats["channel_stats"]["B_uT"].items():
        A(f"| B.{ax} (uT) | {st['n']} | {st['min']:.3f} | {st['max']:.3f} | "
          f"{st['mean']:.4f} | {st['std']:.4f} |")
    for ax, st in stats["channel_stats"]["accel_g"].items():
        A(f"| accel.{ax} (g) | {st['n']} | {st['min']:.3f} | {st['max']:.3f} | "
          f"{st['mean']:.4f} | {st['std']:.4f} |")
    t = stats["channel_stats"]["temp_degC"]
    A(f"| temp (degC) | {t['n']} | {t['min']:.3f} | {t['max']:.3f} | "
      f"{t['mean']:.4f} | {t['std']:.4f} |\n")

    A("## 5. Timestamp statistics\n")
    ts = stats["timestamp_stats"]
    A(f"- Expected dt: {ts['expected_dt_us']:.3f} us  (1600 Hz)")
    A(f"- Observed dt: [{ts['observed_dt_min_us']:.3f}, {ts['observed_dt_max_us']:.3f}] us")
    A(f"- Episode duration range: [{ts['duration_s_min']}, {ts['duration_s_max']}] s\n")

    A("## 6. Sensor range statistics\n")
    sr = stats["sensor_range"]
    A(f"- B within +/-2000 uT: **{sr['B_within_pm2000uT']}**")
    A(f"- accel within +/-16 g: **{sr['accel_within_pm16g']}**")
    A(f"- temp range: [{sr['temp_min']:.2f}, {sr['temp_max']:.2f}] degC")
    A(f"- saturated fraction: {sr['saturated_fraction']}\n")

    A("## 7. Label distributions\n")
    A("Global phase-sample histogram:")
    A("| Phase | Samples |")
    A("|---|---|")
    for ph, c in stats["label_distribution"]["phase_hist_global"].items():
        A(f"| {ph} | {c} |")
    A("\nPer-event mean contact / slip sample fraction:\n")
    A("| Event | mean contact frac | mean slip frac |")
    A("|---|---|---|")
    ld = stats["label_distribution"]
    for ev in events:
        A(f"| {ev} | {ld['mean_contact_fraction'][ev]} | {ld['mean_slip_fraction'][ev]} |")
    A("")

    A("## 8. Validation\n")
    A(f"- Episodes checked: {validation['n_checked']}")
    A(f"- Missing channels: {len(validation['missing_channels'])}")
    A(f"- Episodes with NaNs: {len(validation['nan_episodes'])}")
    A(f"- Synchronization failures: {len(validation['sync_failures'])}")
    A(f"- Export failures: {len(validation['export_failures'])}")
    A(f"- **All passed: {validation['all_passed']}**\n")

    A("## 9. Assumptions\n")
    A("- **Relative/shape-only physics**: signal timing, noise, quantization, "
      "saturation and topology are datasheet/spec-true; absolute force (N) and "
      "field (uT) magnitudes are NOT physical - forces are normalized and field "
      "uses the ESTIMATED `mag.field_scale_uT` presentation scale.")
    A("- **Drift realizations** are a per-episode static per-sensor offset "
      "(`bmm.offset_spread_uT` = 0.3 uT, `lis.offset_spread_mg` = 5 mg), an "
      "ESTIMATED stand-in (confidence 0.30) for the UNKNOWN per-unit calibration "
      "offset spread - NOT a measured drift PSD.")
    A("- **Noise** is white Gaussian at the datasheet RMS (190/450 nT; 90 ug/rtHz); "
      "1/f drift and temperature coefficients remain OFF (UNKNOWN/to-confirm).")
    A("- **Slip vibration** is OFF (coupling UNKNOWN); slip episodes carry "
      "ground-truth slip flags but no synthetic >400 Hz burst.")
    A("- Gestures are scripted profiles, not recorded human motion.\n")

    A("## 10. Limitations\n")
    A("- No absolute accuracy: a model trained on magnitudes will not transfer to "
      "hardware until the calibration bundle (BUILD Part 8, 25-pt 6-DoF grid) "
      "replaces the relative scales.")
    A("- No vision/RGB-D stream (the glove carries no camera; genlock is "
      "architectural). LeRobot export has no video features.")
    A("- Single-instance: cross-unit variability is approximated only by the "
      "drift-offset realizations.")
    A("- MCAP `[R,U,V]` is not materialized; sparse clusters are native.\n")

    A("## 11. Recommended next dataset scale\n")
    A("- **v0.2 (calibrated-ready)**: after first-article measurements, fold in "
      "measured `mag.Br_mT`, elastomer moduli, friction, and the per-unit "
      "calibration bundle; switch to absolute mode and re-issue with the same "
      "matrix so v0.1 pipelines transfer unchanged.")
    A("- **Scale**: 500-1000 episodes/event (5-9k total) with 10+ noise and 8+ "
      "drift realizations, plus recorded-motion variability, once absolute "
      "physics lands. Add genlocked RGB-D when the capture rig exists.")
    A("- **Storage planning**: this run is "
      f"~{gb:.2f} GB for {stats['total_episodes']} episodes "
      f"(~{gb/ max(stats['total_episodes'],1)*1000:.1f} MB/episode); budget "
      "linearly for larger runs and shard/partition by event+date.")

    (out / f"dataset_v{version}_report.md").write_text("\n".join(lines), encoding="utf-8")
