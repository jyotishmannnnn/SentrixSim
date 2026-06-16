"""SentrixSim command-line interface.

    sentrixsim simulate --event tap --out ./out --formats parquet,mcap,lerobot
    sentrixsim list-events
    sentrixsim show-params [--tier UNKNOWN]
    sentrixsim simulate-all --out ./out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .params import ParameterRegistry
from .pipeline import simulate

EVENTS = ["idle", "tap", "press", "hold", "shear", "slip", "release", "pinch", "grasp"]


def _default_config_dir() -> Path:
    # configs/ sits at the repo root, two levels above this file's package dir.
    return Path(__file__).resolve().parents[2] / "configs"


def _export(ep, out_dir: Path, formats: list[str]) -> list[str]:
    from .layers.l7_export import lerobot, mcap, parquet
    written = []
    if "parquet" in formats:
        written.append(str(parquet.write(ep, out_dir)))
    if "mcap" in formats:
        written.append(str(mcap.write(ep, out_dir)))
    if "lerobot" in formats:
        written.append(str(lerobot.write(ep, out_dir)))
    return written


def cmd_simulate(args) -> int:
    cfg = Path(args.config_dir) if args.config_dir else _default_config_dir()
    ep = simulate(args.event, cfg, seed=args.seed,
                  allow_placeholders=args.allow_placeholders)
    out = Path(args.out)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    written = _export(ep, out, formats)
    print(json.dumps({
        "event": ep.name,
        "n_samples": ep.n_samples,
        "physics_fidelity": ep.meta["physics_fidelity"],
        "param_counts": ep.meta["param_counts"],
        "written": written,
    }, indent=2))
    return 0


def cmd_simulate_all(args) -> int:
    cfg = Path(args.config_dir) if args.config_dir else _default_config_dir()
    out = Path(args.out)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for ev in EVENTS:
        ep = simulate(ev, cfg, seed=args.seed,
                      allow_placeholders=args.allow_placeholders)
        _export(ep, out / ev, formats)
        print(f"  {ev:8s} {ep.n_samples:6d} samples -> {out / ev}")
    return 0


def cmd_build_dataset(args) -> int:
    from .dataset import build_dataset
    cfg = Path(args.config_dir) if args.config_dir else _default_config_dir()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    res = build_dataset(
        cfg, args.out, version=args.version, n_noise=args.n_noise,
        n_drift=args.n_drift, master_seed=args.master_seed,
        formats=formats, mcap_stride=args.mcap_stride, hard_mode=args.hard_mode,
    )
    print(json.dumps({
        "out": res["out"],
        "total_episodes": res["stats"]["total_episodes"],
        "storage_gb": round(res["stats"]["storage"]["total_bytes"] / 1e9, 3),
        "elapsed_s": res["stats"]["elapsed_s"],
        "validation_all_passed": res["validation"]["all_passed"],
    }, indent=2))
    return 0 if res["validation"]["all_passed"] else 2


def cmd_list_events(args) -> int:
    for e in EVENTS:
        print(e)
    return 0


def cmd_show_params(args) -> int:
    cfg = Path(args.config_dir) if args.config_dir else _default_config_dir()
    reg = ParameterRegistry.load(cfg / "parameters.yaml")
    rows = reg.provenance_table()
    if args.tier:
        rows = [r for r in rows if r["tier"] == args.tier]
    for r in rows:
        print(f"[{r['tier']:9s}] {r['name']:28s} = {r['value']} {r['units']:14s} "
              f"conf={r['confidence']:.2f}  {r['origin']}")
    print(f"\ncounts: {reg.counts()}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sentrixsim",
                                description=f"SentrixSim v{__version__}")
    p.add_argument("--config-dir", default=None, help="override configs/ directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("simulate", help="simulate one gesture")
    sp.add_argument("--event", required=True, choices=EVENTS)
    sp.add_argument("--out", required=True)
    sp.add_argument("--formats", default="parquet,mcap,lerobot")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--allow-placeholders", action="store_true")
    sp.set_defaults(func=cmd_simulate)

    sa = sub.add_parser("simulate-all", help="simulate all 9 gestures")
    sa.add_argument("--out", required=True)
    sa.add_argument("--formats", default="parquet,mcap,lerobot")
    sa.add_argument("--seed", type=int, default=0)
    sa.add_argument("--allow-placeholders", action="store_true")
    sa.set_defaults(func=cmd_simulate_all)

    bd = sub.add_parser("build-dataset", help="generate a balanced multi-event dataset")
    bd.add_argument("--out", required=True)
    bd.add_argument("--version", default="0.1")
    bd.add_argument("--n-noise", type=int, default=5)
    bd.add_argument("--n-drift", type=int, default=4)
    bd.add_argument("--master-seed", type=int, default=20260601)
    bd.add_argument("--formats", default="parquet,mcap,lerobot")
    bd.add_argument("--mcap-stride", type=int, default=1,
                    help="write MCAP every Nth episode (1 = all)")
    bd.add_argument("--hard-mode", action="store_true",
                    help="enable v0.2 Hard Mode realism augmentation")
    bd.set_defaults(func=cmd_build_dataset)

    sl = sub.add_parser("list-events", help="list available gestures")
    sl.set_defaults(func=cmd_list_events)

    ss = sub.add_parser("show-params", help="print the parameter registry")
    ss.add_argument("--tier", default=None, choices=["KNOWN", "ESTIMATED", "UNKNOWN"])
    ss.set_defaults(func=cmd_show_params)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
