# SentrixSim v1

Forward-model simulator for the **Sentrix Mark 2 visuotactile glove**, built
*before physical hardware exists*. It produces raw sensor streams + ground-truth
labels for data-pipeline, storage, labeling, and ML-prototyping work today, and
is designed to graduate into a **calibrated digital twin** once Mark 2 hardware
data lands — by editing config, not code.

**Status:** v1 implemented and tested (19/19 tests passing). Datasets **v0.1** and
**v0.2 Hard Mode** generated; baseline ML benchmarks run. Exports: Parquet, MCAP,
LeRobot v3.

## Design contract

Every parameter is classified and **no UNKNOWN parameter is silently invented**:

| Tier | Meaning | Behaviour in sim |
|------|---------|------------------|
| **KNOWN** | datasheet-authoritative or frozen in a Sentrix doc | used directly |
| **ESTIMATED** | derived/standard-physics/documented engineering choice | used, flagged, confidence < 1 |
| **UNKNOWN** | not in any source | `value: null`, `enabled: false` → reading it **raises** unless `--allow-placeholders`, which stamps every output `physics_fidelity: placeholder` |

Confidence scale: **C5** 0.95–1.0 datasheet/frozen · **C4** 0.80 derived/standard
· **C3** 0.60 estimate · **C2** 0.40 weak · **C1** ≤0.30 must-measure.

Default run mode is **relative / shape-only**: timing, noise, quantization,
saturation, topology and signal *shape* are trustworthy; absolute force/field
*magnitudes* are gated behind UNKNOWN parameters (moduli, friction, remanence).

## Architecture (layers)

```
L0 ground-truth interaction   events/generator.py
L1 contact mechanics          layers/l1_contact.py
L2 magnetic field (dipole)    layers/l2_field.py
L3 BMM350 model               layers/l3_bmm350.py
L4 LIS2DTW12 model            layers/l4_lis2dtw12.py
L5 noise & drift              layers/l5_noise_drift.py
L6 synchronization            layers/l6_sync.py
L7 export (parquet/mcap/lerobot)  layers/l7_export/
```

Sensor topology: **Layout B** — 21× BMM350 (4/4/4/3/2 fingertips + 4 palm) +
3× LIS2DTW12 (thumb/index/middle). See `configs/topology_layoutB.yaml`.

## Repository

```
configs/          parameters.yaml (registry) · topology · scene_default · scene_hard · events/
src/sentrixsim/   layers/ · params/ · events/ · dataset.py · hardmode.py · cli.py
benchmarks/       run_benchmark.py · data.py
docs/             ASSUMPTIONS.md · HARDMODE_failure_modes.md
tests/
```

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Use

```bash
sentrixsim list-events
sentrixsim show-params --tier UNKNOWN
sentrixsim simulate --event slip --out ./out --formats parquet,mcap,lerobot --seed 0
sentrixsim simulate-all --out ./out

# balanced datasets (900 episodes; Parquet + MCAP + LeRobot)
sentrixsim build-dataset --out ./dataset_v0.1
sentrixsim build-dataset --out ./dataset_v0.2 --version 0.2 --hard-mode

# baseline ML benchmarks (XGBoost / RandomForest / 1D CNN)
python benchmarks/run_benchmark.py --parquet ./dataset_v0.2/parquet --out ./benchmarks_v0.2
```

Gestures: `idle tap press hold shear slip release pinch grasp`.

## Datasheet-authoritative values (KNOWN)

- **BMM350**: ±2000 µT range; 190 nT(xy)/450 nT(z) RMS noise; 0.1 µT resolution
  (24-bit raw); ≤400 Hz ODR. [Bosch BMM350 datasheet]
- **LIS2DTW12**: ±2/4/8/16 g; 0.244 mg/LSB @±2g 14-bit; 90 µg/√Hz (HP);
  1.3 mg LP floor; 1.6–1600 Hz; 32-FIFO; temp 12-bit, 0.8 °C, ≤50 Hz.
  [ST LIS2DTW12/LIS2DW12 datasheet]

To-confirm sub-values (BMM TCO/TCS/cross-axis/AVG table; LIS TCoff, temp LSB/°C)
ship **OFF** (zero effect) until read from the exact datasheet tables.

## Outputs

- **Raw streams**: `tactile.B_raw[21,3]` µT (un-baselined absolute field),
  `dyn.accel[3,3]` g, `dyn.temp[3]` °C, µs timestamps, validity masks.
- **Labels**: ground-truth (`source: ground_truth`) **and** decoded estimates
  (`source: simulated_estimate`) side by side — the dataset is self-validating.
- **Exports**: Parquet (medallion table), MCAP (self-describing log), LeRobot v3
  (native writer, no torch dep).

## Datasets

Both are 900 episodes (9 gestures × 100), balanced, deterministic seeds,
~0.7–0.8 GB, all three export formats, validated (no missing channels, no NaNs,
sync + export integrity). Reproducible via `build-dataset`.

- **v0.1** (`dataset_v0.1/`) — clean baseline; relative/shape-only physics.
- **v0.2 Hard Mode** (`dataset_v0.2/`) — realistic ambiguity: multi-stage
  episodes, gesture overlap, micro-slip, partial contacts, sensor dropouts,
  timestamp jitter, cross-finger coupling, variable styles, per-session
  calibration drift, hard negatives. Knobs in `configs/scene_hard.yaml`; failure
  modes in `docs/HARDMODE_failure_modes.md`.

Each run emits `manifest.json`, `stats.json`, `validation.json`, and a report.

## Benchmarks (baseline)

Three tasks × {XGBoost, RandomForest, 1D CNN}, episode-level split. Best F1 (macro):

| Task | v0.1 | v0.2 Hard | target band |
|------|------|-----------|-------------|
| Event (9-class) | 1.00 | 0.90 | 0.80–0.95 |
| Contact | 1.00 | 0.99 | 0.90–0.99 |
| Slip | 1.00 | 0.86 (CNN) | 0.75–0.95 |

v0.1 was near-ceiling (too clean); v0.2 lands in the realistic bands with
interpretable confusions. Full tables + confusion matrices:
`benchmarks_v0.2/benchmark_report.md`.

## Limitations & synthetic-to-real

- **Relative / shape-only**: absolute force (N) and field (µT) magnitudes are not
  physical (gated behind UNKNOWN moduli / friction / remanence). Only signal
  shape, timing, noise, quantization and topology are spec-true — benchmark
  scores measure separability of *simulated* structure, **not** real-hardware
  performance.
- No vision / RGB-D stream (the glove carries no camera; genlock is architectural).
- Models trained on this data must be re-fit once calibrated hardware data
  replaces the relative scales (path in `docs/ASSUMPTIONS.md`).

## Upgrading to a calibrated twin

See `docs/ASSUMPTIONS.md` for the per-module assumption → measurement map.
Workflow: run the bench measurement → edit the YAML entry (`value`, `tier:
KNOWN`, `enabled: true`, raise `confidence`) → re-run. No code change.

## Provenance

Derived from the Sentrix documents (build spec, BOM, architecture derivation,
data-engine manual, CTO review) and the BMM350 / LIS2DTW12 datasheets. Every
parameter carries its origin in `configs/parameters.yaml`.
