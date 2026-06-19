# SentrixSim — Repository Memory

> Forward simulator for the Sentrix Mark 2 visuotactile glove. Producer of raw
> per-device episodes. See root `../CLAUDE.md` for ecosystem context.

## Role in the ecosystem

First stage: `SentrixSim → SentrixSync → SentrixDataEngine`. Simulates ONE device.
Exists so the rest of the stack can be built and validated before physical hardware.
A real glove/camera producer can later replace it behind SentrixSync's adapter
contract with no change downstream.

- Version: v0.1.0. Python 3.11+. Deps: numpy, scipy, pyarrow, pydantic, pyyaml, mcap.

## Architecture — L0→L7 forward pipeline

| Layer | Module | Responsibility | Output |
|---|---|---|---|
| L0 | `events/generator.py` | ground-truth force/kinematics from event YAML, 1600 Hz master grid | `GroundTruth` |
| L1 | `layers/l1_contact.py` | normalized wrench → magnet kinematics (lumped compliance) | `Deformation` |
| L2 | `layers/l2_field.py` | dipole 1/r³ magnetic field | `B_true[T,21,3]` µT |
| L3 | `layers/l3_bmm350.py` | BMM350 noise / saturation / quantization / dropout | `B_read_uT`, `B_lsb`, `sat_flag`, `dropout` |
| L4 | `layers/l4_lis2dtw12.py` | LIS2DTW12 accel + temperature | `accel_read_g`, `temp_read_c` |
| L5 | `layers/l5_noise_drift.py` | Gaussian noise + per-episode static drift | `NoiseModel` |
| L6 | `layers/l6_sync.py` | master-grid assembly, zero-order hold (latest-at) | `aligned` dict |
| L7 | `layers/l7_export/` | export to Parquet / MCAP / LeRobot | files |

Rates: master grid 1600 Hz (625 µs); BMM field 400 Hz; temp 50 Hz.

## Episode structure

```python
@dataclass
class Episode:
    name: str                          # "tap__d0_n0_r0"
    meta: dict[str, Any]               # seed, duration_s, physics_fidelity, units, rates, n_*
    t_master_us: np.ndarray            # (T,) int64 microseconds
    aligned: dict[str, np.ndarray]     # B_read_uT[T,21,3], accel_read_g[T,3,3], temp_read_c[T,3],
                                       #   bmm_valid[T], temp_valid[T], sat_flag, dropout, phase_id[T]
    labels: dict[str, np.ndarray]      # label.<finger>.* (ground truth) + est.<finger>.* (estimates)
    label_meta: dict[str, dict]        # {source, units, confidence∈[0,1], tier}
    provenance: list[dict]             # full parameter table
```

A **sample** = one master-grid row `t`: timestamp + B-field (21×3) + tripod accel
(3×3) + temp (3) + phase + labels.

## Exporters (`layers/l7_export/`)

- **Parquet** (`parquet.py`): flat per-episode table. Columns: `t_master_us`,
  63× `tactile.bNN.{bx,by,bz}_uT`, 9× `dyn.{thumb,index,middle}.{ax,ay,az}_g`,
  3× temp, validity masks, `sat_any`, `dropout_any`, `phase_id`, `label.*`, `est.*`.
  Schema KV metadata: `sentrixsim_meta`, `sentrixsim_label_meta`, `sentrixsim_provenance`. zstd.
- **MCAP** (`mcap.py`): 3 JSON channels — `tactile_field` (400 Hz, B_uT[[..]x21]),
  `dynamics_accel` (1600 Hz), `dynamics_temp` (50 Hz). Hub-µs ×1000 ns log_time.
- **LeRobot v3** (`lerobot.py` single; `lerobot_dataset.py` multi-episode buffered):
  `meta/info.json` + `meta/episodes.jsonl` + `data/chunk-NNN/file-NNN.parquet`.

## Design philosophy

- **Fidelity tiering.** Every parameter is KNOWN / ESTIMATED / UNKNOWN with a
  confidence score (`params/registry.py`). The sim refuses to silently invent
  unknowns; `meta["physics_fidelity"]` ∈ {`relative`, `placeholder`, `relative+hardmode`}.
- **Relative/shape-only by default.** Signal shapes/timing/noise chains are real;
  absolute magnitudes are presentation scales unless `--allow-placeholders`.
- **Deterministic.** Seeded; reproducible episodes and datasets.

## CLI

```
sentrixsim simulate --event tap --out ./out --formats parquet,mcap,lerobot
sentrixsim simulate-all --out ./out
sentrixsim build-dataset --out ./out [--n-noise 5 --n-drift 4 --hard-mode]
sentrixsim list-events
sentrixsim show-params [--tier KNOWN|ESTIMATED|UNKNOWN]
```

## Extension points

- New gesture → add `configs/events/<name>.yaml`.
- New exporter → add `layers/l7_export/<fmt>.py` with `write(ep, out_dir)`; wire into
  `cli.py::_export` and `dataset.py`.
- Parameters via `ParameterRegistry` (`configs/parameters.yaml`).

## Known limitations

Single-device only; no absolute physics; no vision stream; single unit instance
(cross-unit variation approximated by drift only); no 1/f drift PSD; TCO/TCS off by
default; LeRobot export does not materialize `[R,U,V]` taxel images.

## What must never be changed casually

- **Episode schema** (`core_types.py`): `name / meta / t_master_us / aligned / labels /
  label_meta / provenance`. SentrixSync's `SentrixSimAdapter` reads `t_master_us`.
- **Topology is descriptor-driven** (Migration Phase 1, done). `pipeline.simulate`
  loads a shared `sentrix_contracts` descriptor (default bundled `Mark2_v1`, override
  via `descriptor=`/`--descriptor`) and builds the `Topology` with
  `topology.from_descriptor`. Counts (`n_bmm`/`n_lis`) come from the descriptor; NO
  Layout-B constant (`21`/`3`) in the physics or export path. `meta` carries
  `descriptor_version`, `descriptor_hash`, `bmm_sensor_ids`, `lis_sensor_ids`.
  `Mark2_v1.json` is GENERATED from `build_topology` (see SentrixCapture
  `contracts/tools/gen_mark2_v1.py`) so it is geometrically byte-faithful — the
  topology-source swap is value-identical (`B_lsb` bit-identical, verified).
- **Canonical export columns are sensor_id-keyed**: `mag.<sensor_id>.{bx,by,bz}_uT`,
  `dyn.<sensor_id>.{ax,ay,az}_g`, `dyn.<sensor_id>.temp_c`. Legacy Layout-B names
  (`tactile.bNN`, `dyn.<finger>`) only via `--legacy-columns` / `legacy_columns=True`
  (a back-compat shim, to be retired in Phase 4). DataEngine's resolver must become
  descriptor-driven to read the new columns (Migration Phase 3).
- **KNOWN / ESTIMATED / UNKNOWN semantics + confidence scores** — load-bearing for
  honesty about physics fidelity; mirrored by Sync's `ParamTier`. Do not collapse or
  auto-fill tiers.
- **Exporter contracts / output layout** — LeRobot `info.json` + `episodes.jsonl` +
  chunked parquet convention; MCAP channel names/shapes. Downstream and the manual
  assume these.

## What must never be added

Multi-device synchronization or clock reconciliation; any import of SentrixSync or
SentrixDataEngine; cross-device timeline logic; dataset catalog / validation /
packaging; resolving another producer's payload refs.
