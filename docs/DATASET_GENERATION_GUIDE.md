# Dataset Generation Guide

Practical, first-time-developer guide to generating Sentrix datasets with the
current SentrixSim implementation. For background on the parameter framework and
limitations see the [README](../README.md); for the Hard Mode effects see
[`HARDMODE_failure_modes.md`](HARDMODE_failure_modes.md).

## 1. Prerequisites & setup

- Python **3.11+**, git, ~1 GB free disk per dataset.
- Install the package (pulls in numpy, scipy, pyarrow, pydantic, pyyaml, mcap):

```bash
python -m venv .venv
# activate:
.venv\Scripts\activate        # Windows (PowerShell/cmd)
source .venv/bin/activate      # macOS / Linux
pip install -e .
```

Verify the CLI is available:

```bash
sentrixsim list-events
# -> idle tap press hold shear slip release pinch grasp
```

> The `sentrixsim` command is installed by `pip install -e .`. If it is not on
> your PATH, use `python -m sentrixsim.cli ...` instead — identical arguments.

## 2. Quick start

```bash
# clean baseline dataset (Parquet + MCAP + LeRobot)
sentrixsim build-dataset --out ./dataset_v0.1

# Hard Mode dataset (realistic ambiguity)
sentrixsim build-dataset --out ./dataset_v0.2 --version 0.2 --hard-mode
```

Each command generates **900 episodes** (9 gestures × 100), writes all three
export formats, validates them, and prints a JSON summary:

```json
{ "out": "dataset_v0.1", "total_episodes": 900, "storage_gb": 0.706,
  "elapsed_s": 364.4, "validation_all_passed": true }
```

Exit code is **0** only if `validation_all_passed` is true (else **2**).

## 3. Dataset v0.1 (clean baseline)

```bash
sentrixsim build-dataset --out ./dataset_v0.1
```

Relative/shape-only physics, no Hard Mode augmentation. ~0.7 GB, a few minutes.

## 4. Dataset v0.2 Hard Mode

```bash
sentrixsim build-dataset --out ./dataset_v0.2 --version 0.2 --hard-mode
```

Adds the 10 realism effects (multi-stage episodes, gesture overlap, micro-slip,
partial contacts, sensor dropouts, timestamp jitter, cross-finger coupling,
variable styles, per-session calibration drift, hard negatives). The knobs live
in [`configs/scene_hard.yaml`](../configs/scene_hard.yaml); episodes are stamped
`physics_fidelity: relative+hardmode`.

## 5. Key command-line options (`build-dataset`)

| Option | Default | Meaning |
|--------|---------|---------|
| `--out` | *(required)* | Output directory for the dataset. |
| `--version` | `0.1` | Tag used in the report filename and metadata. |
| `--hard-mode` | off | Enable v0.2 Hard Mode augmentation. |
| `--n-noise` | `5` | Noise (sensor-noise RNG) realizations per duration cell. |
| `--n-drift` | `4` | Drift (per-episode offset) realizations per cell. |
| `--master-seed` | `20260601` | Root seed; all per-episode seeds derive from it (full reproducibility). |
| `--formats` | `parquet,mcap,lerobot` | Comma-separated subset of export formats. |
| `--mcap-stride` | `1` | Write MCAP every Nth episode (use `>1` to cut MCAP size/time). |

Episodes per event = `5 durations × --n-noise × --n-drift`. Defaults give
`5 × 5 × 4 = 100` per event (900 total). Example smaller/faster run:

```bash
sentrixsim build-dataset --out ./mini --n-noise 2 --n-drift 1 --formats parquet
# 5 x 2 x 1 = 10 episodes/event = 90 total, Parquet only
```

> `--config-dir <path>` is a **global** flag placed *before* the subcommand to
> point at a different `configs/` directory:
> `sentrixsim --config-dir ./configs build-dataset --out ./d`.

## 6. Expected outputs

```
dataset_v0.2/
├── parquet/                 # one file per episode, grouped by event
│   ├── idle/idle__d0_n0_r0.parquet
│   ├── tap/ ...
│   └── grasp/ ...
├── mcap/                    # one self-describing log per episode (unless --mcap-stride>1)
│   └── <event>/<episode_id>.mcap
├── lerobot/                 # ONE consolidated LeRobot v3 dataset
│   ├── meta/info.json
│   ├── meta/episodes.jsonl
│   └── data/chunk-000/file-000.parquet  ...  chunk-0NN/
├── manifest.json            # per-episode: id, event, duration, seeds, session, flags
├── stats.json               # counts, channel/timestamp/range stats, label distribution, storage
├── validation.json          # integrity checks (see §7)
└── dataset_v0.2_report.md    # human-readable report (filename uses --version)
```

Per-episode Parquet columns are **sensor_id-keyed** (canonical), driven by the
topology descriptor: `t_master_us`, `mag.<sensor_id>.{bx,by,bz}_uT` (Mark2_v1 →
21 sensors × 3 = 63 columns, e.g. `mag.bmm_index_2.bx_uT`),
`dyn.<sensor_id>.{ax,ay,az}_g` (3 × 3 = 9, e.g. `dyn.lis_index.ax_g`),
`dyn.<sensor_id>.temp_c` (3), `bmm_valid`, `temp_valid`, `sat_any`, `phase_id`,
plus ground-truth `label.*` and decoded `est.*` columns (Hard Mode also adds
`dropout_any`). With `--legacy-columns` the older Layout-B names are emitted
instead (`tactile.bXX.{bx,by,bz}_uT`, `dyn.<finger>.{ax,ay,az}_g`,
`dyn.<finger>.temp_degC`) as a back-compat shim. Parameter provenance is
embedded in the Parquet schema metadata.

## 7. Verifying a successful run

1. **Exit code / summary** — `validation_all_passed: true` in the printed JSON.
2. **`validation.json`** — all of these should be empty / true:

```bash
python -c "import json;v=json.load(open('dataset_v0.2/validation.json'));print(v['all_passed'], {k:len(v[k]) for k in ('missing_channels','nan_episodes','sync_failures','export_failures')})"
# -> True {'missing_channels': 0, 'nan_episodes': 0, 'sync_failures': 0, 'export_failures': 0}
```

3. **Counts & LeRobot integrity:**

```bash
python -c "import json;s=json.load(open('dataset_v0.2/stats.json'));print(s['total_episodes'], s['episode_counts'])"
python -c "import json;i=json.load(open('dataset_v0.2/lerobot/meta/info.json'));print(i['total_episodes'],i['total_frames'])"
```

4. **Spot-check a Parquet file reads back:**

```bash
python -c "import pyarrow.parquet as pq;t=pq.read_table('dataset_v0.2/parquet/grasp/grasp__d0_n0_r0.parquet');print(t.num_rows, len(t.column_names))"
```

5. Read `dataset_v0.2_report.md` for class balance, channel/timestamp/range
   stats, and label distributions.

## 8. Directory structure before generation

```
SentrixSim/
├── configs/        parameters.yaml, topology_layoutB.yaml, scene_default.yaml,
│                   scene_hard.yaml, events/*.yaml
├── src/sentrixsim/ layers/, params/, events/, dataset.py, hardmode.py, cli.py
├── benchmarks/     run_benchmark.py, data.py
├── docs/           ASSUMPTIONS.md, HARDMODE_failure_modes.md, this guide
├── tests/
├── pyproject.toml
└── README.md
```

(After generation, the chosen `--out` directory appears as shown in §6.)

## 9. (Optional) Run baseline benchmarks on a dataset

```bash
pip install scikit-learn xgboost torch        # extra ML deps (CPU torch is fine)
python benchmarks/run_benchmark.py --parquet ./dataset_v0.2/parquet --out ./benchmarks_v0.2 --version 0.2
# writes benchmarks_v0.2/benchmark_report.md and benchmark_results.json
```

Add `--compare ./benchmarks/benchmark_results_v0.1.json` to include a vs-v0.1
delta table.

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `sentrixsim: command not found` | Activate the venv; run `pip install -e .`; or use `python -m sentrixsim.cli ...`. |
| `ModuleNotFoundError: No module named 'sentrixsim'` (or numpy/pyarrow/mcap) | Re-run `pip install -e .` inside the activated venv. |
| Generation is slow / MCAP too large | Use `--mcap-stride 10`, or `--formats parquet,lerobot` to skip MCAP. |
| Dataset too big for disk | Lower `--n-noise` / `--n-drift` (fewer episodes), or drop formats. |
| `validation_all_passed: false` (exit 2) | Open `validation.json`; the non-empty list (`missing_channels` / `nan_episodes` / `sync_failures` / `export_failures`) names the failing episodes. |
| Want a quick smoke run | `--n-noise 1 --n-drift 1 --formats parquet` (90 episodes, seconds). |
| Need byte-for-byte reproducibility | Keep `--master-seed` fixed; every per-episode seed is derived from it and recorded in `manifest.json`. |

## 11. Notes & caveats

Datasets are **relative/shape-only** physics: signal shape, timing, noise,
quantization and topology are spec-true, but absolute force/field magnitudes are
not (see the README "Limitations & synthetic-to-real" section). Use for pipeline,
storage, labeling and ML-prototyping work; re-fit any models once calibrated
hardware data is available.
