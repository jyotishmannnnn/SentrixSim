"""LeRobot v3 native writer (no lerobot/torch dependency).

Writes a minimal but self-consistent LeRobot v3-style layout:

    <out>/lerobot/<name>/
        meta/info.json          # features, fps, layout flag, provenance
        meta/episodes.jsonl     # one record per episode
        data/chunk-000/file-000.parquet   # multiple-episodes-per-file (v3 shard)

Frames are indexed on the master grid (fps = master_rate_hz) so the high-rate
tactile signal is never decimated (ENGINE 5.0 fidelity rule). Tactile is stored
as a fixed-shape per-frame vector observation.tactile = [nb*3] (Bx,By,Bz per
BMM350); the engine's [R,U,V] tensor is produced via schema.project_ruv when
needed, not implied here. No video is emitted (the glove streams carry no
camera; vision genlock is architectural and out of scope for v1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ...core_types import Episode


def _features(nb: int) -> dict:
    return {
        "observation.tactile": {"dtype": "float32", "shape": [nb * 3],
                                 "names": None, "note": "Bx,By,Bz per BMM350 (uT)"},
        "observation.dynamics": {"dtype": "float32", "shape": [9],
                                  "names": None, "note": "ax,ay,az x thumb,index,middle (g)"},
        "observation.temp": {"dtype": "float32", "shape": [3], "names": ["thumb", "index", "middle"]},
        "action": {"dtype": "float32", "shape": [1],
                   "note": "no action - passive human-demo capture; placeholder zeros"},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "phase": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }


def write(ep: Episode, out_dir: str | Path) -> Path:
    root = Path(out_dir) / "lerobot" / ep.name
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    n = ep.n_samples
    nb = ep.aligned["B_read_uT"].shape[1]
    B = ep.aligned["B_read_uT"].reshape(n, nb * 3).astype(np.float32)
    A = ep.aligned["accel_read_g"].reshape(n, 9).astype(np.float32)
    Temp = ep.aligned["temp_read_c"].astype(np.float32)
    t_s = (ep.t_master_us.astype(np.float64) * 1e-6).astype(np.float32)

    done = np.zeros(n, bool)
    done[-1] = True

    cols = {
        "observation.tactile": pa.array(list(B)),
        "observation.dynamics": pa.array(list(A)),
        "observation.temp": pa.array(list(Temp)),
        "action": pa.array([[0.0]] * n),
        "timestamp": pa.array(t_s),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(n, np.int64)),
        "index": pa.array(np.arange(n, dtype=np.int64)),
        "task_index": pa.array(np.zeros(n, np.int64)),
        "phase": pa.array(ep.aligned["phase_id"].astype(np.int64)),
        "next.reward": pa.array(np.zeros(n, np.float32)),
        "next.done": pa.array(done),
    }
    table = pa.table(cols)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet",
                   compression="zstd")

    fps = ep.meta["master_rate_hz"]
    info = {
        "codebase_version": "v3.0-sentrixsim-native",
        "robot_type": "sentrix_mark2_glove",
        "total_episodes": 1,
        "total_frames": n,
        "fps": fps,
        "chunks_size": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": _features(nb),
        "tasks": [ep.meta.get("event")],
        "sentrixsim_meta": ep.meta,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    epis = {
        "episode_index": 0,
        "tasks": [ep.meta.get("event")],
        "length": n,
        "physics_fidelity": ep.meta.get("physics_fidelity"),
        "sync_quality": ep.meta.get("sync_quality"),
    }
    (root / "meta" / "episodes.jsonl").write_text(json.dumps(epis) + "\n", encoding="utf-8")
    return root
