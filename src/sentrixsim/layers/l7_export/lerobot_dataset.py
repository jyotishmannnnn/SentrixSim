"""Consolidated multi-episode LeRobot v3 dataset writer.

Appends many episodes into one dataset with correct global indices and chunked
parquet shards (multiple episodes per file - the v3 small-file fix). Memory is
bounded: episodes are buffered per chunk and flushed to disk.

    <root>/
        meta/info.json
        meta/episodes.jsonl
        data/chunk-000/file-000.parquet
        data/chunk-001/file-000.parquet
        ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ...core_types import Episode
from .lerobot import _features


class LeRobotDatasetWriter:
    def __init__(self, root: str | Path, fps: float, chunk_episodes: int = 50,
                 tasks: list[str] | None = None):
        self.root = Path(root)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.chunk_episodes = chunk_episodes
        self._chunk: list[pa.Table] = []
        self._chunk_idx = 0
        self._ep_idx = 0
        self._global_index = 0
        self._total_frames = 0
        self._nb = None
        self._episodes_meta: list[dict] = []
        self._tasks: list[str] = tasks or []
        self._task_index: dict[str, int] = {}

    def _task_id(self, task: str) -> int:
        if task not in self._task_index:
            self._task_index[task] = len(self._task_index)
            if task not in self._tasks:
                self._tasks.append(task)
        return self._task_index[task]

    def append(self, ep: Episode) -> None:
        n = ep.n_samples
        nb = ep.aligned["B_read_uT"].shape[1]
        self._nb = nb
        task = ep.meta.get("event", "unknown")
        tid = self._task_id(task)

        B = ep.aligned["B_read_uT"].reshape(n, nb * 3).astype(np.float32)
        A = ep.aligned["accel_read_g"].reshape(n, 9).astype(np.float32)
        Temp = ep.aligned["temp_read_c"].astype(np.float32)
        t_s = (ep.t_master_us.astype(np.float64) * 1e-6).astype(np.float32)
        done = np.zeros(n, bool)
        done[-1] = True
        idx = np.arange(self._global_index, self._global_index + n, dtype=np.int64)

        table = pa.table({
            "observation.tactile": pa.array(list(B)),
            "observation.dynamics": pa.array(list(A)),
            "observation.temp": pa.array(list(Temp)),
            "action": pa.array([[0.0]] * n),
            "timestamp": pa.array(t_s),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, self._ep_idx, np.int64)),
            "index": pa.array(idx),
            "task_index": pa.array(np.full(n, tid, np.int64)),
            "phase": pa.array(ep.aligned["phase_id"].astype(np.int64)),
            "next.reward": pa.array(np.zeros(n, np.float32)),
            "next.done": pa.array(done),
        })
        self._chunk.append(table)
        self._episodes_meta.append({
            "episode_index": self._ep_idx,
            "tasks": [task],
            "length": n,
            "duration_s": ep.meta.get("duration_s"),
            "seed": ep.meta.get("seed"),
            "drift_seed": ep.meta.get("drift_seed"),
            "physics_fidelity": ep.meta.get("physics_fidelity"),
        })
        self._ep_idx += 1
        self._global_index += n
        self._total_frames += n
        if len(self._chunk) >= self.chunk_episodes:
            self._flush()

    def _flush(self) -> None:
        if not self._chunk:
            return
        d = self.root / "data" / f"chunk-{self._chunk_idx:03d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self._chunk), d / "file-000.parquet",
                       compression="zstd")
        self._chunk = []
        self._chunk_idx += 1

    def finalize(self, extra_meta: dict | None = None) -> Path:
        self._flush()
        info = {
            "codebase_version": "v3.0-sentrixsim-native",
            "robot_type": "sentrix_mark2_glove",
            "total_episodes": self._ep_idx,
            "total_frames": self._total_frames,
            "total_chunks": self._chunk_idx,
            "fps": self.fps,
            "chunks_size": self.chunk_episodes,
            "data_path": "data/chunk-{chunk_index:03d}/file-000.parquet",
            "features": _features(self._nb or 21),
            "tasks": self._tasks,
            "sentrixsim_meta": extra_meta or {},
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2),
                                                      encoding="utf-8")
        with open(self.root / "meta" / "episodes.jsonl", "w", encoding="utf-8") as fh:
            for e in self._episodes_meta:
                fh.write(json.dumps(e) + "\n")
        return self.root
