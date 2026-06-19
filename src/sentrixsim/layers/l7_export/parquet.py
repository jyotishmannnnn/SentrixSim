"""Parquet exporter (PyArrow).

Writes the aligned master-grid table: timestamp + flattened tactile field +
tripod acceleration + temperature + masks + phase + all labels/estimates.
Metadata (incl. full parameter provenance) is attached to the Arrow schema.

Layout follows the medallion intent (ENGINE 2.2): big object, few files, tactile
kept as fixed-shape columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ...core_types import Episode
from .schema import (TACTILE_AXES, accel_columns, flat_accel_columns,
                     flat_tactile_columns, tactile_columns, temp_columns)


def write(ep: Episode, out_dir: str | Path, legacy_columns: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = ep.n_samples
    cols: dict[str, np.ndarray] = {"t_master_us": ep.t_master_us}

    B = ep.aligned["B_read_uT"]                 # (T, nb, 3)
    A = ep.aligned["accel_read_g"]              # (T, nl, 3)
    temp = ep.aligned["temp_read_c"]            # (T, nl)
    nb, nl = B.shape[1], A.shape[1]

    # Sensor ids come from the topology descriptor (carried in meta). Count-agnostic.
    bmm_ids = ep.meta.get("bmm_sensor_ids")
    lis_ids = ep.meta.get("lis_sensor_ids")

    if legacy_columns or bmm_ids is None or lis_ids is None:
        # Layout-B compatibility shim (ordinal tactile.bNN + dyn.<finger>).
        fingers = [s.replace("lis_", "") for s in lis_ids] if lis_ids else None
        for i, name in enumerate(flat_tactile_columns(nb)):
            cols[name] = B[:, i // 3, i % 3]
        for j, name in enumerate(flat_accel_columns(fingers)):
            cols[name] = A[:, j // 3, j % 3]
        for k, f in enumerate(fingers or ["thumb", "index", "middle"]):
            cols[f"dyn.{f}.temp_degC"] = temp[:, k]
    else:
        # Canonical sensor_id-keyed columns.
        for i, name in enumerate(tactile_columns(bmm_ids)):
            cols[name] = B[:, i // 3, i % 3]
        for j, name in enumerate(accel_columns(lis_ids)):
            cols[name] = A[:, j // 3, j % 3]
        for k, name in enumerate(temp_columns(lis_ids)):
            cols[name] = temp[:, k]

    cols["bmm_valid"] = ep.aligned["bmm_valid"]
    cols["temp_valid"] = ep.aligned["temp_valid"]
    cols["sat_any"] = ep.aligned["sat_flag"].any(axis=(1, 2))
    if "dropout" in ep.aligned:
        cols["dropout_any"] = ep.aligned["dropout"].any(axis=1)
    cols["phase_id"] = ep.aligned["phase_id"]

    for name, arr in ep.labels.items():
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == n:
            cols[name] = a

    arrays, names = [], []
    for k, v in cols.items():
        names.append(k)
        arrays.append(pa.array(np.asarray(v)))
    table = pa.table(arrays, names=names)

    meta = {
        b"sentrixsim_meta": json.dumps(ep.meta).encode(),
        b"sentrixsim_label_meta": json.dumps(ep.label_meta).encode(),
        b"sentrixsim_provenance": json.dumps(ep.provenance).encode(),
    }
    table = table.replace_schema_metadata(meta)

    path = out_dir / f"{ep.name}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path
