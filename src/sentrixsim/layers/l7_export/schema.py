"""Channel schema shared by all exporters.

Native representation = SPARSE MAGNETIC CLUSTERS (21 BMM350 x 3 axes), matching
the hardware - NOT a dense [R,U,V] taxel image. The [R,U,V] tensor the Data
Engine expects (ENGINE 5.1) is produced only by an explicit projection
(project_ruv), never implied. This resolves the cluster-vs-image mismatch and
the ragged-R issue flagged in the CTO review (CTO 6).
"""
from __future__ import annotations

import numpy as np

# Flat per-sample column names for the aligned (master-grid) table.
# B_xx_<axis> per BMM index is expanded at write time from B_read_uT[T,21,3].
TACTILE_AXES = ["bx", "by", "bz"]
ACCEL_AXES = ["ax", "ay", "az"]
TRIPOD = ["thumb", "index", "middle"]

UNITS = {
    "t_master_us": "us",
    "B": "uT",
    "accel": "g",
    "temp": "degC",
}


def flat_tactile_columns(n_bmm: int) -> list[str]:
    cols = []
    for i in range(n_bmm):
        for ax in TACTILE_AXES:
            cols.append(f"tactile.b{i:02d}.{ax}_uT")
    return cols


def flat_accel_columns() -> list[str]:
    cols = []
    for f in TRIPOD:
        for ax in ACCEL_AXES:
            cols.append(f"dyn.{f}.{ax}_g")
    return cols


def project_ruv(B_read_uT: np.ndarray, n_bmm: int) -> np.ndarray:
    """Explicit, documented projection of sparse clusters to a [T, 1, n_bmm]
    pseudo-image (U=1, V=n_bmm). This is a labelling convenience for the Data
    Engine's [R,U,V] contract, NOT a physical taxel grid. R (sub-frame burst) is
    a separate concern handled by the engine's resampling rule."""
    T = B_read_uT.shape[0]
    mag = np.linalg.norm(B_read_uT, axis=-1)  # (T, n_bmm)
    return mag.reshape(T, 1, n_bmm)
