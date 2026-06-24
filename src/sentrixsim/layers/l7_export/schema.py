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

UNITS = {
    "t_master_us": "us",
    "B": "uT",
    "accel": "g",
    "temp": "degC",
}


# ---- canonical, topology-driven, sensor_id-keyed columns (Migration Phase 1) ----
# Convention mirrors sentrix_contracts.columns: mag.<sensor_id>.{bx,by,bz}_uT,
# dyn.<sensor_id>.{ax,ay,az}_g, dyn.<sensor_id>.temp_c. NO ordinal / finger names.
def tactile_columns(bmm_ids: list[str]) -> list[str]:
    return [f"mag.{sid}.{ax}_uT" for sid in bmm_ids for ax in TACTILE_AXES]


def accel_columns(lis_ids: list[str]) -> list[str]:
    return [f"dyn.{sid}.{ax}_g" for sid in lis_ids for ax in ACCEL_AXES]


def temp_columns(lis_ids: list[str]) -> list[str]:
    return [f"dyn.{sid}.temp_c" for sid in lis_ids]


# NOTE: the legacy Layout-B column shim (flat tactile.bNN / dyn.<finger>) was
# retired in SIM-3. SentrixSim emits ONLY the canonical sensor_id-keyed columns
# above. DataEngine still reads pre-migration legacy artifacts via its own
# resolver fallback; SentrixSim no longer produces them.
def project_ruv(B_read_uT: np.ndarray, n_bmm: int) -> np.ndarray:
    """Explicit, documented projection of sparse clusters to a [T, 1, n_bmm]
    pseudo-image (U=1, V=n_bmm). This is a labelling convenience for the Data
    Engine's [R,U,V] contract, NOT a physical taxel grid. R (sub-frame burst) is
    a separate concern handled by the engine's resampling rule."""
    T = B_read_uT.shape[0]
    mag = np.linalg.norm(B_read_uT, axis=-1)  # (T, n_bmm)
    return mag.reshape(T, 1, n_bmm)
