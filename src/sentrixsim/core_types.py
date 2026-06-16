"""Shared data structures passed between simulator layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Skill-grammar phase labels (ENGINE 3.4 / BUILD).
PHASES = ["idle", "reach", "grasp", "transport", "align", "insert", "release", "retreat"]
PHASE_ID = {name: i for i, name in enumerate(PHASES)}


@dataclass
class GroundTruth:
    """Layer 0 output, all on the master grid."""
    t_master_s: np.ndarray                       # (T,)
    fingers: list[str]                           # fingers with contact in this event
    normal: dict[str, np.ndarray]                # finger -> (T,) normalized [0,1]
    shear_x: dict[str, np.ndarray]               # finger -> (T,)
    shear_y: dict[str, np.ndarray]               # finger -> (T,)
    contact: dict[str, np.ndarray]               # finger -> (T,) bool
    slip: dict[str, np.ndarray]                  # finger -> (T,) bool
    slip_vel: dict[str, np.ndarray]              # finger -> (T,) normalized
    contact_loc: dict[str, np.ndarray]           # finger -> (T,2) on-pad (mm, relative)
    accel_true_g: dict[str, np.ndarray]          # lis finger -> (T,3) g
    temp_true_c: np.ndarray                      # (T,)
    phase_id: np.ndarray                         # (T,) int
    object_pos_mm: np.ndarray                    # (T,3) coarse object translation stub


@dataclass
class Deformation:
    """Layer 1 output: per-cluster magnet kinematics (relative units)."""
    dz_mm: dict[str, np.ndarray]                 # finger -> (T,) inward travel (>=0)
    dx_mm: dict[str, np.ndarray]                 # finger -> (T,) lateral x
    dy_mm: dict[str, np.ndarray]                 # finger -> (T,) lateral y


@dataclass
class Episode:
    name: str
    meta: dict[str, Any]
    t_master_us: np.ndarray                      # (T,) int64 microseconds
    aligned: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    label_meta: dict[str, dict] = field(default_factory=dict)
    provenance: list[dict] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.t_master_us.shape[0])
