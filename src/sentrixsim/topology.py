"""Sensor topology - Layout B.

Builds the 21x BMM350 + 3x LIS2DTW12 site list with per-sensor positions in the
hand frame, and within-cluster offsets synthesized from the cluster pitch and the
named arrangement.

Assumptions
-----------
* Cluster centres are nominal placeholders (geo.sensor_coords is UNKNOWN).
* Within-cluster arrangements:
    quad     -> plus pattern (N/E/S/W) at +/-pitch/2
    triangle -> equilateral, circumradius = pitch/sqrt(3)
    pair     -> opposed along x at +/-pitch/2
    coarse_grid (palm) -> 4 corners of a square of side = 3*pitch
* Each sensor's package frame is aligned with the hand frame (no per-sensor
  mounting rotation modelled in v1).

Limitations
-----------
* Real per-sensor coordinates and mounting rotations are unknown.

Hardware-upgrade path
---------------------
* Replace cluster centres + offsets with CT / optical metrology of a first
  article; populate geo.sensor_coords and set its tier to KNOWN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .params import ParameterRegistry


@dataclass
class SensorSite:
    sid: int
    kind: str          # "bmm350" | "lis2dtw12"
    finger: str
    role: str          # cluster arrangement label
    position_mm: np.ndarray  # (3,) hand frame
    sensor_id: str     # e.g. "bmm_index_2" / "lis_index"


@dataclass
class Topology:
    name: str
    sites: list[SensorSite] = field(default_factory=list)

    @property
    def bmm_sites(self) -> list[SensorSite]:
        return [s for s in self.sites if s.kind == "bmm350"]

    @property
    def lis_sites(self) -> list[SensorSite]:
        return [s for s in self.sites if s.kind == "lis2dtw12"]

    @property
    def n_bmm(self) -> int:
        return len(self.bmm_sites)

    @property
    def n_lis(self) -> int:
        return len(self.lis_sites)


def _cluster_offsets(arrangement: str, pitch: float) -> np.ndarray:
    """Return (n,3) in-plane (x,y) offsets for a cluster arrangement (z=0)."""
    h = pitch / 2.0
    if arrangement == "quad":
        return np.array([[0, h, 0], [h, 0, 0], [0, -h, 0], [-h, 0, 0]], float)
    if arrangement == "triangle":
        r = pitch / np.sqrt(3.0)
        ang = np.deg2rad([90.0, 210.0, 330.0])
        return np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(3)])
    if arrangement == "pair":
        return np.array([[h, 0, 0], [-h, 0, 0]], float)
    if arrangement == "coarse_grid":
        s = 1.5 * pitch
        return np.array([[-s, s, 0], [s, s, 0], [-s, -s, 0], [s, -s, 0]], float)
    raise ValueError(f"Unknown arrangement: {arrangement}")


def build_topology(topo_path: str | Path, reg: ParameterRegistry) -> Topology:
    spec = yaml.safe_load(Path(topo_path).read_text(encoding="utf-8"))
    pitch = float(reg.get("geo.cluster_pitch_mm"))
    topo = Topology(name=spec["name"])
    sid = 0
    for cl in spec["clusters"]:
        center = np.asarray(cl["center_mm"], float)
        offs = _cluster_offsets(cl["arrangement"], pitch)
        n = int(cl["n_bmm350"])
        for k in range(n):
            topo.sites.append(
                SensorSite(
                    sid=sid,
                    kind="bmm350",
                    finger=cl["finger"],
                    role=cl["arrangement"],
                    position_mm=center + offs[k % len(offs)],
                    sensor_id=f"bmm_{cl['finger']}_{k}",
                )
            )
            sid += 1
        if cl.get("dynamics"):
            topo.sites.append(
                SensorSite(
                    sid=sid,
                    kind="lis2dtw12",
                    finger=cl["finger"],
                    role="dynamics",
                    position_mm=center.copy(),
                    sensor_id=f"lis_{cl['finger']}",
                )
            )
            sid += 1
    return topo
