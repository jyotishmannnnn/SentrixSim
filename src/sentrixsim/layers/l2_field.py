"""Layer 2 - Magnetic field model (dipole superposition, 1/r^3).

Models each fingertip's magnetized pad as a single magnetic dipole above the
sensor cluster. Pressing moves the dipole toward the sensors (z decreases);
shear moves it laterally. The field at each BMM350 site is the dipole field; the
perturbation Delta-B is the field relative to the rest state.

Inputs:  topology (BMM site positions), Deformation, params.
Outputs: B_true[T, 21, 3] in microtesla (ambient B0 + Delta-B).

Equations
---------
g(r, m_hat) = [3 (m_hat . r_hat) r_hat - m_hat] / |r|^3          (dipole shape)
Delta-B(site) = scale * ( g(site - p_now, m_hat) - g(site - p_rest, m_hat) )
B_true = B0 + Delta-B

`scale` is fixed so that a unit normalized normal press produces a peak
|Delta-B| equal to mag.field_scale_uT at the nearest sensor. This makes the
stream dimensionally usable (uT) for the BMM350 model while remaining a
relative presentation scale, NOT a physical magnitude.

Assumptions
-----------
* Single dipole per cluster, isotropic remanence, aligned through-thickness.
* 1/r^3 decay (BUILD Correction 1). Inter-finger magnetic crosstalk not modelled.

Limitations
-----------
* Absolute |Delta-B| is unknown (proportional to mag.Br_mT, UNKNOWN). The shape
  is physical; the magnitude is a presentation scale (mag.field_scale_uT).

Hardware-upgrade path
---------------------
* Replace the dipole with a measured magnetization map (gaussmeter/Helmholtz
  scan of a jig-magnetized cartridge); set mag.Br_mT and remove field_scale_uT.
"""
from __future__ import annotations

import numpy as np

from ..core_types import Deformation
from ..params import ParameterRegistry
from ..topology import Topology


def _dipole_shape(r: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    """r: (...,3) vector from dipole to field point (mm). Returns (...,3)."""
    dist = np.linalg.norm(r, axis=-1, keepdims=True)
    dist = np.clip(dist, 1e-6, None)
    r_hat = r / dist
    proj = np.sum(r_hat * m_hat, axis=-1, keepdims=True)
    return (3.0 * proj * r_hat - m_hat) / dist**3


class FieldModel:
    def __init__(self, topo: Topology, reg: ParameterRegistry, scene: dict):
        self.topo = topo
        self.B0 = np.asarray(reg.get("env.B0_uT"), float)
        self.field_scale = float(reg.get("mag.field_scale_uT"))
        self.standoff = float(reg.get("geo.standoff_mm"))
        self.layer_t = float(reg.get("mag.layer_thickness_mm"))
        self.m_hat = np.asarray(scene.get("l2", {}).get("dipole_moment_axis", [0, 0, -1]), float)
        self.m_hat = self.m_hat / np.linalg.norm(self.m_hat)

        # Gate the UNKNOWN remanence (honours --allow-placeholders).
        if reg.allow_placeholders:
            _ = reg.get("mag.Br_mT")

        # Per-cluster local sensor offsets and the calibration scale.
        self._clusters: dict[str, np.ndarray] = {}
        for s in topo.bmm_sites:
            self._clusters.setdefault(s.finger, [])
            self._clusters[s.finger].append(s.position_mm)
        self._clusters = {f: np.asarray(v) - np.asarray(v).mean(0) for f, v in self._clusters.items()}

        # Rest dipole height above the cluster plane (mm), toward sensors.
        self.z_rest = self.standoff + 0.5 * self.layer_t
        self._scale = self._calibrate_scale()

    def _cluster_field(self, offsets: np.ndarray, dz: float, dx: float, dy: float) -> np.ndarray:
        p_rest = np.array([0.0, 0.0, self.z_rest])
        p_now = np.array([dx, dy, self.z_rest - dz])
        r_now = offsets - p_now
        r_rest = offsets - p_rest
        return _dipole_shape(r_now, self.m_hat) - _dipole_shape(r_rest, self.m_hat)

    def _cluster_field_t(self, offsets: np.ndarray, dz, dx, dy) -> np.ndarray:
        """Vectorized over time. offsets:(k,3); dz,dx,dy:(T,). Returns (T,k,3)."""
        T = dz.shape[0]
        p_now = np.stack([dx, dy, self.z_rest - dz], axis=-1)        # (T,3)
        p_rest = np.array([0.0, 0.0, self.z_rest])
        r_now = offsets[None, :, :] - p_now[:, None, :]              # (T,k,3)
        r_rest = np.broadcast_to(offsets - p_rest, (T,) + offsets.shape)
        return _dipole_shape(r_now, self.m_hat) - _dipole_shape(r_rest, self.m_hat)

    def _calibrate_scale(self) -> float:
        # Unit normal press, max compression from l1 default (0.30 mm proxy).
        ref_dz = 0.30
        peak = 0.0
        for offs in self._clusters.values():
            db = self._cluster_field(offs, ref_dz, 0.0, 0.0)
            peak = max(peak, float(np.linalg.norm(db, axis=-1).max()))
        return self.field_scale / max(peak, 1e-12)

    def run(self, deform: Deformation, n: int) -> np.ndarray:
        """Return B_true[T, n_bmm, 3] in uT."""
        nb = self.topo.n_bmm
        B = np.tile(self.B0, (n, nb, 1)).astype(float)
        # Map global bmm index -> (finger, local index)
        finger_local: dict[str, list[int]] = {}
        for i, s in enumerate(self.topo.bmm_sites):
            finger_local.setdefault(s.finger, []).append(i)
        for f, idxs in finger_local.items():
            offs = self._clusters[f]
            if f not in deform.dz_mm:
                continue  # no contact on this finger -> only B0
            dz, dx, dy = deform.dz_mm[f], deform.dx_mm[f], deform.dy_mm[f]
            db = self._scale * self._cluster_field_t(offs, dz, dx, dy)   # (T,k,3)
            for k, gi in enumerate(idxs):
                B[:, gi, :] += db[:, k, :]
        return B
