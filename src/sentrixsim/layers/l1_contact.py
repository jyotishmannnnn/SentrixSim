"""Layer 1 - Contact mechanics approximation (relative / shape-only by default).

Maps the normalized contact wrench to magnet kinematics (inward travel dz and
lateral travel dx,dy of the magnetized pad) using a lumped-compliance model.

Inputs:  GroundTruth (normalized normal + shear per finger), scene shaping consts.
Outputs: Deformation (dz_mm, dx_mm, dy_mm per finger).

Equations
---------
Relative mode (default): dz = normal * normal_compression_mm
                         (dx,dy) = (shear_x, shear_y) * shear_travel_mm
This is a linear lumped-compliance map with the compliance folded into the two
shaping constants. The full hyperelastic continuum (sigma(u) from W, Coulomb
friction) is the absolute-mode target and requires UNKNOWN moduli/friction.

Assumptions
-----------
* Linear, decoupled normal/shear compliance (relative mode).
* Shaping constants (l1.*) set travel magnitudes only, not physical force->mm.

Limitations
-----------
* No true contact mechanics; magnitudes are normalized, not Newtons->mm.
* mech.E_body_kPa and mech.friction_mu are UNKNOWN; absolute mode is gated.

Hardware-upgrade path
---------------------
* Fit a compliance matrix (or FEM surrogate) from the 6-DoF calibration grid;
  populate mech.E_body_kPa / mech.friction_mu and switch to absolute mode.
"""
from __future__ import annotations

from ..core_types import Deformation, GroundTruth
from ..params import ParameterRegistry


def run(gt: GroundTruth, reg: ParameterRegistry, scene: dict) -> Deformation:
    l1 = scene.get("l1", {})
    comp = float(l1.get("normal_compression_mm", 0.30))
    shear_travel = float(l1.get("shear_travel_mm", 0.50))

    # Touch the UNKNOWN moduli/friction so that --allow-placeholders is honoured
    # and absolute claims are never made silently. In relative mode these reads
    # are skipped (kept commented to document the gate).
    if reg.allow_placeholders:
        _ = reg.get("mech.E_body_kPa")
        _ = reg.get("mech.friction_mu")

    dz, dx, dy = {}, {}, {}
    for f in gt.fingers:
        dz[f] = gt.normal[f] * comp
        dx[f] = gt.shear_x[f] * shear_travel
        dy[f] = gt.shear_y[f] * shear_travel
    return Deformation(dz_mm=dz, dx_mm=dx, dy_mm=dy)
