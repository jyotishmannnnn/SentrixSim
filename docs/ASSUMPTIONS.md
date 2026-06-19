# SentrixSim v1 — Per-module assumptions, limitations & hardware-upgrade path

This is the requirement-8 register: for every module, what it assumes, where it
is limited, and exactly which future hardware measurement replaces the estimate.

## Parameter registry (`params/registry.py`)
- **Assumes:** the YAML is authoritative; confidence scores are human-assigned.
- **Limits:** confidence is judgement, not learned.
- **Upgrade:** edit the YAML entry when a value is measured (tier→KNOWN).

## Topology (`topology.py`) — descriptor-driven (default `Mark2_v1` = Layout B)
- **Assumes:** counts, geometry and sensor ids come from a shared
  `sentrix_contracts` topology descriptor (`from_descriptor`); they are **not**
  hardcoded. The default bundled descriptor `Mark2_v1` is Layout B and is
  generated from `build_topology`, so its centres/offsets carry the same nominal
  assumptions (cluster centres; within-cluster offsets from pitch + arrangement;
  sensor frames aligned to hand frame). The pipeline is count-agnostic — a
  different descriptor runs end-to-end with no code change.
- **Limits:** `geo.sensor_coords`, `geo.tip_radius_mm`, `geo.pad_area_mm2` UNKNOWN.
- **Upgrade:** CT/optical metrology of a first article → set `geo.sensor_coords`
  (or supply a measured descriptor).

## L0 ground truth (`events/generator.py`)
- **Assumes:** scripted gestures; forces NORMALIZED (no Newtons); slip onset
  scripted; object pose is a coarse stub.
- **Limits:** no human variability; absolute force scale absent.
- **Upgrade:** replace scripts with recorded sessions; add `force_scale` (N)
  from the 6-DoF calibration grid (BUILD Part 8 step 11).

## L1 contact mechanics (`layers/l1_contact.py`)
- **Assumes:** linear, decoupled normal/shear lumped compliance (relative mode).
- **Limits:** `mech.E_body_kPa`, `mech.friction_mu` UNKNOWN → absolute mode gated.
- **Upgrade:** DMA/indentation for moduli + tribometer for µ → fit a compliance
  matrix / FEM surrogate; switch to absolute mode.

## L2 magnetic field (`layers/l2_field.py`)
- **Assumes:** single dipole per cluster, isotropic through-thickness remanence,
  1/r³ decay (BUILD Correction 1); no inter-finger crosstalk.
- **Limits:** absolute |ΔB| unknown (∝ `mag.Br_mT`); `mag.field_scale_uT` is a
  presentation scale, **not** a physical claim.
- **Upgrade:** gaussmeter/Helmholtz scan of a jig-magnetized cartridge → set
  `mag.Br_mT` + magnetization map; remove `field_scale_uT`.

## L3 BMM350 (`layers/l3_bmm350.py`)
- **Assumes:** identity per-unit calibration; TCO/TCS/cross-axis/nonlinearity OFF.
- **Limits:** can't reproduce a real per-unit calibration signature pre-hardware.
- **Upgrade:** load S, b (+cross-axis) from the per-unit calibration bundle;
  enable `bmm.tco`/`bmm.tcs` from a thermal-chamber sweep.

## L4 LIS2DW12 (`layers/l4_lis2dtw12.py`)
- **Assumes:** dynamics sites and their order come from the descriptor (Mark2_v1
  → [thumb,index,middle]); HP 14-bit; temp modelled in °C with a coarse quant
  (LSB/°C UNKNOWN); TCoff OFF.
- **Limits:** slip→vibration coupling unknown (injected only if event enables it).
- **Upgrade:** calibrate temp LSB/°C + zero-g TCoff from a thermal sweep; fit a
  measured slip-vibration spectrum from an instrumented slip rig.

## L5 noise & drift (`layers/l5_noise_drift.py`)
- **Assumes:** white, independent noise at datasheet RMS (benign-EMI glove).
- **Limits:** 1/f drift + correlated EMI not modelled (default off).
- **Upgrade:** fit drift PSD + temperature coefficients from logged hardware.

## L6 synchronization (`layers/l6_sync.py`)
- **Assumes:** ideal hub clock (α=1, β=0); skews are TARGETS, not measured;
  latest-at hold to the master grid.
- **Limits:** models the sync architecture, not real jitter (`sync_quality =
  simulated-target`).
- **Upgrade:** measured genlock latency-probe residuals (DERIV R2); fit α,β.

## L7 export (`layers/l7_export/*`)
- **Assumes:** native sparse-cluster schema with **sensor_id-keyed** canonical
  columns (`mag.<sensor_id>.*` / `dyn.<sensor_id>.*`); legacy Layout-B names are
  a back-compat shim under `--legacy-columns`. `[R,U,V]` only via explicit
  `project_ruv`; LeRobot frames at master rate; no video (glove carries none).
- **Limits:** labels inherit L1/L2 magnitude caveats.
- **Upgrade:** attach real camera streams + PTP-aligned video once the capture
  rig exists; adopt the engine's finalized `[R,U,V]` resampling rule.

## Decode demo (`decode.py`)
- **Assumes:** light contact detector (ENGINE 3.2): aggregate → filtfilt → Schmitt.
- **Limits:** measures consistency, not accuracy vs Newtons (CTO 4.3); force is a
  normalized field-deflection proxy.
- **Upgrade:** train the per-unit regression on the 6-DoF grid; report measured
  force-MAE / pose-RMSE in each dataset card.
