"""End-to-end: topology, simulation, determinism, and all three exporters."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sentrixsim.pipeline import simulate
from sentrixsim.layers.l7_export import lerobot, mcap, parquet

EVENTS = ["idle", "tap", "press", "hold", "shear", "slip", "release", "pinch", "grasp"]
TINY_DESCRIPTOR = Path(__file__).resolve().parent / "Mark2_tiny.json"


def test_topology_counts(config_dir):
    ep = simulate("idle", config_dir, seed=0)
    assert ep.meta["n_bmm350"] == 21
    assert ep.meta["n_lis2dtw12"] == 3
    assert ep.meta["layout"] == "Mark2_v1"


def test_all_events_run(config_dir):
    for ev in EVENTS:
        ep = simulate(ev, config_dir, seed=0)
        assert ep.n_samples > 1
        assert ep.aligned["B_read_uT"].shape[1] == 21
        assert "phase" in ep.labels
        assert ep.meta["physics_fidelity"] == "relative"


def test_ground_truth_and_estimate_present(config_dir):
    ep = simulate("press", config_dir, seed=0)
    assert "label.index.normal_force" in ep.labels
    assert ep.label_meta["label.index.normal_force"]["source"] == "ground_truth"
    assert "est.index.contact" in ep.labels
    assert ep.label_meta["est.index.contact"]["source"] == "simulated_estimate"


def test_determinism(config_dir):
    a = simulate("tap", config_dir, seed=42)
    b = simulate("tap", config_dir, seed=42)
    assert np.array_equal(a.aligned["B_lsb"], b.aligned["B_lsb"])


def test_seed_changes_noise(config_dir):
    a = simulate("tap", config_dir, seed=1)
    b = simulate("tap", config_dir, seed=2)
    assert not np.array_equal(a.aligned["B_read_uT"], b.aligned["B_read_uT"])


def test_decode_detects_press_contact(config_dir):
    ep = simulate("press", config_dir, seed=0)
    est = ep.labels["est.index.contact"]
    gt = ep.labels["label.index.contact"]
    # decoded contact should overlap ground-truth contact for most of the plateau
    overlap = (est & gt).sum() / max(gt.sum(), 1)
    assert overlap > 0.5


def test_parquet_roundtrip(config_dir, tmp_path):
    ep = simulate("shear", config_dir, seed=0)
    path = parquet.write(ep, tmp_path)
    t = pq.read_table(path)
    assert t.num_rows == ep.n_samples
    assert b"sentrixsim_meta" in t.schema.metadata
    # canonical sensor_id-keyed columns; no ordinal Layout-B names
    assert "mag.bmm_thumb_0.bx_uT" in t.column_names
    assert "dyn.lis_thumb.ax_g" in t.column_names
    assert "dyn.lis_thumb.temp_c" in t.column_names
    assert not any(c.startswith("tactile.b") for c in t.column_names)


def test_mcap_writes(config_dir, tmp_path):
    ep = simulate("tap", config_dir, seed=0)
    path = mcap.write(ep, tmp_path)
    assert path.exists() and path.stat().st_size > 0


def test_lerobot_layout(config_dir, tmp_path):
    ep = simulate("grasp", config_dir, seed=0)
    root = lerobot.write(ep, tmp_path)
    assert (root / "meta" / "info.json").exists()
    assert (root / "data" / "chunk-000" / "file-000.parquet").exists()


def test_slip_has_ground_truth_slip(config_dir):
    ep = simulate("slip", config_dir, seed=0)
    assert ep.labels["label.index.slip"].any()


# ---- Migration Phase 1: topology-driven ----

def test_descriptor_faithful_to_yaml_topology(config_dir):
    """GOLDEN: the bundled Mark2_v1 descriptor must reproduce build_topology's
    geometry/order/ids exactly. Physics is a pure function of these, so this is
    the value-identical guarantee for the topology-source swap."""
    from sentrix_contracts import bundled_descriptor_path, load_descriptor
    from sentrixsim.params import ParameterRegistry
    from sentrixsim.topology import build_topology, from_descriptor

    reg = ParameterRegistry.load(config_dir / "parameters.yaml")
    yaml_topo = build_topology(config_dir / "topology_layoutB.yaml", reg)
    desc_topo = from_descriptor(load_descriptor(bundled_descriptor_path("Mark2_v1")))

    assert len(yaml_topo.sites) == len(desc_topo.sites)
    for a, b in zip(yaml_topo.sites, desc_topo.sites):
        assert a.sensor_id == b.sensor_id and a.kind == b.kind and a.finger == b.finger
        np.testing.assert_allclose(a.position_mm, b.position_mm, atol=1e-6)


def test_descriptor_determinism(config_dir):
    a = simulate("tap", config_dir, seed=7)
    b = simulate("tap", config_dir, seed=7)
    assert np.array_equal(a.aligned["B_lsb"], b.aligned["B_lsb"])
    assert a.meta["descriptor_version"] == "Mark2_v1"
    assert a.meta["bmm_sensor_ids"][0] == "bmm_thumb_0"


def test_second_descriptor_end_to_end(config_dir, tmp_path):
    """A different revision (6 BMM + 1 LIS) runs the whole pipeline + export with
    NO code change - the acceptance gate for topology-drive."""
    ep = simulate("press", config_dir, seed=0, descriptor=str(TINY_DESCRIPTOR))
    assert ep.meta["n_bmm350"] == 6
    assert ep.meta["n_lis2dtw12"] == 1
    assert ep.aligned["B_read_uT"].shape[1] == 6
    assert ep.aligned["accel_read_g"].shape[1] == 1

    path = parquet.write(ep, tmp_path)
    t = pq.read_table(path)
    mag_cols = [c for c in t.column_names if c.startswith("mag.")]
    accel_cols = [c for c in t.column_names if c.startswith("dyn.") and c.endswith("_g")]
    assert len(mag_cols) == 6 * 3
    assert len(accel_cols) == 1 * 3
    assert "mag.bmm_index_1.bz_uT" in t.column_names


def test_legacy_column_shim_retired(config_dir, tmp_path):
    """SIM-3: the legacy Layout-B shim is gone — no `legacy_columns` kwarg, and
    output carries only canonical sensor_id-keyed columns."""
    import inspect
    assert "legacy_columns" not in inspect.signature(parquet.write).parameters
    ep = simulate("tap", config_dir, seed=0)
    t = pq.read_table(parquet.write(ep, tmp_path / "canon"))
    assert not any(c.startswith("tactile.b") for c in t.column_names)
    assert not any(c.endswith(".temp_degC") for c in t.column_names)
    assert "mag.bmm_thumb_0.bx_uT" in t.column_names
