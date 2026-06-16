"""End-to-end: topology, simulation, determinism, and all three exporters."""
import numpy as np
import pyarrow.parquet as pq

from sentrixsim.pipeline import simulate
from sentrixsim.layers.l7_export import lerobot, mcap, parquet

EVENTS = ["idle", "tap", "press", "hold", "shear", "slip", "release", "pinch", "grasp"]


def test_topology_counts(config_dir):
    ep = simulate("idle", config_dir, seed=0)
    assert ep.meta["n_bmm350"] == 21
    assert ep.meta["n_lis2dtw12"] == 3
    assert ep.meta["layout"] == "layout_B"


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
    assert any(c.startswith("tactile.b00") for c in t.column_names)


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
