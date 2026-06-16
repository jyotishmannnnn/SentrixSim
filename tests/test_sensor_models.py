"""Datasheet-fidelity checks on the BMM350 and LIS2DTW12 models."""
import numpy as np

from sentrixsim.layers import l3_bmm350, l4_lis2dtw12
from sentrixsim.layers.l5_noise_drift import NoiseModel
from sentrixsim.params import ParameterRegistry


def _reg(config_dir):
    return ParameterRegistry.load(config_dir / "parameters.yaml")


def test_bmm_noise_matches_datasheet(config_dir):
    reg = _reg(config_dir)
    T, nb = 20000, 1
    B_true = np.zeros((T, nb, 3))           # zero field -> output is pure noise
    out = l3_bmm350.run(B_true, reg, NoiseModel(seed=1))
    std = out["B_read_uT"].std(axis=0)[0]   # per-axis (uT)
    # datasheet: 190 nT xy, 450 nT z -> 0.19/0.45 uT, allow +/-15%
    assert abs(std[0] - 0.190) < 0.030
    assert abs(std[1] - 0.190) < 0.030
    assert abs(std[2] - 0.450) < 0.070


def test_bmm_saturation_and_quant(config_dir):
    reg = _reg(config_dir)
    B_true = np.full((4, 1, 3), 5000.0)     # way past +/-2000 uT
    out = l3_bmm350.run(B_true, reg, NoiseModel(seed=0))
    assert out["sat_flag"].all()
    assert out["B_read_uT"].max() <= 2000.0 + 1e-6
    # quantization step = 0.1 uT
    q = out["B_read_uT"] / 0.1
    assert np.allclose(q, np.round(q))


def test_lis_sensitivity_and_fs(config_dir):
    reg = _reg(config_dir)
    T = 1000
    accel = {"index": np.tile([0.0, 0.0, 1.0], (T, 1))}  # 1 g on z
    temp = np.full(T, 30.0)
    out = l4_lis2dtw12.run(accel, temp, reg, NoiseModel(seed=2), scene={})
    # index is tripod site 1; recovered ~1 g
    z = out["accel_read_g"][:, 1, 2].mean()
    assert abs(z - 1.0) < 0.05
    # LSB at 14-bit, +/-16 g
    lsb = 16.0 / (2 ** 13)
    codes = out["accel_lsb"][:, 1, 2]
    recon = codes * lsb
    assert np.allclose(recon, out["accel_read_g"][:, 1, 2])


def test_lis_clips_to_fs(config_dir):
    reg = _reg(config_dir)
    T = 50
    accel = {"thumb": np.tile([100.0, 0.0, 0.0], (T, 1))}  # 100 g >> 16 g FS
    out = l4_lis2dtw12.run(accel, np.full(T, 25.0), reg, NoiseModel(0), scene={})
    assert out["accel_read_g"][:, 0, 0].max() <= 16.0 + 1e-6
