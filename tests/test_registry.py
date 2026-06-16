"""The no-silent-invention guarantee + tier accounting."""
import pytest

from sentrixsim.params import ParameterRegistry, ParamTier, UnknownParameterError


def test_unknown_disabled_raises(config_dir):
    reg = ParameterRegistry.load(config_dir / "parameters.yaml", allow_placeholders=False)
    # mech.E_body_kPa is UNKNOWN + disabled -> reading must raise.
    with pytest.raises(UnknownParameterError):
        reg.get("mech.E_body_kPa")


def test_unknown_placeholder_allowed(config_dir):
    reg = ParameterRegistry.load(config_dir / "parameters.yaml", allow_placeholders=True)
    val = reg.get("mech.E_body_kPa")          # midpoint of placeholder_range [40,120]
    assert val == pytest.approx(80.0)
    assert reg.physics_fidelity() == "placeholder"
    assert reg.used_placeholder is True


def test_known_values_authoritative(config_dir):
    reg = ParameterRegistry.load(config_dir / "parameters.yaml")
    assert reg.get("bmm.range_uT") == 2000.0
    assert reg.get("bmm.noise_xy_nT") == 190.0
    assert reg.get("bmm.noise_z_nT") == 450.0
    assert reg.get("lis.sens_2g_mg_lsb") == 0.244
    assert reg.param("bmm.range_uT").tier == ParamTier.KNOWN


def test_default_fidelity_relative(config_dir):
    reg = ParameterRegistry.load(config_dir / "parameters.yaml")
    assert reg.physics_fidelity() == "relative"


def test_tier_counts_present(config_dir):
    reg = ParameterRegistry.load(config_dir / "parameters.yaml")
    c = reg.counts()
    assert c["KNOWN"] > 0 and c["ESTIMATED"] > 0 and c["UNKNOWN"] > 0
