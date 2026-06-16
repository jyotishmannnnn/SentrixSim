from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config_dir():
    return CONFIG_DIR
