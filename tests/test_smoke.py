from pathlib import Path
from utils.validation import load_config


def test_config_loads():
    cfg = load_config("config.json")
    assert "images_root" in cfg