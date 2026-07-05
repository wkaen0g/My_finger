import json
from pathlib import Path

from microgesture.config import Config


def test_config_reload_updates_values(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"camera": {"width": 320}}), encoding="utf-8")

    config = Config(path=config_path, watch=False)
    assert config.get("camera", "width") == 320

    config_path.write_text(json.dumps({"camera": {"width": 480}}), encoding="utf-8")
    config.reload()
    assert config.get("camera", "width") == 480
