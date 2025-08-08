import json
from pathlib import Path

try:
    import jsonschema  # type: ignore
except Exception:  # noqa: BLE001
    jsonschema = None


CFG_SCHEMA = {
    "type": "object",
    "required": [
        "index_path",
        "metadata_csv",
        "memory_path",
        "images_root"
    ],
}


def load_config(path: str) -> dict:
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if jsonschema:
        jsonschema.validate(cfg, CFG_SCHEMA)
    return cfg