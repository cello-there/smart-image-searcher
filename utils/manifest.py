# utils/manifest.py
from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Dict, Tuple, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def _normalize_rel(path: Path, root: Path) -> str:
    # Keep Windows-style backslashes, since your CSV uses them
    return os.path.relpath(path, root)

def scan_image_dir(image_dir: str) -> Dict[str, dict]:
    """Return {relpath: {size:int, mtime:int}} snapshot for all images."""
    root = Path.cwd()
    base = Path(image_dir)
    out: Dict[str, dict] = {}
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = _normalize_rel(p, root)
        out[rel] = {"size": int(st.st_size), "mtime": int(st.st_mtime)}
    return out

def load_manifest(path: str) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_manifest(path: str, manifest: Dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def diff_manifests(old: Dict[str, dict], new: Dict[str, dict]) -> Tuple[List[str], List[str], List[str]]:
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    modified = []
    for k in (old_keys & new_keys):
        o, n = old[k], new[k]
        if o.get("size") != n.get("size") or o.get("mtime") != n.get("mtime"):
            modified.append(k)
    return added, removed, modified
