from __future__ import annotations
import os
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from .indexer import VectorIndex

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_sig(p: Path) -> str:
    """Stable content signature using filename, size, and mtime."""
    try:
        stat = p.stat()
        base = f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}".encode()
        return hashlib.sha1(base).hexdigest()
    except Exception:
        return hashlib.sha1(p.name.encode()).hexdigest()


def read_existing(csv_path: Path):
    """Return (id->sig map, existing rows) from metadata CSV if present."""
    if not csv_path.exists():
        return {}, []
    sigs = {}
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            sigs[row["id"]] = row.get("sig", "")
    return sigs, rows


def write_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "image_path", "sig"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def incremental_reindex(cfg: dict, embedder, full: bool = False):
    """Incrementally (re)index images under images_root.

    - When `full` is False, only new/changed files (by signature) are embedded.
    - Maintains CSV metadata and a FAISS (or numpy) index.
    """
    root = Path(cfg["images_root"]).expanduser()
    csv_path = Path(cfg["metadata_csv"]).expanduser()
    index_path = Path(cfg["index_path"]).expanduser()

    existing_sigs, _ = read_existing(csv_path)

    # Scan filesystem for supported image files
    all_paths: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in SUPPORTED:
                all_paths.append(Path(dirpath) / fn)

    # Compute signatures and decide which files need embedding
    to_embed = []  # list of (id, path, sig)
    kept_rows = []
    for p in all_paths:
        sig = file_sig(p)
        _id = str(p)
        if full or existing_sigs.get(_id) != sig:
            to_embed.append((_id, p, sig))
        kept_rows.append({"id": _id, "image_path": str(p), "sig": sig})

    # Embed new/changed files
    if to_embed:
        ids = [tid for tid, _, _ in to_embed]
        vecs = embedder.encode_images([str(p) for _, p, _ in to_embed])
    else:
        ids, vecs = [], np.zeros((0, 512), dtype="float32")

    # Build or update index
    idx = VectorIndex.from_config(cfg)
    if len(getattr(idx, "_ids", [])) == 0 and len(vecs) > 0:
        idx.build(vecs, ids)
    elif len(vecs) > 0:
        idx.add(vecs, ids)

    # Save artifacts
    idx.save(str(index_path))
    write_rows(csv_path, kept_rows)

    # Persist dimension sidecar so VectorIndex.from_config can infer dim next time
    try:
        meta_path = csv_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({"dim": int(vecs.shape[1] if vecs.size else idx.dim)}), encoding="utf-8")
    except Exception:
        pass