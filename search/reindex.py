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
    """Stable signature using name+size+mtime (fast)."""
    try:
        st = p.stat()
        base = f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode()
        return hashlib.sha1(base).hexdigest()
    except Exception:
        return hashlib.sha1(p.name.encode()).hexdigest()


def read_existing(csv_path: Path):
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_path", "sig"])
        w.writeheader()
        w.writerows(rows)


def _scan(root: Path):
    all_paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in SUPPORTED:
                all_paths.append(Path(dp) / fn)
    return all_paths


def incremental_reindex(cfg: dict, embedder, full: bool = False):
    """
    Incremental index with **safe pruning**:
      - If any files were removed/renamed (ids disappear), we **rebuild** the index cleanly.
      - Otherwise, we embed only new/changed files and append.
    """
    root = Path(cfg["images_root"]).expanduser()
    csv_path = Path(cfg["metadata_csv"]).expanduser()
    index_path = Path(cfg["index_path"]).expanduser()

    # previous state
    existing_sigs, _ = read_existing(csv_path)

    # current state
    curr_paths = _scan(root)
    curr_id_set = {str(p) for p in curr_paths}

    # compute sigs + change sets
    to_embed = []       # (id, Path, sig)
    kept_rows = []
    for p in curr_paths:
        _id = str(p)
        sig = file_sig(p)
        kept_rows.append({"id": _id, "image_path": _id, "sig": sig})
        if full or existing_sigs.get(_id) != sig:
            to_embed.append((_id, p, sig))

    removed_ids = set(existing_sigs.keys()) - curr_id_set
    need_clean_rebuild = full or bool(removed_ids)

    # embed vectors
    if need_clean_rebuild:
        # rebuild from scratch for correctness
        ids = [str(p) for p in curr_paths]
        vecs = embedder.encode_images(ids) if ids else np.zeros((0, 512), dtype="float32")
        idx = VectorIndex(dim=(vecs.shape[1] if vecs.size else 512),
                          metric=cfg.get("distance", "cosine"))
        if vecs.size:
            idx.build(vecs, ids)
    else:
        # incremental append
        idx = VectorIndex.from_config(cfg)
        if to_embed:
            ids = [i for i, _, _ in to_embed]
            vecs = embedder.encode_images([str(p) for _, p, _ in to_embed])
            if len(getattr(idx, "_ids", [])) == 0:
                idx.build(vecs, ids)
            else:
                idx.add(vecs, ids)

    # write artifacts
    idx.save(str(index_path))
    write_rows(csv_path, kept_rows)

    # dimension sidecar (so we can re-load with right dim)
    meta_path = csv_path.with_suffix(".meta.json")
    dim = getattr(idx, "dim", 512)
    meta_path.write_text(json.dumps({"dim": int(dim)}), encoding="utf-8")
