# search/reindex.py
from __future__ import annotations
import os, csv, json, hashlib
from pathlib import Path
import numpy as np
from .indexer import VectorIndex

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def file_sig(p: Path) -> str:
    try:
        st = p.stat()
        return hashlib.sha1(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()
    except Exception:
        return hashlib.sha1(p.name.encode()).hexdigest()

def read_existing(csv_path: Path):
    if not csv_path.exists():
        return {}, []
    sigs, rows = {}, []
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

def incremental_reindex(cfg: dict, embedder, full: bool = False):
    root = Path(cfg["images_root"]).expanduser()
    csv_path = Path(cfg["metadata_csv"]).expanduser()
    index_path = Path(cfg["index_path"]).expanduser()

    # Current filesystem set
    all_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in SUPPORTED:
                all_paths.append(Path(dirpath) / fn)
    current_ids = {str(p) for p in all_paths}

    existing_sigs, existing_rows = read_existing(csv_path)
    existing_ids = {r["id"] for r in existing_rows}

    deleted_ids = existing_ids - current_ids
    if deleted_ids:
        # A flat FAISS index can’t delete rows; safest is a clean rebuild
        print(f"[reindex] detected {len(deleted_ids)} deleted files -> full rebuild")
        full = True

    # Decide what to embed
    to_embed, kept_rows = [], []
    if full:
        # Re-embed everything present
        for p in all_paths:
            sig = file_sig(p)
            to_embed.append((str(p), p, sig))
            kept_rows.append({"id": str(p), "image_path": str(p), "sig": sig})
    else:
        for p in all_paths:
            sig = file_sig(p)
            _id = str(p)
            if existing_sigs.get(_id) != sig:
                to_embed.append((_id, p, sig))
            kept_rows.append({"id": _id, "image_path": str(p), "sig": sig})

    # Embed
    if to_embed:
        ids = [tid for tid, _, _ in to_embed]
        vecs = embedder.encode_images([str(p) for _, p, _ in to_embed])
    else:
        ids, vecs = [], np.zeros((0, 512), dtype="float32")

    # Build/update index
    if full:
        idx = VectorIndex(dim=vecs.shape[1] if vecs.size else 512, metric=cfg.get("distance", "cosine"))
        if vecs.size:
            idx.build(vecs, ids)
    else:
        idx = VectorIndex.from_config(cfg)
        if idx.ntotal() == 0 and vecs.size:
            idx.build(vecs, ids)
        elif vecs.size:
            idx.add(vecs, ids)

    # Persist artifacts
    idx.save(str(index_path))
    write_rows(csv_path, kept_rows)

    # Persist a tiny sidecar with dim to make future loads robust
    try:
        meta_path = csv_path.with_suffix(".meta.json")
        dim = int(vecs.shape[1]) if vecs.size else int(idx.dim)
        meta_path.write_text(json.dumps({"dim": dim}), encoding="utf-8")
    except Exception:
        pass
