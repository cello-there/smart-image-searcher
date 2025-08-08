from __future__ import annotations
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from search.reindex import incremental_reindex


def _load_paths(metadata_csv: str) -> List[str]:
    p = Path(metadata_csv)
    if not p.exists():
        return []
    out = []
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row["image_path"])
    return out


def cluster_images(cfg: dict, embedder, k: int = 8, move: bool = False,
                   out_root: str = "clusters", dry_run: bool = False, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Clusters all images in the current metadata into K groups.
    - Returns (labels, centers, paths)
    - If move=True, moves files to `<images_root>/<out_root>/cluster_<i>/...` and reindexes.
    """
    image_paths = _load_paths(cfg["metadata_csv"])
    if not image_paths:
        print("No metadata found. Run: python main.py reindex")
        return np.array([]), np.zeros((0,)), []

    vecs = embedder.encode_images(image_paths)  # re-embed for clustering
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(vecs)

    # Save a simple report
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    rep = data_dir / "clusters.csv"
    with rep.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "image_path"])
        for lab, path in zip(labels, image_paths):
            w.writerow([int(lab), path])
    print(f"Wrote cluster assignments -> {rep}")

    # Optionally move files
    if move:
        images_root = Path(cfg["images_root"])
        out_base = images_root / out_root
        for lab, src in zip(labels, image_paths):
            src_p = Path(src)
            # keep original filename; put under cluster_<i>/
            dst_dir = out_base / f"cluster_{int(lab):02d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src_p.name
            if dry_run:
                print(f"[dry-run] move {src_p} -> {dst}")
            else:
                try:
                    dst.unlink(missing_ok=True)
                    shutil.move(str(src_p), str(dst))
                except Exception as e:
                    print(f"Move failed for {src_p}: {e}")
        if not dry_run:
            # Reindex cleanly so searches reflect new paths
            from search.embedder import ImageEmbedder
            incremental_reindex(cfg, ImageEmbedder(cfg), full=True)

    return labels, km.cluster_centers_, image_paths
