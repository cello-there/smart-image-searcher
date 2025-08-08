# organize/cluster.py
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Optional (recommended): pip install scikit-learn hdbscan
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
except Exception:
    MiniBatchKMeans = None
    silhouette_score = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None


def load_vectors(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Best-effort: load vectors + ids from FAISS or sidecars."""
    idx_path = Path(cfg["index_path"])
    ids_path = Path(str(idx_path) + ".ids")
    npy_path = Path(str(idx_path) + ".npy")
    # Preferred: sidecar
    if npy_path.exists() and ids_path.exists():
        vecs = np.load(npy_path)
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        return vecs.astype("float32"), ids
    # FAISS flat: try to read and reconstruct all vectors
    if faiss and idx_path.exists():
        index = faiss.read_index(str(idx_path))
        n = index.ntotal
        # Works on flat indexes; IVF/HNSW won’t reconstruct like this
        try:
            xb = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
            vecs = np.array(xb, dtype="float32").reshape(n, index.d)
        except Exception:
            # reconstruct_n is slower but often available on flat indexes
            vecs = np.vstack([index.reconstruct(i) for i in range(n)]).astype("float32")
        ids = Path(str(idx_path) + ".ids").read_text(encoding="utf-8").splitlines()
        return vecs, ids
    raise RuntimeError("No vectors found. Enable sidecar save in reindex or use flat FAISS.")


def _auto_k(vecs: np.ndarray, k_min=8, k_max=40, sample=5000) -> int:
    if MiniBatchKMeans is None or silhouette_score is None:
        return 16
    X = vecs if len(vecs) <= sample else vecs[np.random.choice(len(vecs), sample, replace=False)]
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1, 4):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(X, labels, metric="cosine")
        except Exception:
            s = -1.0
        if s > best_score:
            best_k, best_score = k, s
    return best_k or 16


def cluster_vectors(vecs: np.ndarray, algo="auto", min_size=5, max_k=40):
    # Normalize for cosine
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    if (algo in {"auto", "hdbscan"}) and hdbscan is not None:
        c = hdbscan.HDBSCAN(min_cluster_size=min_size, metric="euclidean")
        labels = c.fit_predict(vecs)
        ncl = len({l for l in labels if l >= 0})
        return labels, ncl
    # Fallback: k-means with auto-k
    k = _auto_k(vecs, k_min=max(4, min_size), k_max=max_k)
    if MiniBatchKMeans is None:
        # dumb fallback: bin by angle
        # NOT great, but avoids adding heavy deps
        ang = np.arctan2(vecs[:,0], vecs[:,1])
        bins = np.linspace(ang.min(), ang.max(), k+1)
        labels = np.digitize(ang, bins) - 1
        return labels, k
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
    labels = km.fit_predict(vecs)
    return labels, k
