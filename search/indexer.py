from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None


class VectorIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None
        self._faiss_index = None

    @staticmethod
    def from_config(cfg: dict) -> "VectorIndex":
        # Infer dimension from saved sidecar if available; else 512 default
        meta_path = Path(cfg["metadata_csv"]).with_suffix(".meta.json")
        dim = 512
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                dim = int(meta.get("dim", dim))
            except Exception:
                pass
        idx = VectorIndex(dim=dim, metric=cfg.get("distance", "cosine"))
        idx.load(cfg["index_path"])  # if exists
        return idx

    def _build_faiss(self):
        if faiss is None:
            return None
        if self.metric == "cosine":
            # Use IP with normalized vectors
            self._faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            self._faiss_index = faiss.IndexFlatL2(self.dim)
        if self._vecs is not None and len(self._ids) == len(self._vecs):
            self._faiss_index.add(self._vecs.astype("float32"))
        return self._faiss_index

    def build(self, vectors: np.ndarray, ids: list[str]):
        assert vectors.shape[0] == len(ids)
        self._ids = list(ids)
        self._vecs = vectors.astype("float32")
        if self.metric == "cosine":
            # normalize in-place
            n = np.linalg.norm(self._vecs, axis=-1, keepdims=True) + 1e-12
            self._vecs /= n
        if faiss:
            self._build_faiss()

    def add(self, vectors: np.ndarray, ids: list[str]):
        vectors = vectors.astype("float32")
        if self.metric == "cosine":
            vectors = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)
        if self._vecs is None:
            self._vecs = vectors
            self._ids = list(ids)
        else:
            self._vecs = np.vstack([self._vecs, vectors])
            self._ids.extend(ids)
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Always persist ids for mapping
        Path(str(p) + ".ids").write_text("\n".join(self._ids), encoding="utf-8")

        if faiss and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(p))
            return

        # numpy fallback
        if self._vecs is None or len(self._ids) == 0:
            return  # nothing to save
        np.save(str(p) + ".npy", self._vecs.astype("float32"))


    def load(self, path: str):
        p = Path(path)
        ids_path = Path(str(p) + ".ids")
        npy_path = Path(str(p) + ".npy")

        # Prefer FAISS file if present
        if faiss and p.exists():
            self._faiss_index = faiss.read_index(str(p))
            if ids_path.exists():
                self._ids = ids_path.read_text(encoding="utf-8").splitlines()
            else:
                self._ids = []  # will guard in search()
            self._vecs = None
            return

        # Numpy fallback
        if not npy_path.exists():
            return
        try:
            arr = np.load(str(npy_path), allow_pickle=False)
            if arr.dtype == object:
                raise ValueError("legacy object array")
            self._vecs = arr.astype("float32")
            if self.metric == "cosine":
                self._vecs /= (np.linalg.norm(self._vecs, axis=-1, keepdims=True) + 1e-12)
            self._ids = ids_path.read_text(encoding="utf-8").splitlines() if ids_path.exists() else []
        except Exception:
            # Treat as corrupt; force rebuild on next reindex
            self._vecs, self._ids, self._faiss_index = None, [], None


    def search(self, q: np.ndarray, topk: int = 10):
        q = q.astype("float32")[None, :]
        if self.metric == "cosine":
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

        # If nothing indexed, bail gracefully
        if (self._faiss_index is None and (self._vecs is None or len(self._ids) == 0)) or len(self._ids) == 0:
            return []

        if self._faiss_index is not None:
            D, I = self._faiss_index.search(q, topk)
            idxs = [i for i in I[0] if 0 <= i < len(self._ids)]
            scores = [float(D[0][k]) for k, i in enumerate(I[0]) if 0 <= i < len(self._ids)]
            ids = [self._ids[i] for i in idxs]
            return list(zip(ids, scores))

        # numpy brute-force
        if self.metric == "cosine":
            sims = (self._vecs @ q.T).ravel()
            order = np.argsort(-sims)[:topk]
            return [(self._ids[i], float(sims[i])) for i in order]
        d2 = ((self._vecs - q) ** 2).sum(axis=1)
        order = np.argsort(d2)[:topk]
        return [(self._ids[i], float(-d2[i])) for i in order]
