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
        meta_path = Path(cfg["metadata_csv"]).with_suffix(".meta.json")
        dim = 512
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dim = int(meta.get("dim", dim))
            except Exception:
                pass
        idx = VectorIndex(dim=dim, metric=cfg.get("distance", "cosine"))
        idx.load(cfg["index_path"])
        return idx

    def _build_faiss(self):
        if faiss is None:
            return None
        self._faiss_index = faiss.IndexFlatIP(self.dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.dim)
        if self._vecs is not None and len(self._ids) == len(self._vecs):
            self._faiss_index.add(self._vecs.astype("float32"))
        return self._faiss_index

    def ntotal(self) -> int:
        if self._faiss_index is not None:
            return self._faiss_index.ntotal
        return 0 if self._vecs is None else int(self._vecs.shape[0])

    def build(self, vectors: np.ndarray, ids: list[str]):
        assert vectors.shape[0] == len(ids)
        self._ids = list(ids)
        self._vecs = vectors.astype("float32")
        if self.metric == "cosine":
            n = np.linalg.norm(self._vecs, axis=-1, keepdims=True) + 1e-12
            self._vecs /= n
        if faiss:
            self._build_faiss()

    def add(self, vectors: np.ndarray, ids: list[str]):
        vectors = vectors.astype("float32")
        if self.metric == "cosine":
            vectors = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)

        # If FAISS is present, extend the FAISS index and only extend _ids.
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)
            self._ids.extend(ids)
            return

        # Numpy fallback
        if self._vecs is None:
            self._vecs = vectors
            self._ids = list(ids)
        else:
            self._vecs = np.vstack([self._vecs, vectors])
            self._ids.extend(ids)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # ALWAYS write the sidecar to keep FAISS rows -> IDs consistent
        Path(str(p) + ".ids").write_text("\n".join(self._ids), encoding="utf-8")
        if faiss and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(p))
        else:
            np.save(str(p) + ".npy", self._vecs)

    def load(self, path: str):
        p = Path(path)
        ids_path = Path(str(p) + ".ids")
        if not p.exists() and not Path(str(p) + ".npy").exists():
            return

        if faiss and p.exists():
            self._faiss_index = faiss.read_index(str(p))
            self._ids = ids_path.read_text(encoding="utf-8").splitlines() if ids_path.exists() else []
            # Sanity: FAISS rows must match _ids length
            if len(self._ids) != self._faiss_index.ntotal:
                # Invalidate so the caller can rebuild; safer than returning bad mappings
                self._faiss_index = None
                self._vecs = None
                self._ids = []
        else:
            self._vecs = np.load(str(p) + ".npy")
            self._ids = ids_path.read_text(encoding="utf-8").splitlines()

    def search(self, q: np.ndarray, topk: int = 10):
        q = q.astype("float32")[None, :]
        if self.metric == "cosine":
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

        if self._faiss_index is not None:
            D, I = self._faiss_index.search(q, topk)
            scores = D[0]
            idxs = I[0]
            ids = []
            for i in idxs:
                if 0 <= i < len(self._ids):
                    ids.append(self._ids[i])
                else:
                    ids.append("")  # guard against bad mapping
            return [(i, float(s)) for i, s in zip(ids, scores) if i]
        else:
            if self._vecs is None or len(self._ids) == 0:
                return []
            if self.metric == "cosine":
                sims = (self._vecs @ q.T).ravel()
                order = np.argsort(-sims)[:topk]
                return [(self._ids[i], float(sims[i])) for i in order]
            else:
                d2 = ((self._vecs - q) ** 2).sum(axis=1)
                order = np.argsort(d2)[:topk]
                return [(self._ids[i], float(-d2[i])) for i in order]