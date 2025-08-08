# search/indexer.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # faiss is optional
    faiss = None


class VectorIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = int(dim)
        self.metric = metric  # "cosine" or "l2"
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None
        self._faiss = None

    @staticmethod
    def from_config(cfg: dict) -> "VectorIndex":
        # Try to infer dim from sidecar if present; default 512
        dim = 512
        meta_path = Path(cfg["metadata_csv"]).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dim = int(meta.get("dim", dim))
            except Exception:
                pass
        idx = VectorIndex(dim=dim, metric=cfg.get("distance", "cosine"))
        idx.load(cfg["index_path"])
        return idx

    # ---------- build / add ----------

    def build(self, vectors: np.ndarray, ids: list[str]):
        assert vectors.shape[0] == len(ids), "vectors/ids length mismatch"
        vecs = vectors.astype("float32")
        if self.metric == "cosine":
            vecs /= (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)
        self._vecs = vecs
        self._ids = list(ids)
        self._build_faiss_if_possible()

    def add(self, vectors: np.ndarray, ids: list[str]):
        assert vectors.shape[1] == self.dim, "vector dim mismatch"
        add = vectors.astype("float32")
        if self.metric == "cosine":
            add /= (np.linalg.norm(add, axis=-1, keepdims=True) + 1e-12)
        if self._vecs is None:
            self._vecs = add
            self._ids = list(ids)
        else:
            self._vecs = np.vstack([self._vecs, add])
            self._ids.extend(ids)
        if self._faiss is not None:
            self._faiss.add(add)

    def _build_faiss_if_possible(self):
        if faiss is None or self._vecs is None:
            self._faiss = None
            return
        if self.metric == "cosine":
            self._faiss = faiss.IndexFlatIP(self.dim)
        else:
            self._faiss = faiss.IndexFlatL2(self.dim)
        self._faiss.add(self._vecs)

    # ---------- save / load ----------

    def save(self, path: str):
        """Persist index + sidecars so we can reload safely."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Always write ids sidecar (needed to map FAISS indices -> file paths)
        Path(str(p) + ".ids").write_text("\n".join(self._ids), encoding="utf-8")

        # Save vectors as .npy (useful for clustering / fallback search)
        if self._vecs is not None:
            np.save(str(p) + ".npy", self._vecs.astype("float32"))

        # Save FAISS index if available
        if faiss is not None and self._faiss is not None:
            faiss.write_index(self._faiss, str(p))

    def load(self, path: str):
        """Load if any of (.faiss or .npy) exists; ids are required for mapping."""
        p = Path(path)
        ids_p = Path(str(p) + ".ids")
        npy_p = Path(str(p) + ".npy")

        # If nothing exists, do nothing
        if not (p.exists() or npy_p.exists()):
            return

        # Load ids (may be empty if missing)
        if ids_p.exists():
            self._ids = ids_p.read_text(encoding="utf-8").splitlines()
        else:
            self._ids = []

        # Prefer FAISS file when present
        if faiss is not None and p.exists():
            self._faiss = faiss.read_index(str(p))
            # _vecs optional; only needed for brute-force fallback
            self._vecs = np.load(str(npy_p)) if npy_p.exists() else None
        else:
            # Fallback: pure numpy
            self._faiss = None
            self._vecs = np.load(str(npy_p)) if npy_p.exists() else None

        # If we loaded vecs but not faiss, ensure dims align
        if self._vecs is not None:
            self.dim = int(self._vecs.shape[1])
        # If we have vecs and no faiss, we can still search brute-force

    # ---------- search ----------

    def search(self, q: np.ndarray, topk: int = 10):
        if q.ndim == 1:
            q = q[None, :]
        q = q.astype("float32")
        if self.metric == "cosine":
            q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

        # Fast path: FAISS
        if self._faiss is not None and len(self._ids) > 0:
            D, I = self._faiss.search(q, topk)
            scores = D[0]
            idxs = I[0]
            ids = [self._ids[i] if 0 <= i < len(self._ids) else "" for i in idxs]
            return [(ids[i], float(scores[i])) for i in range(len(ids)) if ids[i]]

        # Fallback: brute-force numpy
        if self._vecs is None or len(self._ids) == 0:
            return []
        if self.metric == "cosine":
            sims = (self._vecs @ q.T).ravel()
            order = np.argsort(-sims)[:topk]
            scores = sims[order]
        else:
            d2 = ((self._vecs - q)**2).sum(axis=1)
            order = np.argsort(d2)[:topk]
            scores = -d2[order]
        ids = [self._ids[i] for i in order]
        return [(ids[i], float(scores[i])) for i in range(len(ids))]
