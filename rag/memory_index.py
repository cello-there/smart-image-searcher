from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def _doc_to_text(doc: dict) -> str:
    parts = []
    for k in ("type", "key", "name", "value"):
        if k in doc and doc[k]:
            parts.append(str(doc[k]))
    if doc.get("aliases"):
        parts.extend([str(a) for a in doc["aliases"]])
    if doc.get("tags"):
        parts.extend([str(t) for t in doc["tags"]])
    return " | ".join(parts)


class MemoryVectorIndex:
    """FAISS (or numpy) index over memory documents for RAG context."""

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self._ids: List[str] = []
        self._vecs: np.ndarray | None = None
        self._docs: List[dict] = []
        self._faiss = None

    @staticmethod
    def from_docs(docs: List[dict], text_embedder, path: str, metric: str = "cosine") -> "MemoryVectorIndex":
        # Embed docs
        texts = [_doc_to_text(d) for d in docs]
        if len(texts) == 0:
            # empty index with default dim 512
            idx = MemoryVectorIndex(dim=512, metric=metric)
            return idx
        vecs = np.vstack([text_embedder.encode_text(t) for t in texts]).astype("float32")
        dim = vecs.shape[1] if vecs.ndim == 2 else vecs.shape[0]
        idx = MemoryVectorIndex(dim=dim, metric=metric)
        idx.build(vecs, [str(i) for i in range(len(texts))], docs)
        idx.save(path)
        return idx

    def build(self, vectors: np.ndarray, ids: List[str], docs: List[dict]):
        self._vecs = vectors
        if self.metric == "cosine":
            self._vecs = self._vecs / (np.linalg.norm(self._vecs, axis=-1, keepdims=True) + 1e-12)
        self._ids = list(ids)
        self._docs = list(docs)
        if faiss is not None:
            self._faiss = faiss.IndexFlatIP(self.dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.dim)
            self._faiss.add(self._vecs)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"dim": self.dim, "count": len(self._ids)}
        Path(str(p) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        # Store docs for transparency/debugging
        Path(str(p) + ".docs.json").write_text(json.dumps(self._docs, ensure_ascii=False, indent=2), encoding="utf-8")
        if faiss is not None and self._faiss is not None:
            faiss.write_index(self._faiss, str(p))
        else:
            np.save(str(p) + ".npy", self._vecs)
            Path(str(p) + ".ids").write_text("\n".join(self._ids), encoding="utf-8")

    def search(self, qvec: np.ndarray, topk: int = 5) -> List[Tuple[dict, float]]:
        q = qvec.astype("float32")[None, :]
        if self.metric == "cosine":
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        if self._faiss is not None:
            D, I = self._faiss.search(q, topk)
            idxs = I[0]
            scores = D[0]
        else:
            if self._vecs is None or len(self._ids) == 0:
                return []
            if self.metric == "cosine":
                sims = (self._vecs @ q.T).ravel()
                idxs = np.argsort(-sims)[:topk]
                scores = sims[idxs]
            else:
                d2 = ((self._vecs - q)**2).sum(axis=1)
                idxs = np.argsort(d2)[:topk]
                scores = -d2[idxs]
        return [(self._docs[i], float(scores[k])) for k, i in enumerate(idxs) if 0 <= i < len(self._docs)]