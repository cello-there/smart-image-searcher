import csv
from typing import List, Dict
from pathlib import Path
from .indexer import VectorIndex


class Retriever:
    def __init__(self, index: VectorIndex, metadata_csv: str, distance: str = "cosine"):
        self.index = index
        self.metadata = self._load_meta(metadata_csv)
        self.distance = distance

    def _load_meta(self, csv_path: str):
        meta = {}
        p = Path(csv_path)
        if not p.exists():
            return meta
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                meta[row["id"]] = row["image_path"]
        return meta

    def search(self, query_vec, topk: int = 10) -> List[Dict]:
        pairs = self.index.search(query_vec, topk=topk)
        out = []
        for _id, score in pairs:
            out.append({"id": _id, "path": self.metadata.get(_id, _id), "score": score})
        return out