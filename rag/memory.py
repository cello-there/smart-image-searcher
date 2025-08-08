from __future__ import annotations
from pathlib import Path
import json
import os
from .prompts import ANSWERS_TO_MEMORY_PROMPT, MEMORY_DOCS_SCHEMA
from utils import llm

class MemoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.docs: list[dict] = []

    def load(self) -> "MemoryStore":
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.docs = [json.loads(line) for line in f if line.strip()]
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def upsert(self, doc: dict) -> None:
        self.docs.append(doc)

    def upsert_from_answers(self, answers: dict):
        """Use LLM to map clarifier answers to structured memory docs.
        Falls back to storing generic context entries if no LLM is configured.
        """
        prompt = ANSWERS_TO_MEMORY_PROMPT.format(answers=json.dumps(answers, ensure_ascii=False))
        data = llm.complete_json(
            prompt,
            MEMORY_DOCS_SCHEMA,
            temperature=0.1,
            max_tokens=300,
            provider=os.getenv("LLM_PROVIDER"),
            model=os.getenv("LLM_MODEL"),
        )
        if not data:
            for k, v in answers.items():
                if v:
                    self.upsert({"type": "context", "key": k, "value": v, "source": "user-input"})
            return
        for doc in data.get("docs", []):
            if isinstance(doc, dict):
                self.upsert(doc)

    def search(self, query: str, topk: int = 5) -> list[dict]:
        q = query.lower()
        hits = [d for d in self.docs if any(q in str(v).lower() for v in d.values())]
        return hits[:topk]

    def all(self) -> list[dict]:
        return list(self.docs)