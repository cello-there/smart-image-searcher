from __future__ import annotations
from .prompts import AUGMENT_PROMPT, AUGMENT_SCHEMA
from utils import llm

class QueryAugmenter:
    """LLM-driven augmentation; no hard-coded synonyms.

    If no LLM is configured, returns an identity expansion.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def expand(self, query: str, context: dict | None = None) -> dict:
        prompt = AUGMENT_PROMPT.format(query=query, context=str(context or {}))
        data = llm.complete_json(
            prompt,
            AUGMENT_SCHEMA,
            temperature=0.2,
            max_tokens=300,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        if not data:
            return {"expanded_query": query, "terms": [], "entities": [], "filters": {}}
        data.setdefault("expanded_query", query)
        data.setdefault("terms", [])
        data.setdefault("entities", [])
        data.setdefault("filters", {})
        return data