from __future__ import annotations
from typing import List
from .prompts import (
    CLARITY_PROMPT,
    CLARIFY_QS_PROMPT,
    INCORPORATE_PROMPT,
    CLARITY_SCHEMA,
    CLARIFY_QS_SCHEMA,
    INCORPORATE_SCHEMA,
)
from utils import llm

class Clarifier:
    """LLM-driven clarifier with no hard-coded heuristics.

    If no LLM provider is configured, methods degrade gracefully:
      - `is_ambiguous` -> False
      - `questions` -> []
      - `incorporate_answers` -> original query
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _ctx_summary(self, context: dict | None) -> str:
        if not context:
            return "[]"
        mem = context.get("memory") if isinstance(context, dict) else None
        if not mem:
            return "[]"
        items = []
        for d in mem[:20]:
            items.append({k: d.get(k) for k in ("type","key","name","aliases","value") if k in d})
        return str(items)

    def is_ambiguous(self, query: str, context: dict | None = None) -> bool:
        prompt = CLARITY_PROMPT.format(query=query, memory=self._ctx_summary(context))
        data = llm.complete_json(
            prompt,
            CLARITY_SCHEMA,
            temperature=0.1,
            max_tokens=200,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        return bool(data and data.get("ambiguous") is True)

    def questions(self, query: str, context: dict | None = None) -> List[str]:
        prompt = CLARIFY_QS_PROMPT.format(query=query, memory=self._ctx_summary(context))
        data = llm.complete_json(
            prompt,
            CLARIFY_QS_SCHEMA,
            temperature=0.2,
            max_tokens=200,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        return list((data or {}).get("questions", []))[:2]

    def incorporate_answers(self, query: str, answers: dict) -> str:
        prompt = INCORPORATE_PROMPT.format(query=query, answers=str(answers))
        data = llm.complete_json(
            prompt,
            INCORPORATE_SCHEMA,
            temperature=0.1,
            max_tokens=200,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        return (data or {}).get("rewritten_query", query)