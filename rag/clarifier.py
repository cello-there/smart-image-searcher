from __future__ import annotations
from typing import List
from .prompts import (
    CLARITY_PROMPT,
    CLARIFY_QS_PROMPT,
    INCORPORATE_PROMPT,
    CLARITY_SCHEMA,
    CLARIFY_QS_SCHEMA,
    INCORPORATE_SCHEMA,
    EXTRACT_ENTITIES_PROMPT, EXTRACT_ENTITIES_SCHEMA,
    ENRICH_ENTITY_PROMPT, ENRICH_ENTITY_SCHEMA,
    PROPOSE_ALIAS_PROMPT, PROPOSE_ALIAS_SCHEMA,
)
from utils import llm
import json

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
        if not data or not isinstance(data.get("entities"), list):
            return []
        
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
    
    def enrich_entity(self, name: str, category: str | None, kind: str | None, memory_docs: list[dict], query: str) -> dict:
        """Return {entity:{...}, questions:[{q,key},...]} using the LLM."""
        known = []
        for d in memory_docs or []:
            if d.get("type") == "entity" and d.get("name"):
                known.append(d["name"])
        prompt = ENRICH_ENTITY_PROMPT.format(
            query=query,
            name=name,
            category=category or "null",
            kind=kind or "null",
            known_names=sorted(known),
        )
        data = llm.complete_json(
            prompt,
            ENRICH_ENTITY_SCHEMA,
            temperature=0.1,
            max_tokens=250,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        return data or {"entity": {"aliases": [], "attributes": {}, "tags": [], "description": ""}, "questions": []}

    
    def find_new_entities(self, query: str, memory_docs: list[dict]) -> list[dict]:
            """Use LLM to extract candidate entities from the query and return only
            those not already present in memory (by name/alias)."""
            known = set()
            for d in memory_docs or []:
                if d.get("type") == "entity":
                    if d.get("name"):
                        known.add(d["name"].lower())
                    for a in (d.get("aliases") or []):
                        known.add(str(a).lower())

            prompt = EXTRACT_ENTITIES_PROMPT.format(
                query=query,
                known_names=sorted(list(known))
            )
            data = llm.complete_json(
                prompt,
                EXTRACT_ENTITIES_SCHEMA,
                temperature=0.1,
                max_tokens=200,
                provider=self.cfg.get("llm_provider"),
                model=self.cfg.get("llm_model"),
                cfg=self.cfg,
            )
            out = []
            for e in (data or {}).get("entities", []):
                name = (e.get("name") or "").strip()
                if name and name.lower() not in known:
                    out.append({
                        "type": "entity",
                        "name": name,
                        "category": e.get("category"),
                        "kind": e.get("kind"),
                        "user_owned": True,   # tentative; we confirm below
                        "source": "clarifier",
                        "persistent": True
                    })
            return out

    def propose_alias(self, query: str, entities_ctx: list[dict]) -> list[dict]:
        """Suggest (entity, alias) links with confidence."""
        ents = [
            {"name": e.get("name"), "kind": e.get("kind"), "aliases": e.get("aliases") or []}
            for e in entities_ctx if e.get("type") == "entity"
        ]
        prompt = PROPOSE_ALIAS_PROMPT.format(
            query=query,
            entities=json.dumps(ents, ensure_ascii=False)
        )
        data = llm.complete_json(
            prompt,
            PROPOSE_ALIAS_SCHEMA,
            temperature=0.1,
            max_tokens=200,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        )
        return list((data or {}).get("links", []))