from __future__ import annotations
import json
import re
from typing import Any, Dict

from .prompts import AUGMENT_PROMPT, AUGMENT_SCHEMA
from utils import llm

class QueryAugmenter:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _ctx_summary(self, context: dict | None) -> str:
        if not context:
            return "[]"
        mem = (context or {}).get("memory") or []
        items = []
        for d in mem[:12]:
            if d.get("type") == "entity":
                items.append({
                    "name": d.get("name"),
                    "aliases": d.get("aliases", []),
                    "category": d.get("category"),
                    "kind": d.get("kind"),
                    "attributes": d.get("attributes", {})
                })
        return json.dumps(items, ensure_ascii=False)

    def _postprocess(self, expanded: str, original: str) -> str:
        s = expanded.strip()
        if not s:
            return original

        # strip schema-ish prefixes like person:Jason
        s = re.sub(r"\b(?:person|subject|category)\s*:\s*", "", s, flags=re.IGNORECASE)

        # normalize synonyms to 'photos'
        s = re.sub(r"\b(images?|pictures?|pics?)\b", "photos", s, flags=re.IGNORECASE)

        # collapse multiple spaces
        s = re.sub(r"\s{2,}", " ", s).strip()

        # if it still contains standalone 'or', keep only the first segment that has >=2 tokens
        if re.search(r"\bor\b", s, flags=re.IGNORECASE):
            parts = [p.strip() for p in re.split(r"\bor\b", s, flags=re.IGNORECASE)]
            for p in parts:
                if len(re.findall(r"\w+", p)) >= 2:
                    s = p
                    break

        return s or original

    def expand(self, query: str, context: dict | None) -> Dict[str, Any]:
        prompt = AUGMENT_PROMPT.format(query=query, context=self._ctx_summary(context))
        data = llm.complete_json(
            prompt,
            AUGMENT_SCHEMA,
            temperature=0.2,
            max_tokens=220,
            provider=self.cfg.get("llm_provider"),
            model=self.cfg.get("llm_model"),
            cfg=self.cfg,
        ) or {}

        expanded = data.get("expanded_query", query)
        expanded = self._postprocess(expanded, query)

        # hydrate minimal terms/entities for logging/UI
        terms = list({t for t in (data.get("terms") or []) if isinstance(t, str) and t.strip()})
        entities = list({e for e in (data.get("entities") or []) if isinstance(e, str) and e.strip()})
        return {
            "expanded_query": expanded,
            "terms": terms,
            "entities": entities,
            "filters": data.get("filters") or {}
        }
