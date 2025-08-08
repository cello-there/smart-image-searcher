# rag/clarifier.py
from typing import List
from .prompts import (
    CLARITY_PROMPT, CLARIFY_QS_PROMPT, INCORPORATE_PROMPT,
    CLARITY_SCHEMA, CLARIFY_QS_SCHEMA, INCORPORATE_SCHEMA,
    EXTRACT_ENTITIES_PROMPT, EXTRACT_ENTITIES_SCHEMA,
    ENRICH_ENTITY_PROMPT, ENRICH_ENTITY_SCHEMA,
    PROPOSE_ALIAS_PROMPT, PROPOSE_ALIAS_SCHEMA,
)
from utils import llm

from .normalizers import is_sentenceish
import re

_GENERIC_NAMES = {
    "photo","photos","picture","pictures","image","images","pic","pics",
    "portrait","portraits","selfie","selfies","shot","shots","snapshot","snapshots"
}


class Clarifier:
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
            items.append({k: d.get(k) for k in ("type","name","aliases","category","kind") if k in d})
        return str(items)

    def is_ambiguous(self, query: str, context: dict | None = None) -> bool:
        prompt = CLARITY_PROMPT.format(query=query, memory=self._ctx_summary(context))
        data = llm.complete_json(prompt, CLARITY_SCHEMA, temperature=0.1, max_tokens=200,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        return bool(data and data.get("ambiguous") is True)

    def questions(self, query: str, context: dict | None = None) -> List[str]:
        prompt = CLARIFY_QS_PROMPT.format(query=query, memory=self._ctx_summary(context))
        data = llm.complete_json(prompt, CLARIFY_QS_SCHEMA, temperature=0.2, max_tokens=200,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        return list((data or {}).get("questions", []))[:2]

    def incorporate_answers(self, query: str, answers: dict) -> str:
        prompt = INCORPORATE_PROMPT.format(query=query, answers=str(answers))
        data = llm.complete_json(prompt, INCORPORATE_SCHEMA, temperature=0.1, max_tokens=200,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        return (data or {}).get("rewritten_query", query)

    def find_new_entities(self, query: str, mem_ctx: list[dict]) -> list[dict]:
        prompt = EXTRACT_ENTITIES_PROMPT.format(query=query, memory=self._ctx_summary({"memory": mem_ctx}))
        data = llm.complete_json(prompt, EXTRACT_ENTITIES_SCHEMA, temperature=0.2, max_tokens=200,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        ents = list((data or {}).get("entities", []))

        # known forms (names + aliases) to filter out
        known_forms = set()
        for d in mem_ctx or []:
            if d.get("type") == "entity":
                if d.get("name"): known_forms.add(str(d["name"]).lower())
                for a in (d.get("aliases") or []):
                    known_forms.add(str(a).lower())

        out = []
        for e in ents:
            name = (e.get("name") or "").strip()
            low = name.lower()
            # drop generic words / junk / overly long phrases
            if not name or low in known_forms or low in _GENERIC_NAMES or len(name.split()) > 3:
                continue
            out.append({
                "type": "entity",
                "name": name,
                "category": (e.get("category") or None),
                "kind": e.get("kind"),
                "user_owned": True,
                "source": "clarifier",
                "persistent": True
            })
        return out

    def enrich_entity(self, name: str, category: str | None, kind: str | None,
                      mem_ctx: list[dict], query: str) -> dict:
        prompt = ENRICH_ENTITY_PROMPT.format(
            name=name, category=category or "", kind=kind or "",
            memory=self._ctx_summary({"memory": mem_ctx}), query=query
        )
        data = llm.complete_json(prompt, ENRICH_ENTITY_SCHEMA, temperature=0.2, max_tokens=300,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        qs = (data or {}).get("questions", []) or []
        ent = (data or {}).get("entity", {}) or {}

        # de-dup questions by key & drop already-filled ones
        seen = set()
        filtered = []
        for q in qs:
            k = (q.get("key") or "").strip()
            if not k or k in seen:
                continue
            if k == "kind" and (ent.get("kind") or kind):
                continue
            if k == "aliases" and ent.get("aliases"):
                continue
            if k.startswith("attributes.") and ent.get("attributes", {}).get(k.split(".", 1)[1]):
                continue
            seen.add(k)
            filtered.append(q)
        return {"questions": filtered[:2], "entity": ent}

    def propose_alias(self, query: str, entities_ctx: list[dict]) -> list[dict]:
        # Flatten known names/aliases for the prompt
        names = []
        mem_summary = []
        for d in entities_ctx or []:
            if d.get("type") != "entity":
                continue
            forms = [str(d.get("name", "")).strip()] + [str(a).strip() for a in (d.get("aliases") or [])]
            forms = [f for f in forms if f]
            if forms:
                names.extend(forms)
                mem_summary.append({"name": d.get("name"), "aliases": d.get("aliases", [])})

        prompt = PROPOSE_ALIAS_PROMPT.format(
            query=query,
            names=str(sorted(set(names))),
            memory=str(mem_summary),
        )
        data = llm.complete_json(prompt, PROPOSE_ALIAS_SCHEMA, temperature=0.1, max_tokens=200,
                                 provider=self.cfg.get("llm_provider"), model=self.cfg.get("llm_model"), cfg=self.cfg)
        links = list((data or {}).get("links", []))

        # Extra guardrails
        filtered = []
        q_lower = query.lower()
        q_tokens = set(re.findall(r"[a-z0-9']+", q_lower))

        for link in links:
            ent = (link.get("entity") or "").strip()
            alias = (link.get("alias") or "").strip()
            if not ent or not alias:
                continue
            a_low = alias.lower()

            # reject sentence-ish or long phrases or generics
            if is_sentenceish(alias) or len(alias.split()) > 3 or a_low in _GENERIC_NAMES:
                continue

            # alias must appear in query (token or substring)
            if a_low not in q_lower and a_low not in q_tokens:
                continue

            # keep confidence if present, else default to 0.5 (will be filtered later in main)
            conf = float(link.get("confidence") or 0.5)
            filtered.append({"entity": ent, "alias": alias, "confidence": conf})

        return filtered
