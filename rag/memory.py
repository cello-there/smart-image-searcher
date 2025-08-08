from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re

from .prompts import ANSWERS_TO_MEMORY_PROMPT, MEMORY_DOCS_SCHEMA
from utils import llm

# ----------------------------
# Helpers / normalization
# ----------------------------

_TRIVIAL_NEGATIONS = {"", "no", "none", "n/a", "na", "null", "nil"}
# --- add near top of file (below imports) ---
def _norm(s):
    return str(s).strip().lower() if s is not None else ""

def _name_set(d: dict) -> set[str]:
    """Lowercased set of the canonical name + all aliases."""
    names = set()
    if d.get("name"):
        names.add(_norm(d["name"]))
    for a in (d.get("aliases") or []):
        if a:
            names.add(_norm(a))
    return names

def _merge_entity_docs(base: dict, inc: dict) -> dict:
    """Merge inc into base (entity docs): union aliases/tags, merge attributes, fill blanks."""
    out = dict(base)
    # prefer already-canonical name from base; if names differ, keep base name and add inc name as alias
    if inc.get("name") and _norm(inc["name"]) not in _name_set(base):
        out["aliases"] = sorted(set((out.get("aliases") or []) + [inc["name"]]))
    # union aliases
    if inc.get("aliases"):
        out["aliases"] = sorted(set((out.get("aliases") or []) + list(inc["aliases"])))
    # kind/category: adopt if base missing
    if not out.get("kind") and inc.get("kind"):
        out["kind"] = inc["kind"]
    if not out.get("category") and inc.get("category"):
        out["category"] = inc["category"]
    # attributes (inc wins per-key)
    if inc.get("attributes"):
        attrs = dict(out.get("attributes") or {})
        attrs.update(inc["attributes"])
        out["attributes"] = attrs
    # tags union
    if inc.get("tags"):
        out["tags"] = sorted(set((out.get("tags") or []) + list(inc["tags"])))
    # description: keep the longer/more-informative
    if inc.get("description"):
        if not out.get("description") or len(str(inc["description"])) > len(str(out["description"])):
            out["description"] = inc["description"]
    # persistent stickiness
    if inc.get("persistent"):
        out["persistent"] = True
    # user_owned stickiness to True
    if inc.get("user_owned") is True:
        out["user_owned"] = True
    return out

def _strip_nullish(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [], {}, "null", "None")}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _looks_trivial_context(d: dict) -> bool:
    desc = (d.get("description") or "").strip().lower()
    if not desc:
        return True
    bad_bits = [
        "any photo at night",
        "any photos at night",
        "does not have a preference",
        "no preference",
        "just any photo",
        "general photos",
    ]
    return any(b in desc for b in bad_bits)

def _looks_trivial_preference(d: dict) -> bool:
    val = str(d.get("value", "")).strip().lower()
    return (val in _TRIVIAL_NEGATIONS) or (len(val) < 2)

def _looks_valid_entity(d: dict) -> bool:
    # Require name and kind for entities; category optional but useful
    return bool(d.get("name")) and bool(d.get("kind"))

def _make_key(d: dict) -> str:
    """Stable merge key so duplicates update instead of multiplying."""
    t = d.get("type") or "unknown"
    cat = d.get("category") or "na"
    name = _slug(d.get("name", "") or "na")
    kind = _slug(d.get("kind", "") or "na")
    return f"{t}:{cat}:{name}:{kind}"

def _normalize_docs(docs: list[dict]) -> list[dict]:
    """Clean, enrich, and validate LLM-produced docs before saving."""
    out: list[dict] = []
    for raw in docs or []:
        d = _strip_nullish(dict(raw))
        t = d.get("type")
        if t not in {"entity", "preference", "context"}:
            continue

        # Drop trivial / invalid
        if t == "context" and _looks_trivial_context(d):
            continue
        if t == "preference" and _looks_trivial_preference(d):
            continue
        if t == "entity" and not _looks_valid_entity(d):
            continue

        # Enrich with consistent, helpful fields
        d.setdefault("aliases", [])
        d.setdefault("attributes", {})  # e.g., {"color":"black","breed":"DLH"}
        d.setdefault("tags", [])        # e.g., ["indoor","family"]
        d.setdefault("description", "") # short human summary

        # Defaults / flags
        d.setdefault("persistent", t == "entity")
        d.setdefault("source", "clarifier")
        if t == "entity":
            d.setdefault("user_owned", True)

        # Merge key + timestamps
        d["key"] = _make_key(d)
        now = _now_iso()
        d.setdefault("created_at", now)
        d["updated_at"] = now

        out.append(d)
    return out

def _merge_key(d: dict):
    return d.get("key") or _make_key(d)

# ----------------------------
# Store
# ----------------------------

class MemoryStore:
    def __init__(self, path: str, cfg: dict | None = None):
        self.path = Path(path)
        self.cfg = cfg or {}
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

    def _merge_upsert(self, doc: dict) -> None:
        """
        Upsert with alias-aware entity merge:
        - For entities: if any name/alias overlaps with an existing entity (by name or alias), merge into it.
        - For non-entities: fallback to previous key-based merge.
        """
        t = doc.get("type")
        if t == "entity":
            inc_names = _name_set(doc)
            # find an existing entity whose name/aliases overlap, preferring exact-name match
            exact_idx = None
            overlap_idx = None
            for i, ex in enumerate(self.docs):
                if ex.get("type") != "entity":
                    continue
                ex_names = _name_set(ex)
                if _norm(ex.get("name")) in inc_names:
                    exact_idx = i
                    break
                if inc_names & ex_names:
                    overlap_idx = i
            idx = exact_idx if exact_idx is not None else overlap_idx
            if idx is not None:
                self.docs[idx] = _strip_nullish(_merge_entity_docs(self.docs[idx], doc))
                return
            # no match: new entity
            self.docs.append(_strip_nullish(doc))
            return

        # non-entity fallback: merge by a coarse key
        def _merge_key(d: dict):
            tt = d.get("type")
            if tt == "preference":
                return (tt, _norm(d.get("value")))
            # context: merge on description text if present
            if tt == "context":
                return (tt, _norm(d.get("description")) or _norm(d.get("value")))
            return (tt, json.dumps(d, sort_keys=True))
        key = _merge_key(doc)
        for i, ex in enumerate(self.docs):
            if _merge_key(ex) == key:
                merged = {**ex, **doc}
                self.docs[i] = _strip_nullish(merged)
                return
        self.docs.append(_strip_nullish(doc))


    def upsert(self, doc: dict) -> None:
        self._merge_upsert(doc)

    def upsert_from_answers(self, answers: dict):
        """Use LLM to map clarifier answers to structured docs; persist only durable ones."""
        prompt = ANSWERS_TO_MEMORY_PROMPT.format(answers=json.dumps(answers, ensure_ascii=False))
        data = llm.complete_json(
            prompt,
            MEMORY_DOCS_SCHEMA,
            temperature=0.1,
            max_tokens=300,
            provider=self.cfg.get("llm_provider") or os.getenv("LLM_PROVIDER"),
            model=self.cfg.get("llm_model") or os.getenv("LLM_MODEL"),
            cfg=self.cfg,
        )

        docs = _normalize_docs((data or {}).get("docs", []))
        for d in docs:
            if d.get("persistent", False):
                self._merge_upsert(d)
        # Intentionally ignore ephemeral / trivial items

    def search(self, query: str, topk: int = 5) -> list[dict]:
        """Very simple keyword search across values, aliases, tags, and attributes."""
        q = query.lower()

        def _as_strings(d: dict):
            vals = [str(v) for v in d.values() if isinstance(v, (str, int, float, bool))]
            vals += [str(a) for a in d.get("aliases") or []]
            vals += [str(t) for t in d.get("tags") or []]
            for k, v in (d.get("attributes") or {}).items():
                vals.append(f"{k}:{v}")
            return vals

        hits = [d for d in self.docs if any(q in s.lower() for s in _as_strings(d))]
        return hits[:topk]

    def all(self) -> list[dict]:
        return list(self.docs)