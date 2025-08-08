from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import itertools

from .prompts import ANSWERS_TO_MEMORY_PROMPT, MEMORY_DOCS_SCHEMA
from .normalizers import normalize_kind_category
from utils import llm

_TRIVIAL_NEGATIONS = {"", "no", "none", "n/a", "na", "null", "nil"}

def _norm(s):
    return str(s).strip().lower() if s is not None else ""

def _name_set(d: dict) -> set[str]:
    names = set()
    if d.get("name"):
        names.add(_norm(d["name"]))
    for a in (d.get("aliases") or []):
        if a:
            names.add(_norm(a))
    return names

def _merge_entity_docs(base: dict, inc: dict) -> dict:
    out = dict(base)
    # if the incoming "name" is new, add as alias
    if inc.get("name") and _norm(inc["name"]) not in _name_set(base):
        out["aliases"] = sorted(set((out.get("aliases") or []) + [inc["name"]]))

    # clean & union aliases robustly
    merged_aliases = _coerce_aliases(out.get("aliases")) + _coerce_aliases(inc.get("aliases"))
    if merged_aliases:
        # de-dupe while preserving order
        seen = set()
        deduped = []
        for a in merged_aliases:
            low = a.lower()
            if low not in seen:
                seen.add(low)
                deduped.append(a)
        out["aliases"] = deduped

    # fill missing scalars
    if not out.get("kind") and inc.get("kind"):
        out["kind"] = inc["kind"]
    if not out.get("category") and inc.get("category"):
        out["category"] = inc["category"]

    # attributes: repair incoming, then merge
    attrs_base = _repair_attributes(out.get("attributes"))
    attrs_inc = _repair_attributes(inc.get("attributes"))
    attrs = dict(attrs_base)
    attrs.update(attrs_inc)  # inc wins per key
    out["attributes"] = attrs

    # tags
    if inc.get("tags"):
        out["tags"] = sorted(set((out.get("tags") or []) + list(inc["tags"])))

    # description
    if inc.get("description"):
        if not out.get("description") or len(str(inc["description"])) > len(str(out["description"])):
            out["description"] = inc["description"]

    if inc.get("persistent"):
        out["persistent"] = True
    if inc.get("user_owned") is True:
        out["user_owned"] = True
    return out

def _strip_nullish(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [], {}, "null", "None")}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _make_key(d: dict) -> str:
    t = d.get("type") or "unknown"
    cat = d.get("category") or "na"
    name = re.sub(r"[^a-z0-9]+", "-", (d.get("name", "") or "na").lower()).strip("-")
    kind = re.sub(r"[^a-z0-9]+", "-", (d.get("kind", "") or "na").lower()).strip("-")
    return f"{t}:{cat}:{name}:{kind}"

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
    if not d.get("name"):
        return False
    # allow alias-only or attributes-only updates (no hard requirement on kind)
    return bool(d.get("kind") or d.get("aliases") or d.get("attributes"))

def _normalize_docs(docs: list[dict], cfg: dict | None) -> list[dict]:
    out: list[dict] = []
    for raw in docs or []:
        d = _strip_nullish(dict(raw))
        t = d.get("type")
        if t not in {"entity", "preference", "context"}:
            continue

        if t == "context" and _looks_trivial_context(d):
            continue
        if t == "preference" and _looks_trivial_preference(d):
            continue
        if t == "entity" and not _looks_valid_entity(d):
            continue

        if t == "entity":
            d["kind"], d["category"] = normalize_kind_category(d.get("kind"), d.get("category"), cfg=cfg)
            # was: if isinstance(d.get("aliases"), (str, list)): d["aliases"] = clean_aliases(d.get("aliases"))
            d["aliases"] = _coerce_aliases(d.get("aliases"))
            d.setdefault("user_owned", True)

        d.setdefault("aliases", [])
        d.setdefault("attributes", {})
        d.setdefault("tags", [])
        d.setdefault("description", "")

        d.setdefault("persistent", t == "entity")
        d.setdefault("source", "clarifier")

        d["key"] = _make_key(d)
        now = _now_iso()
        d.setdefault("created_at", now)
        d["updated_at"] = now

        out.append(d)
    return out

def _coerce_aliases(val) -> list[str]:
    """
    Accepts str | list[str] | None and returns a cleaned alias list.
    - Splits strings by commas/and/etc.
    - Flattens lists of strings.
    - Drops sentence-ish or overly long entries.
    """
    from .normalizers import clean_aliases, is_sentenceish

    if val is None:
        return []
    if isinstance(val, list):
        # flatten & join to reuse the same cleaner
        flat = [s for s in itertools.chain.from_iterable(
            [v] if isinstance(v, str) else [] for v in val
        )]
        joined = ", ".join(flat)
        cleaned = clean_aliases(joined)
    elif isinstance(val, str):
        cleaned = clean_aliases(val)
    else:
        cleaned = []

    # filter sentence-ish leftovers just in case
    return [a for a in cleaned if not is_sentenceish(a)]

def _repair_attributes(attrs: dict | None) -> dict:
    """
    Fix attribute keys like 'color|attributes.note' or 'attributes.color'.
    Keep the last meaningful segment; if ambiguous, store under 'note'.
    """
    if not attrs:
        return {}
    fixed = {}
    for k, v in attrs.items():
        if not k:
            continue
        key = str(k)
        # split on pipe and dots, prefer the last meaningful token
        pieces = []
        for part in key.split("|"):
            pieces.extend(part.split("."))
        pieces = [p for p in (p.strip() for p in pieces) if p and p.lower() not in {"attributes"}]
        target = pieces[-1].lower() if pieces else "note"
        # if we already have a proper short target like 'color', keep it
        if target in {"color", "breed", "age", "size", "pattern"}:
            fixed[target] = v
        else:
            # default to note for weird keys
            # if 'color' mentioned in the key, try to map to color
            if "color" in key.lower():
                fixed["color"] = v
            else:
                fixed.setdefault("note", v)
    return fixed

def _repair_doc_inplace(d: dict, cfg: dict | None):
    """Repair a single memory doc in-place (aliases, attributes, kind/category)."""
    if d.get("type") == "entity":
        # normalize kind/category
        from .normalizers import normalize_kind_category
        k_norm, c_norm = normalize_kind_category(d.get("kind"), d.get("category"), cfg=cfg)
        if k_norm:
            d["kind"] = k_norm
        if c_norm:
            d["category"] = c_norm

        # aliases may be str | list | junk
        d["aliases"] = _coerce_aliases(d.get("aliases"))

        # attributes may have bad keys
        d["attributes"] = _repair_attributes(d.get("attributes"))

    # ensure always-present optional fields
    d.setdefault("aliases", [])
    d.setdefault("attributes", {})
    d.setdefault("tags", [])
    d.setdefault("description", "")


# --- robust alias extraction from clarifier free-text answers ---

_ALIAS_PATTERNS = [
    # "J is the nickname of Jason" / "J is a nickname for Jason" / "J is an alias of Jason"
    re.compile(
        r"(?P<alias>[A-Za-z0-9\-']+)\s+is\s+(?:an?\s+)?(?:nickname|alias)\s+(?:of|for)\s+(?P<entity>[A-Za-z][A-Za-z0-9\-' ]+)",
        re.I,
    ),
    # "nickname of Jason is J" / "alias of Jason is J"
    re.compile(
        r"(?:nickname|alias)\s+of\s+(?P<entity>[A-Za-z][A-Za-z0-9\-' ]+)\s+is\s+(?P<alias>[A-Za-z0-9\-']+)",
        re.I,
    ),
    # "Jason (nicknamed J)" / "Jason, nicknamed J"
    re.compile(
        r"(?P<entity>[A-Za-z][A-Za-z0-9\-' ]+?)[^A-Za-z0-9]+nick(?:name|named)\s+(?P<alias>[A-Za-z0-9\-']+)",
        re.I,
    ),
    # "alias for Jason: J" / "Jason alias: J"
    re.compile(
        r"(?:(?:alias|nickname)\s*(?:for)?\s*(?P<entity>[A-Za-z][A-Za-z0-9\-' ]+)\s*[:\-]\s*(?P<alias>[A-Za-z0-9\-']+))",
        re.I,
    ),
]

_STOPWORDS = {"the","a","an","of","for","and","or","to","is","are","am"}
_GENERIC_ENTITY_PREFIX = re.compile(
    r"^(?:the|a|an|my|our|their|his|her|its)\s+",
    re.I,
)
_GENERIC_ENTITY_CLASS = re.compile(
    r"^(?:cat|dog|person|pet|animal|human|bread|loaf(?:\s+of\s+bread)?)(?:\s+(?:named|called))?\s+",
    re.I,
)

def _sanitize_alias_token(s: str) -> str | None:
    if not s:
        return None
    s = s.strip().strip("()[]{}.,:;\"' ")
    s = re.sub(r"[^A-Za-z0-9\-']+", "", s)
    if not s or s.lower() in _STOPWORDS:
        return None
    if len(s) > 32:
        return None
    # allow single-token aliases (J, Jas) or hyphenated
    if " " in s:
        return None
    return s

def _sanitize_entity_phrase(s: str, known_names: list[str]) -> str | None:
    if not s:
        return None
    s = s.strip().strip("()[]{}.,:;\"' ")
    s = _GENERIC_ENTITY_PREFIX.sub("", s)
    s = _GENERIC_ENTITY_CLASS.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return None

    # Prefer a known entity if similar
    try:
        import difflib
        cand_list = [s]
        # also try the last capitalized token (e.g., "the cat Jason" -> "Jason")
        last_tok = s.split()[-1]
        if last_tok and last_tok[0].isalpha():
            cand_list.append(last_tok)
        best = None
        best_score = 0.0
        for cand in cand_list:
            for kn in known_names:
                score = difflib.SequenceMatcher(a=cand.lower(), b=kn.lower()).ratio()
                if score > best_score:
                    best_score, best = score, kn
        if best and best_score >= 0.75:
            return best
    except Exception:
        pass

    # fall back: if phrase ends with a Capitalized token, keep that
    toks = s.split()
    if toks and toks[-1][0].isalpha():
        # choose last token if it looks like a proper name
        return toks[-1]
    return s

def _extract_alias_links_from_answers(answers: dict, known_names: list[str]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    if not answers:
        return links
    for v in answers.values():
        s = str(v or "").strip()
        if not s:
            continue
        for pat in _ALIAS_PATTERNS:
            m = pat.search(s)
            if not m:
                continue
            raw_alias = (m.group("alias") or "").strip()
            raw_entity = (m.group("entity") or "").strip()

            alias = _sanitize_alias_token(raw_alias)
            entity = _sanitize_entity_phrase(raw_entity, known_names)

            if alias and entity and alias.lower() not in _STOPWORDS and entity.lower() not in _STOPWORDS:
                links.append((entity, alias))
    return links


class MemoryStore:
    def __init__(self, path: str, cfg: dict | None = None):
        self.path = Path(path)
        self.cfg = cfg or {}
        self.docs: list[dict] = []

    def load(self) -> "MemoryStore":
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                raw = [json.loads(line) for line in f if line.strip()]
            # repair in place
            for d in raw:
                _repair_doc_inplace(d, self.cfg)
            self.docs = raw
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _merge_upsert(self, doc: dict) -> None:
        t = doc.get("type")
        if t == "entity":
            inc_names = _name_set(doc)
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
            self.docs.append(_strip_nullish(doc))
            return

        # non-entity — merge by coarse key
        def _k(d: dict):
            tt = d.get("type")
            if tt == "preference":
                return (tt, _norm(d.get("value")))
            if tt == "context":
                return (tt, _norm(d.get("description")) or _norm(d.get("value")))
            return (tt, json.dumps(d, sort_keys=True))
        key = _k(doc)
        for i, ex in enumerate(self.docs):
            if _k(ex) == key:
                merged = {**ex, **doc}
                self.docs[i] = _strip_nullish(merged)
                return
        self.docs.append(_strip_nullish(doc))

    def upsert(self, doc: dict) -> None:
        """
        Normalize & sanitize *any* incoming doc (including those built in main.py)
        so bad aliases like 'Jason is the orange cats name' don't get stored.
        """
        d = dict(doc)

        if d.get("type") == "entity":
            d["kind"], d["category"] = normalize_kind_category(d.get("kind"), d.get("category"), cfg=self.cfg)
            # was: if isinstance(d.get("aliases"), (str, list)): d["aliases"] = clean_aliases(d.get("aliases"))
            d["aliases"] = _coerce_aliases(d.get("aliases"))
            d.setdefault("user_owned", True)
            d.setdefault("persistent", True)
        
        _repair_doc_inplace(d, self.cfg)
        self._merge_upsert(d)

    def upsert_from_answers(self, answers: dict):
        # Build known names/aliases list for disambiguation
        known_names = []
        for d in self.docs:
            if d.get("type") == "entity":
                if d.get("name"):
                    known_names.append(str(d["name"]))
                for a in (d.get("aliases") or []):
                    if a:
                        known_names.append(str(a))

        # 1) Heuristically capture alias links directly (robust parser)
        for entity, alias in _extract_alias_links_from_answers(answers, known_names):
            self.upsert({
                "type": "entity",
                "name": entity,
                "aliases": [alias],
                "persistent": True,
                "source": "alias-from-answers",
            })

        # 2) Ask the LLM for additional structured docs
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

        docs = (data or {}).get("docs", [])
        cleaned = []
        for d in docs or []:
            if d.get("type") not in {"entity", "preference", "context"}:
                continue
            if d.get("type") == "entity":
                k, c = normalize_kind_category(d.get("kind"), d.get("category"), cfg=self.cfg)
                if k:
                    d["kind"] = k
                if c:
                    d["category"] = c
            _repair_doc_inplace(d, self.cfg)
            # allow alias-only updates
            if d.get("type") == "entity" and not (d.get("name") and (d.get("kind") or d.get("aliases") or d.get("attributes"))):
                continue
            cleaned.append(d)

        for d in cleaned:
            if d.get("persistent", False):
                self._merge_upsert(d)

    def search(self, query: str, topk: int = 5) -> list[dict]:
        q = query.lower()
        def _as_strings(d: dict):
            vals = [str(v) for v in d.values() if isinstance(v, (str, int, float, bool))]
            vals += [str(a) for a in (d.get("aliases") or [])]
            vals += [str(t) for t in (d.get("tags") or [])]
            for k, v in (d.get("attributes") or {}).items():
                vals.append(f"{k}:{v}")
            return vals
        hits = [d for d in self.docs if any(q in s.lower() for s in _as_strings(d))]
        return hits[:topk]

    def all(self) -> list[dict]:
        return list(self.docs)
