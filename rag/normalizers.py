# rag/normalizers.py
from __future__ import annotations
from typing import Optional, Tuple, List, Set, Dict
import re

# ---------- sentence-ish detection (for filtering goofy alias answers) ----------

_VERBISH = re.compile(
    r"\b(is|are|was|were|am|be|been|being|has|have|had|does|do|did|can|could|will|would|should|seems|looks)\b",
    re.IGNORECASE,
)

_PRONOUN_IS = re.compile(
    r"\b(he|she|it|they|this|that|there)\b\s+(?:is|are|was|were|\'s)\b",
    re.IGNORECASE,
)

def is_sentenceish(s: str) -> bool:
    if not s:
        return False
    # crude but effective: contains a verb-like ' is/are/was/were ' or punctuation of a sentence
    if re.search(r"\b(is|are|was|were|'s)\b", s, re.IGNORECASE):
        return True
    if re.search(r"[.!?]{1,}\s*$", s):
        return True
    # too many words looks sentence-ish
    if len(s.split()) > 6:
        return True
    return False

# ---------- kind/category normalization ----------

def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = re.sub(r"\s+", " ", str(s).strip().lower())
    return s2 or None

def normalize_kind_category(
    kind: Optional[str],
    category: Optional[str],
    cfg: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generic, config-driven normalization:
      - lowercase & trim
      - map kind via cfg["kind_synonyms"] (supports canonical->variants OR variant->canonical)
      - if category missing or not allowed, infer from cfg["category_hints"] (cat -> [kinds])
    Returns (kind, category).
    """
    k = _norm(kind)
    c = _norm(category)

    cfg = cfg or {}
    allowed: Set[str] = set(cfg.get(
        "allowed_categories",
        ["pet", "person", "food", "object", "place", "event", "trip", "thing"]
    ))

    # Build variant->canonical map from either mapping style
    syn = cfg.get("kind_synonyms") or {}
    variant_to_canon: Dict[str, str] = {}
    for left, right in syn.items():
        left_n = _norm(left)
        if left_n is None:
            continue
        if isinstance(right, (list, tuple, set)):
            canon = left_n
            for v in right:
                v_n = _norm(v)
                if v_n:
                    variant_to_canon[v_n] = canon
        else:
            # already variant -> canonical
            v_n = left_n
            canon = _norm(right)
            if canon:
                variant_to_canon[v_n] = canon

    if k and k in variant_to_canon:
        k = variant_to_canon[k]

    # Infer category if missing or not allowed
    if not c or c not in allowed:
        hints = cfg.get("category_hints") or {}
        # normalize hint keys
        norm_hints = {cat: { _norm(x) for x in xs } for cat, xs in hints.items()}
        if k:
            for cat, keys in norm_hints.items():
                if k in keys:
                    c = cat
                    break

    return (k or None), (c or None)

# ---------- alias cleaning ----------

_GENERIC_ALIAS_STOPS = {
    "photo","photos","picture","pictures","image","images","pic","pics",
    "the","a","an","my","our","your","their","his","her","its","and",
}

def clean_aliases(
    ans: str | list[str],
    *,
    max_aliases: int = 5,
    stopwords: Optional[Set[str]] = None,
    allow_single_char: bool = True,
) -> List[str]:
    """
    - Accepts str or list[str].
    - Splits on commas/slashes/pipes/semicolons and 'and'.
    - Allows one-letter aliases (e.g., 'J') if they’re alphabetic.
    - Filters sentence-ish or long phrases.
    - De-dupes case-insensitively.
    """
    base_stops = {
        "photo","photos","picture","pictures","image","images","pic","pics",
        "the","a","an","my","our","your","their","his","her","its","and",
    }
    if stopwords:
        base_stops |= {s.lower() for s in stopwords}

    parts: list[str] = []
    if isinstance(ans, list):
        parts = [str(x) for x in ans if isinstance(x, str)]
        text = ", ".join(parts)
    else:
        text = str(ans or "")

    chunks = re.split(r"(?:,|/|\||;|\band\b)", text, flags=re.IGNORECASE)

    cleaned, seen = [], set()
    for raw in chunks:
        tok = re.sub(r"[^A-Za-z0-9 _\-']", "", raw).strip()
        tok = re.sub(r"\s{2,}", " ", tok)
        if not tok:
            continue

        if is_sentenceish(tok):
            continue

        low = tok.lower()

        # length filter — allow one-letter alphabetic aliases if enabled
        if len(low) < 2:
            if not (allow_single_char and len(low) == 1 and low.isalpha()):
                continue

        if low in base_stops:
            continue
        if low.isdigit():
            continue

        if low not in seen:
            seen.add(low)
            cleaned.append(tok)

        if len(cleaned) >= max_aliases:
            break

    return cleaned
