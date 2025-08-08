# rag/query_utils.py
import re
from typing import List


def strip_unmentioned_entities(original_query: str, expanded_query: str, memory_docs: list[dict]) -> str:
    """
    Remove any entity surface forms (name or alias) that the user did NOT actually type.
    This is stricter than the old behavior (which kept all aliases for mentioned entities).
    """
    oq = original_query.lower()
    eq = expanded_query

    # Collect all forms from memory
    forms = []
    for d in memory_docs or []:
        if d.get("type") != "entity":
            continue
        if d.get("name"):
            forms.append(str(d["name"]))
        for a in (d.get("aliases") or []):
            if a:
                forms.append(str(a))

    # Partition forms into (typed vs not typed)
    typed = set()
    not_typed = set()
    for f in forms:
        f_low = f.lower()
        if re.search(rf"\b{re.escape(f_low)}\b", oq):
            typed.add(f)
        else:
            not_typed.add(f)

    # Strip any NOT typed forms from the expanded query
    for f in sorted(not_typed, key=lambda x: -len(x)):
        eq = re.sub(rf"\b{re.escape(f)}\b", "", eq, flags=re.IGNORECASE)

    # Cleanup: orphan connectors and multi-space
    eq = re.sub(r"\s*(?:,|\band\b|\bor\b)\s*(?=(?:,|\band\b|\bor\b|$))", " ", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\s*(?:,|\band\b|\bor\b)\s*$", "", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\s{2,}", " ", eq).strip()

    # If we stripped too hard, fall back to original query
    if len(re.findall(r"\w+", eq)) < 2:
        return original_query
    return eq

def split_or_alternatives(q: str) -> List[str]:
    """Split a query on standalone 'or' tokens and return trimmed non-empty parts."""
    parts = [p.strip() for p in re.split(r"\bor\b", q, flags=re.IGNORECASE) if p and p.strip()]
    cleaned = []
    for p in parts:
        if len(re.findall(r"\w+", p)) >= 2:
            cleaned.append(re.sub(r"\s{2,}", " ", p))
    return cleaned or [q]


def merge_results(result_lists: list[list[dict]], topk: int) -> list[dict]:
    """Merge multiple result lists by path, keeping the best score per image."""
    best: dict[str, dict] = {}
    for results in result_lists:
        for r in results:
            path = r.get("path") or r.get("id")
            if path is None:
                continue
            if path not in best or r["score"] > best[path]["score"]:
                best[path] = r
    merged = list(best.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:topk]
