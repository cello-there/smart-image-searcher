import argparse
import json
import re
from pathlib import Path
from typing import List

from utils.logging import get_logger
from utils.io import ensure_dir
from utils.validation import load_config

from search.reindex import incremental_reindex
from search.embedder import TextEmbedder
from search.indexer import VectorIndex
from search.retriever import Retriever

from rag.clarifier import Clarifier
from rag.augmenter import QueryAugmenter
from rag.memory import MemoryStore
from rag.memory_index import MemoryVectorIndex
from rag.normalizers import clean_aliases
from utils.manifest import scan_image_dir, load_manifest, save_manifest, diff_manifests

from rag.query_utils import (
    strip_unmentioned_entities,
    split_or_alternatives,
    merge_results,
)

logger = get_logger(__name__)

# --- Generic detection helpers ---
_GENERIC_ENTITY_TERMS = {
    "cat","cats","dog","dogs","pet","pets","animal","animals","person","people",
    "kid","kids","child","children","man","men","woman","women","guy","guys",
    "photos","photo","pictures","picture","images","image","someone","something",
    "thing","things","object","objects","place","places"
}
_ANYLIKE = {"any", "anything", "whatever", "dont care", "don't care", "na", "n/a", "none", "no preference"}

def _manifest_path(cfg): 
    return cfg.get("manifest_path", "data/image_manifest.json")

def _image_dir(cfg):
    return cfg.get("image_dir", "images")

def _auto_reindex_if_changed(cfg) -> bool:
    """Scan images/, and if anything changed, run incremental reindex + update manifest."""
    man_path = _manifest_path(cfg)
    old = load_manifest(man_path)
    new = scan_image_dir(_image_dir(cfg))
    added, removed, modified = diff_manifests(old, new)
    if added or removed or modified:
        logger.info(f"[watch] changes detected: +{len(added)} -{len(removed)} ~{len(modified)} — reindexing…")
        ensure_dir(Path(cfg["index_path"]).parent)
        ensure_dir(Path(cfg["metadata_csv"]).parent)
        from search.embedder import ImageEmbedder
        img_emb = ImageEmbedder(cfg)
        incremental_reindex(cfg, img_emb, full=False)
        save_manifest(man_path, new)
        return True
    return False

def _is_generic_entity_name(name: str) -> bool:
    if not name:
        return True
    low = name.strip().lower()
    # single common noun or super generic plural
    if low in _GENERIC_ENTITY_TERMS:
        return True
    # drop obvious plurals of generic things (e.g., "cats", "dogs")
    if low.endswith("s") and low[:-1] in _GENERIC_ENTITY_TERMS:
        return True
    # avoid saving queries that are literally media words like "photos"
    if re.fullmatch(r"(photos?|pictures?|images?)", low):
        return True
    return False

def _is_generic_entity(e: dict, original_query: str) -> bool:
    name = (e.get("name") or "").strip()
    if _is_generic_entity_name(name):
        return True
    # super weak categories from LLM like "animal|pet" or "specific"
    cat = (e.get("category") or "").strip().lower()
    kind = (e.get("kind") or "").strip().lower() if e.get("kind") else ""
    if "|" in cat or cat in {"generic","specific"}:
        return True
    if kind in {"generic","specific"}:
        return True
    # If the original query is clearly generic (no proper name, plural common noun), skip memory prompts.
    tokens = re.findall(r"[a-z]+", original_query.lower())
    has_proper_name = any(t[0].isupper() for t in original_query.split() if t.isalpha())
    if not has_proper_name and any(t in _GENERIC_ENTITY_TERMS for t in tokens):
        return True
    return False

def cmd_status(cfg):
    index = Path(cfg["index_path"]).exists()
    meta = Path(cfg["metadata_csv"]).exists()
    mem = Path(cfg["memory_path"]).exists()
    mem_idx = Path(cfg.get("memory_index_path", "data/memory.faiss")).exists()
    logger.info({
        "index_present": index,
        "metadata_present": meta,
        "memory_present": mem,
        "memory_index_present": mem_idx,
    })

def cmd_reindex(cfg, full: bool = False):
    ensure_dir(Path(cfg["index_path"]).parent)
    ensure_dir(Path(cfg["metadata_csv"]).parent)
    from search.embedder import ImageEmbedder
    img_emb = ImageEmbedder(cfg)
    incremental_reindex(cfg, img_emb, full=full)
    # NEW: refresh manifest after any reindex
    save_manifest(_manifest_path(cfg), scan_image_dir(_image_dir(cfg)))


def _build_memory_index(cfg, text_emb, mem_store: MemoryStore) -> MemoryVectorIndex:
    mpath = cfg.get("memory_index_path", "data/memory.faiss")
    return MemoryVectorIndex.from_docs(mem_store.all(), text_emb, mpath)


def _filter_memory(mem_hits, query: str, min_sim: float = 0.30, max_docs: int = 5,
                   debug: bool = False, logger=None) -> list[dict]:
    """
    Keep relevant memory docs.

    Entities:
      - Keep if the query explicitly mentions the entity by NAME or ALIAS.
      - If the query has a possessive pronoun (my/our/their/his/her/its) but NO names,
        allow pronoun to pass (helps queries like "my cat" when there's only one candidate).
      - Do NOT match on description/kind/category to avoid false positives.

    Non-entities:
      - Keep if similarity >= min_sim (and drop trivial 'no/none' preferences).
    """
    import re

    STOPWORDS = {
        "a","an","the","of","in","on","at","to","for","and","or","but","with","by","from","as","is","are","was","were",
        "it","this","that","these","those","any","some","all","about","into","over","under","up","down"
    }
    pronouns = {"my", "our", "their", "his", "her", "its"}
    valid_types = {"entity", "preference", "context"}

    toks = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
    q_tokens = set(toks)
    pronoun_present = len(q_tokens & pronouns) > 0

    # detect if ANY entity name/alias is explicitly in the query
    any_name_mentioned = False
    entity_name_sets = []
    for d, _ in mem_hits:
        if d.get("type") != "entity":
            entity_name_sets.append(set())
            continue
        names = set()
        if d.get("name"):
            names |= {w for w in re.findall(r"\w+", d["name"].lower()) if w not in STOPWORDS}
        for a in (d.get("aliases") or []):
            names |= {w for w in re.findall(r"\w+", str(a).lower()) if w not in STOPWORDS}
        entity_name_sets.append(names)
        if q_tokens & names:
            any_name_mentioned = True

    out = []
    for (d, s), name_tokens in zip(mem_hits, entity_name_sets):
        t = d.get("type")
        if t not in valid_types:
            if debug and logger: logger.info(f"[mem_filter] drop malformed doc: {d}")
            continue

        if t == "entity":
            if q_tokens & name_tokens:
                out.append(d)
                if debug and logger: logger.info(f"[mem_filter] keep entity (named): {d.get('name')}")
                continue
            if pronoun_present and not any_name_mentioned:
                out.append(d)
                if debug and logger: logger.info(f"[mem_filter] keep entity (pronoun, no names in query): {d.get('name')}")
            else:
                if debug and logger: logger.info(f"[mem_filter] drop entity (not referenced): {d.get('name')}")
            continue

        if s >= min_sim:
            if t == "preference" and str(d.get("value", "")).strip().lower() in {"", "no", "none"}:
                if debug and logger: logger.info(f"[mem_filter] drop trivial preference: {d}")
                continue
            out.append(d)
            if debug and logger: logger.info(f"[mem_filter] keep {t} (sim {s:.2f} >= {min_sim:.2f})")
        else:
            if debug and logger: logger.info(f"[mem_filter] drop {t} (sim {s:.2f} < {min_sim:.2f})")

    return out[:max_docs]


def _maybe_learn_alias(query: str, mem_ctx: list[dict], mem: MemoryStore, clarifier: Clarifier,
                       text_emb, cfg: dict, debug: bool = False) -> list[dict]:
    import difflib

    alias_min_conf = float(cfg.get("alias_min_conf", 0.65))

    # Use ALL entities to consider possible links (not just filtered context)
    entities = [d for d in mem.docs if d.get("type") == "entity" and d.get("name")]
    if not entities:
        return mem_ctx

    # Build canonical forms per entity (lowercased)
    forms_by_ent: dict[str, set[str]] = {}
    all_known_forms = set()
    for d in entities:
        forms = [str(d.get("name", "")).strip()]
        forms += [str(a).strip() for a in (d.get("aliases") or [])]
        forms = [f for f in forms if f]
        lower = {f.lower() for f in forms}
        forms_by_ent[d["name"]] = lower
        all_known_forms |= lower

    # Tokens seen in user query
    q_lower = query.lower()
    q_tokens = set(re.findall(r"[a-z0-9']+", q_lower))

    # LLM proposal (optional)
    links = clarifier.propose_alias(query, entities)
    if debug:
        logger.info(f"[alias.links] {json.dumps(links, ensure_ascii=False)}")

    def _valid_link(ent: str, alias: str, conf: float) -> bool:
        if not ent or not alias:
            return False
        if conf < alias_min_conf:
            return False
        a = alias.strip().lower()
        # alias must appear in the query (token or substring)
        if a not in q_tokens and a not in q_lower:
            return False
        # must point to an existing entity
        if ent not in forms_by_ent:
            return False
        # cannot be identical to any known form for that entity
        if a in forms_by_ent[ent]:
            return False
        # avoid 1-char or very long nonsense
        if not (2 <= len(a) <= 40):
            return False
        return True

    filtered = []
    for link in links or []:
        ent = (link.get("entity") or "").strip()
        alias = (link.get("alias") or "").strip()
        conf = float(link.get("confidence") or 0.0)
        if _valid_link(ent, alias, conf):
            filtered.append({"entity": ent, "alias": alias, "confidence": conf})

    # Fallback: fuzzy local guess (only if nothing passed filters)
    if not filtered:
        novel_tokens = [t for t in q_tokens if t not in all_known_forms]
        best = None
        best_ratio = 0.0
        best_ent = None
        for tok in novel_tokens:
            for ent, forms in forms_by_ent.items():
                for f in forms:
                    r = difflib.SequenceMatcher(a=tok, b=f).ratio()
                    if r > best_ratio:
                        best_ratio, best, best_ent = r, tok, ent
        min_ratio = float(cfg.get("alias_min_ratio", 0.82))
        if best and best_ent and best_ratio >= min_ratio:
            filtered.append({"entity": best_ent, "alias": best, "confidence": best_ratio})

    if not filtered:
        return mem_ctx

    changed = False
    for link in filtered:
        ent = link["entity"]
        alias = link["alias"]
        yn = input(f"Is '{alias}' a nickname/alias for '{ent}'? (y/N) ").strip().lower()
        if not yn.startswith("y"):
            continue

        mem.upsert({
            "type": "entity",
            "name": ent,
            "aliases": [alias],
            "persistent": True,
            "source": "alias-confirm",
        })
        changed = True

    if changed:
        mem.save()
        mem_index = _build_memory_index(cfg, text_emb, mem)
        raw_qvec = text_emb.encode_text(query)
        mem_hits = mem_index.search(raw_qvec, topk=8)
        min_sim = float(cfg.get("memory_min_sim", 0.20))
        mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
        if debug:
            logger.info(f"[mem_ctx after alias] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")
    return mem_ctx

def _resolve_alias_target(
    alias: str,
    mem: MemoryStore,
    clarifier: Clarifier,
    query: str,
    cfg: dict,
    debug: bool = False,
    mem_ctx: list[dict] | None = None,
) -> str | None:
    """
    Map `alias` to an existing entity name.

    Rules:
    - If alias is a single character and exactly one entity name starts with it, prefer that.
    - Otherwise score candidates with simple heuristics + (optionally) LLM proposals.
    - Return canonical entity name or None if not confident.
    """
    alias_raw = (alias or "").strip()
    alias_l = alias_raw.lower()
    if not alias_l:
        return None

    ents = [d for d in mem.docs if d.get("type") == "entity" and d.get("name")]
    if not ents:
        return None

    # ---- Heuristic 1: single-letter initials win if unique ----
    if len(alias_l) == 1:
        starts = [d["name"] for d in ents if isinstance(d.get("name"), str) and d["name"][:1].lower() == alias_l]
        if len(starts) == 1:
            if debug: logger.info(f"[alias.resolve] single-letter initial unique -> {starts[0]}")
            return starts[0]
        # if not unique, we won't trust LLM blindly; we’ll keep scoring below

    # Optional: grab LLM proposals (but don't trust blindly)
    llm_links = clarifier.propose_alias(query, ents) or []
    if debug:
        logger.info(f"[alias.resolve.llm_links] {llm_links}")

    llm_conf: dict[str, float] = {}
    for link in llm_links:
        ent = (link.get("entity") or "").strip()
        al  = (link.get("alias") or "").strip().lower()
        conf = float(link.get("confidence") or 0.0)
        if al == alias_l and ent:
            # keep max conf per ent
            llm_conf[ent] = max(llm_conf.get(ent, 0.0), conf)

    # Prep scoring
    q_l = query.lower()
    mem_ctx_names = {d.get("name") for d in (mem_ctx or []) if d.get("type") == "entity"}

    def score_entity(name: str, doc: dict) -> float:
        s = 0.0
        n_l = name.lower()
        # Strong signals
        if n_l.startswith(alias_l): s += 3.0
        if alias_l in n_l:          s += 1.0
        # If alias already in aliases (rare in this flow) – de-prioritize (means nothing to add)
        aliases = {str(a).lower() for a in (doc.get("aliases") or []) if a}
        if alias_l in aliases: s -= 1.0
        # Memory-context boost
        if name in mem_ctx_names: s += 0.5
        # Name mentioned in query?
        if name.lower() in q_l: s += 0.5
        # LLM nudge (weak)
        if name in llm_conf:
            # cap tiny nudge; we don't let it override initials when wrong
            s += min(llm_conf[name], 0.6)
        return s

    candidates = [(score_entity(d["name"], d), d["name"]) for d in ents]
    candidates.sort(reverse=True)  # highest score first

    if debug:
        logger.info(f"[alias.resolve.scores] {candidates[:5]}")

    best_score, best_name = candidates[0]
    # require minimum confidence; initials/path-based usually >= 2–3
    threshold = 1.5 if len(alias_l) > 1 else 2.5
    return best_name if best_score >= threshold else None


def _add_alias_for_entity(entity_name: str, alias: str, mem: MemoryStore, debug: bool = False):
    """Persist alias to existing entity."""
    mem.upsert({
        "type": "entity",
        "name": entity_name,
        "aliases": [alias],
        "persistent": True,
        "source": "alias-confirm",
        "user_owned": True,
    })
    mem.save()
    if debug:
        logger.info(f"[alias.added] {alias} -> {entity_name}")


def _maybe_learn_alias_from_answers(answers: dict, mem: MemoryStore, debug: bool = False):
    """
    Parse clarifier answers like 'J is the nickname of Jason' and add alias.
    Rule-based (no LLM) so it won't hallucinate.
    """
    if not answers:
        return

    text = " ".join(str(v) for v in answers.values() if v).strip()
    if not text:
        return

    # common patterns: "J is the nickname of Jason", "J is short for Jason", "J = Jason"
    patterns = [
        # "J is a/the nickname of/for Jason" | "J is alias of/for Jason" | "J is short for Jason" | "J is aka Jason"
        r"^\s*(?P<alias>[A-Za-z0-9'._-]{1,40})\s+(?:is|=)\s+(?:a|the\s+)?(?:nickname|alias|short\s*for|aka)\s+(?:of|for)?\s*(?P<name>[A-Za-z0-9 _'-]{2,80})\s*$",
        # "J = Jason"
        r"^\s*(?P<alias>[A-Za-z0-9'._-]{1,40})\s*=\s*(?P<name>[A-Za-z0-9 _'-]{2,80})\s*$",
        # "Jason aka J"
        r"^\s*(?P<name>[A-Za-z0-9 _'-]{2,80})\s+(?:aka|also\s+known\s+as)\s+(?P<alias>[A-Za-z0-9'._-]{1,40})\s*$",
    ]

    m = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            break
    if not m:
        return

    alias = (m.group("alias") or "").strip().strip("'\"")
    name  = (m.group("name")  or "").strip().strip("'\"")
    if not alias or not name:
        return

    # Does `name` exist as an entity?
    for d in mem.docs:
        if d.get("type") == "entity" and str(d.get("name","")).strip().lower() == name.lower():
            _add_alias_for_entity(d["name"], alias, mem, debug=debug)
            return

def cmd_search(cfg, query, topk=None, show=0, gallery=True, debug=False):
    topk = topk or cfg.get("topk", 12)
    dbg = {"raw_query": query}

    # Components
    text_emb = TextEmbedder(cfg)
    mem = MemoryStore(cfg["memory_path"]).load()
    
    if cfg.get("auto_reindex_on_search", True):
        try:
            _auto_reindex_if_changed(cfg)
        except Exception as e:
            logger.warning(f"[watch] auto-reindex check failed: {e}")


    # Auto-clean previously saved junk (e.g., sentence-like aliases)
    if hasattr(mem, "sanitize") and mem.sanitize():
        mem.save()

    clarifier = Clarifier(cfg)
    augmenter = QueryAugmenter(cfg)

    # Memory RAG (build + search + filter)
    mem_index = _build_memory_index(cfg, text_emb, mem)
    raw_qvec = text_emb.encode_text(query)
    mem_hits = mem_index.search(raw_qvec, topk=8)  # [(doc, score)]
    min_sim = float(cfg.get("memory_min_sim", 0.20))
    mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
    if debug:
        logger.info(f"[mem_ctx top8 raw] {json.dumps(mem_hits[:8], ensure_ascii=False)[:800]}")
        logger.info(f"[mem_ctx filtered] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")
    dbg["memory_ctx"] = mem_ctx

    # Learn aliases (safe, confirmed)
    mem_ctx = _maybe_learn_alias(query, mem_ctx, mem, clarifier, text_emb, cfg, debug)

    # (Keep for augmenter context if you ever need it later)
    ephemeral_ents: List[dict] = []

    # Detect new entities mentioned in the query that aren't in memory yet
    new_ents = clarifier.find_new_entities(query, mem_ctx)
    if debug:
        logger.info(f"[new_entities_from_query] {json.dumps(new_ents, ensure_ascii=False)}")

    # Drop generic/common-noun “entities” like cats/dogs/photos/images/etc.
    new_ents = [e for e in (new_ents or []) if not _is_generic_entity(e, query)]

    if cfg.get("confirm_new_entities", True) and new_ents:
        print("I think you mentioned something new:")
        for e in new_ents[:2]:
            label = e.get("name")
            cat = e.get("category")
            kind = e.get("kind")

            yn = input(f"- Should I remember '{label}'? (y/N) ").strip().lower()
            if not yn.startswith("y"):
                target = _resolve_alias_target(label, mem, clarifier, query, cfg, debug, mem_ctx=mem_ctx)

                # Never auto-confirm super-short aliases (e.g., "J")
                min_chars = int(cfg.get("alias_min_chars_for_autoconfirm", 2))
                can_autoconfirm = cfg.get("alias_autoconfirm_on_decline", False) and len(label.strip()) >= min_chars

                if target:
                    if can_autoconfirm:
                        _add_alias_for_entity(target, label, mem, debug=debug)
                    else:
                        yn2 = input(f"  · Add '{label}' as an alias for '{target}'? (y/N) ").strip().lower()
                        if yn2.startswith("y"):
                            _add_alias_for_entity(target, label, mem, debug=debug)
                # If no confident target, do nothing (don’t save junk doc)
                continue


            # User said YES -> enrich + upsert
            enrich = clarifier.enrich_entity(label, cat, kind, mem_ctx, query)
            ent = enrich.get("entity", {}) or {}
            qs = enrich.get("questions", []) or []

            ent_doc = {
                "type": "entity",
                "name": label,
                "category": ent.get("category") or cat or None,
                "kind": ent.get("kind") or kind,
                "aliases": ent.get("aliases") or [],
                "attributes": ent.get("attributes") or {},
                "tags": ent.get("tags") or [],
                "description": ent.get("description") or "",
                "user_owned": True,
                "source": "confirm",
                "persistent": True,
            }

            # Ask targeted questions (up to 3)
            for q in qs[:3]:
                qtxt = q.get("q"); key = q.get("key") or ""
                if not qtxt:
                    continue
                ans = input(f"  · {qtxt} ").strip()
                if not ans:
                    continue
                if key == "kind":
                    ent_doc["kind"] = ans
                elif key == "aliases":
                    parts = [a.strip() for a in ans.split(",") if a.strip()]
                    ent_doc["aliases"] = sorted(set((ent_doc.get("aliases") or []) + parts))
                elif key.startswith("attributes."):
                    k = key.split(".", 1)[1]
                    ent_doc.setdefault("attributes", {})[k] = ans
                elif key == "attributes":
                    ent_doc.setdefault("attributes", {})["note"] = ans

            mem.upsert(ent_doc)

        # Persist + refresh ONCE after processing all new-entity candidates
        mem.save()
        mem_index = _build_memory_index(cfg, text_emb, mem)
        mem_hits = mem_index.search(text_emb.encode_text(query), topk=8)
        mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
        if debug:
            logger.info(f"[mem_ctx after new-entity confirm] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")

    # Clarify if ambiguous (with filtered memory)
    if cfg.get("enable_clarifier", True) and clarifier.is_ambiguous(query, {"memory": mem_ctx}):
        qs = clarifier.questions(query, {"memory": mem_ctx})[:2]
        if debug:
            logger.info(f"[clarifier.questions] {qs}")
        if qs:
            print("Clarifying questions:")
            answers = {q: input(f"- {q} ") for q in qs}
            if debug:
                logger.info(f"[clarifier.answers] {answers}")

            query = clarifier.incorporate_answers(query, answers)
            if debug:
                logger.info(f"[clarifier.rewritten_query] {query}")
            dbg["clarifier"] = {"questions": qs, "answers": answers, "rewritten_query": query}

            # (NEW) Learn aliases from free-form answers like “J is a nickname of Jason”
            _maybe_learn_alias_from_answers(answers, mem, debug=debug)

            # Optional: structured memory write
            if cfg.get("enable_memory_write", True):
                mem.upsert_from_answers(answers)

            # Persist & refresh ONCE after both of the above
            mem.save()
            mem_index = _build_memory_index(cfg, text_emb, mem)
            mem_hits = mem_index.search(text_emb.encode_text(query), topk=8)
            mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
            if debug:
                logger.info(f"[mem_ctx post-write] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")

    # Augment (include filtered memory + ephemeral entities)
    ctx = {"memory": mem_ctx, "ephemeral_entities": ephemeral_ents}
    aug = augmenter.expand(query, ctx)
    final_query = aug.get("expanded_query", query)

    # Sanitize entity forms the user did not mention
    final_query_before_sanitize = final_query
    final_query = strip_unmentioned_entities(query, final_query, mem_ctx)
    if debug and final_query != final_query_before_sanitize:
        logger.info(f"[final_query sanitized] {final_query_before_sanitize}  ->  {final_query}")

    if debug:
        logger.info(f"[augmenter.json] {json.dumps(aug, ensure_ascii=False)}")
        logger.info(f"[final_query] {final_query}")
    dbg["augmenter"] = aug
    dbg["final_query"] = final_query

    # Persist debug snapshot
    if debug:
        Path("data/debug").mkdir(parents=True, exist_ok=True)
        Path("data/debug/last_query.json").write_text(
            json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Embed & retrieve — split on standalone 'or' and merge results
    alt_queries = split_or_alternatives(final_query)
    if debug and len(alt_queries) > 1:
        logger.info(f"[final_query.split_or] {alt_queries}")

    index = VectorIndex.from_config(cfg)
    retriever = Retriever(index, cfg["metadata_csv"], distance=cfg.get("distance", "cosine"))

    result_lists = []
    for aq in alt_queries:
        qvec = text_emb.encode_text(aq)
        result_lists.append(retriever.search(qvec, topk=topk))

    results = merge_results(result_lists, topk=topk)

    # Optional CLIP reranker
    if cfg.get("enable_rerank", False) and results:
        from search.embedder import ImageEmbedder
        img_emb = ImageEmbedder(cfg)
        topN = min(cfg.get("rerank_topN", 100), len(results))
        paths = [r["path"] for r in results[:topN]]
        ivecs = img_emb.encode_images(paths)         # (N, d) L2-normalized by embedder
        qvec_full = text_emb.encode_text(final_query)  # rerank against the full sanitized query
        s = ivecs @ qvec_full.astype("float32")      # cosine vs normalized qvec
        for i, score in enumerate(s.tolist()):
            results[i]["score"] = float(score)
        results[:topN] = sorted(results[:topN], key=lambda r: r["score"], reverse=True)

    print("\nTop results:")
    for r in results:
        print(f"{r['score']:.4f}\t{r['path']}")

    from ui.display import present_results
    present_results(results, cfg, show=show, gallery=gallery)



def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_re = sub.add_parser("reindex")
    p_re.add_argument("--full", action="store_true")

    p_s = sub.add_parser("search")
    p_s.add_argument("query")
    p_s.add_argument("--topk", type=int, default=None)
    p_s.add_argument("--show", type=int, default=0,
                     help="Open top N results in the default image viewer")
    p_s.add_argument("--no-gallery", dest="gallery", action="store_false",
                     help="Do not open an HTML gallery after search")
    p_s.add_argument("--debug", action="store_true",
                     help="Verbose logging + write data/debug/last_query.json")
    p_s.set_defaults(gallery=True)

    args = parser.parse_args()
    cfg = load_config("config.json")

    if args.cmd == "status":
        cmd_status(cfg)
    elif args.cmd == "reindex":
        cmd_reindex(cfg, full=args.full)
    elif args.cmd == "search":
        cmd_search(cfg, args.query, topk=args.topk,
                   show=args.show, gallery=args.gallery, debug=args.debug)


if __name__ == "__main__":
    main()
