import argparse
import json
import re
from pathlib import Path

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

from ui.display import present_results

logger = get_logger(__name__)


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

    # First pass: detect if ANY entity name/alias is explicitly in the query
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


def _strip_unmentioned_entities(original_query: str, expanded_query: str, memory_docs: list[dict]) -> str:
    import re
    oq = original_query.lower()
    eq = expanded_query

    # Build entity -> surface forms (name + aliases)
    entities = []
    for d in memory_docs or []:
        if d.get("type") != "entity":
            continue
        forms = []
        if d.get("name"):
            forms.append(str(d["name"]))
        forms += [str(a) for a in (d.get("aliases") or []) if a]
        if not forms:
            continue
        # Consider ENTIRE entity "mentioned" if ANY of its forms appear in the original query.
        # This keeps "J" / "Jason" when the user typed "jas", after alias learning.
        mentioned = any(re.search(rf"\b{re.escape(f.lower())}\b", oq) for f in forms)
        entities.append({"forms": forms, "mentioned": mentioned})

    # Remove ONLY unmentioned forms; keep all forms for mentioned entities
    for ent in entities:
        if ent["mentioned"]:
            continue
        for form in ent["forms"]:
            eq = re.sub(rf"\b{re.escape(form)}\b", "", eq, flags=re.IGNORECASE)

    # Clean up connectors and orphaned "named"
    eq = re.sub(r"\bnamed\s*(?:,|\bor\b|\band\b)?\s*$", " ", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\s*(?:,|\bor\b|\band\b)\s*(?=(?:,|\bor\b|\band\b|$))", " ", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\s*(?:,|\bor\b|\band\b)\s*$", "", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\s{2,}", " ", eq).strip()

    # Fallbacks if we were too aggressive: try dropping just "named ..." tail;
    # if still too short, fall back to the original user query.
    if len(re.findall(r"\w+", eq)) < 2:
        tmp = re.sub(r"\bnamed\b.*$", "", expanded_query, flags=re.IGNORECASE).strip()
        tmp = re.sub(r"\s{2,}", " ", tmp)
        if len(re.findall(r"\w+", tmp)) >= 2:
            return tmp
        return original_query

    return eq


def _maybe_learn_alias(query: str, mem_ctx: list[dict], mem: MemoryStore, clarifier: Clarifier,
                       text_emb, cfg: dict, debug: bool = False) -> list[dict]:
    alias_min_conf = float(cfg.get("alias_min_conf", 0.65))

    # Use entities from filtered context, else fall back to ALL known entities
    entities_ctx = [d for d in (mem_ctx if mem_ctx else mem.docs) if d.get("type") == "entity"]

    links = clarifier.propose_alias(query, entities_ctx)
    if debug:
        logger.info(f"[alias.links] {json.dumps(links, ensure_ascii=False)}")
    changed = False

    for link in links:
        ent = (link.get("entity") or "").strip()
        alias = (link.get("alias") or "").strip()
        conf = float(link.get("confidence") or 0.0)
        if not ent or not alias or conf < alias_min_conf:
            continue
        yn = input(f"Is '{alias}' a nickname/alias for '{ent}'? (y/N) ").strip().lower()
        if not yn.startswith("y"):
            continue

        # Find the target entity by name (case-insensitive); if missing, create minimal
        target = None
        for d in mem.docs:
            if d.get("type") == "entity" and str(d.get("name", "")).lower() == ent.lower():
                target = d
                break
        if not target:
            target = {"type": "entity", "name": ent, "aliases": []}

        # Upsert alias – your MemoryStore merge is alias-aware and will union
        mem.upsert({
            "type": "entity",
            "name": target.get("name"),
            "kind": target.get("kind"),
            "aliases": [alias],
            "persistent": True,
            "source": "alias-confirm",
        })
        changed = True

    if changed:
        mem.save()
        # Rebuild memory index and re-filter
        mem_index = _build_memory_index(cfg, text_emb, mem)
        raw_qvec = text_emb.encode_text(query)
        mem_hits = mem_index.search(raw_qvec, topk=8)
        min_sim = float(cfg.get("memory_min_sim", 0.20))
        mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
        if debug:
            logger.info(f"[mem_ctx after alias] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")
    return mem_ctx

def cmd_search(cfg, query, topk=None, show=0, gallery=True, debug=False):
    topk = topk or cfg.get("topk", 12)
    dbg = {"raw_query": query}

    # Components
    text_emb = TextEmbedder(cfg)
    mem = MemoryStore(cfg["memory_path"]).load()
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

    #check if new memory is adding details to a previous one
    mem_ctx = _maybe_learn_alias(query, mem_ctx, mem, clarifier, text_emb, cfg, debug)

    # --- New: collect ephemeral entities (ask questions even if user declines saving)
    ephemeral_ents = []

    # Detect new entities mentioned in the query that aren't in memory yet
    new_ents = clarifier.find_new_entities(query, mem_ctx)
    if debug:
        logger.info(f"[new_entities_from_query] {json.dumps(new_ents, ensure_ascii=False)}")

    if cfg.get("confirm_new_entities", True) and new_ents:
        print("I think you mentioned something new:")
        for e in new_ents[:2]:
            label = e.get("name")
            cat = e.get("category")
            kind = e.get("kind")
            yn = input(f"- Should I remember '{label}'? (y/N) ").strip().lower()

            # Ask the LLM for targeted questions + a draft entity record (for both Yes/No paths)
            enrich = clarifier.enrich_entity(label, cat, kind, mem_ctx, query)
            ent = enrich.get("entity", {}) or {}
            qs = enrich.get("questions", []) or []

            def _make_ent_doc(source_tag: str) -> dict:
                return {
                    "type": "entity",
                    "name": label,
                    "category": ent.get("category") or cat or "pet",
                    "kind": ent.get("kind") or kind,
                    "aliases": ent.get("aliases") or [],
                    "attributes": ent.get("attributes") or {},
                    "tags": ent.get("tags") or [],
                    "description": ent.get("description") or "",
                    "user_owned": True,
                    "source": source_tag,
                    "persistent": (source_tag == "confirm"),
                }

            ent_doc = _make_ent_doc("confirm" if yn.startswith("y") else "ephemeral")

            # Helper to set values via dotted keys (e.g., attributes.color / aliases)
            def _assign(doc: dict, key: str, value: str):
                if not value:
                    return
                if key == "kind":
                    doc["kind"] = value
                elif key == "aliases":
                    parts = [a.strip() for a in value.split(",") if a.strip()]
                    doc["aliases"] = sorted(set((doc.get("aliases") or []) + parts))
                elif key.startswith("attributes."):
                    k = key.split(".", 1)[1]
                    doc.setdefault("attributes", {})[k] = value
                elif key == "attributes":
                    if isinstance(value, str):
                        doc.setdefault("attributes", {})["note"] = value

            # Ask targeted questions (up to 3) even if user chose not to save
            for q in qs[:3]:
                qtxt = q.get("q")
                key = q.get("key") or ""
                if not qtxt:
                    continue
                ans = input(f"  · {qtxt} ").strip()
                _assign(ent_doc, key, ans)

            # If kind is still missing, ask a generic species question once
            if not ent_doc.get("kind"):
                ans = input("  · What species/kind is it? (e.g., cat, dog, bread) ").strip()
                if ans:
                    ent_doc["kind"] = ans

            if yn.startswith("y"):
                # Save to memory
                mem.upsert(ent_doc)
            else:
                # Keep for this search only
                ephemeral_ents.append(ent_doc)

        # Persist memory (if any added) and refresh memory ctx
        mem.save()
        mem_index = _build_memory_index(cfg, text_emb, mem)
        mem_hits = mem_index.search(text_emb.encode_text(query), topk=8)
        mem_ctx = _filter_memory(mem_hits, query, min_sim=min_sim, max_docs=5, debug=debug, logger=logger)
        if debug:
            logger.info(f"[mem_ctx after new-entity confirm] {json.dumps(mem_ctx, ensure_ascii=False)[:800]}")
            if ephemeral_ents:
                logger.info(f"[ephemeral_entities] {json.dumps(ephemeral_ents, ensure_ascii=False)[:800]}")

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

            if cfg.get("enable_memory_write", True):
                mem.upsert_from_answers(answers)
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

    final_query_before_sanitize = final_query
    final_query = _strip_unmentioned_entities(query, final_query, mem_ctx)
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

    # Embed & retrieve
    qvec = text_emb.encode_text(final_query)
    index = VectorIndex.from_config(cfg)
    retriever = Retriever(index, cfg["metadata_csv"], distance=cfg.get("distance", "cosine"))
    results = retriever.search(qvec, topk=topk)

    # Optional CLIP reranker
    if cfg.get("enable_rerank", False) and results:
        from search.embedder import ImageEmbedder
        img_emb = ImageEmbedder(cfg)
        topN = min(cfg.get("rerank_topN", 100), len(results))
        paths = [r["path"] for r in results[:topN]]
        ivecs = img_emb.encode_images(paths)         # (N, d) L2-normalized by embedder
        s = ivecs @ qvec.astype("float32")           # cosine vs normalized qvec
        for i, score in enumerate(s.tolist()):
            results[i]["score"] = float(score)
        results[:topN] = sorted(results[:topN], key=lambda r: r["score"], reverse=True)

    print("\nTop results:")
    for r in results:
        print(f"{r['score']:.4f}\t{r['path']}")

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