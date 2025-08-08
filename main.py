import argparse
import os
import time
import webbrowser
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


def cmd_reindex(cfg, full=False):
    ensure_dir(Path(cfg["index_path"]).parent)
    ensure_dir(Path(cfg["metadata_csv"]).parent)

    from search.embedder import ImageEmbedder
    img_emb = ImageEmbedder(cfg)
    incremental_reindex(cfg, img_emb, full=full)


def _build_memory_index(cfg, text_emb, mem_store: MemoryStore) -> MemoryVectorIndex:
    mpath = cfg.get("memory_index_path", "data/memory.faiss")
    mem_idx = MemoryVectorIndex.from_docs(mem_store.all(), text_emb, mpath)
    return mem_idx


def cmd_search(cfg, query, topk=None, show=0, gallery=True):
    topk = topk or cfg.get("topk", 12)

    # Load components
    text_emb = TextEmbedder(cfg)
    mem = MemoryStore(cfg["memory_path"]).load()
    clarifier = Clarifier(cfg)
    augmenter = QueryAugmenter(cfg)

    # Build/query FAISS-backed memory index for RAG context
    mem_index = _build_memory_index(cfg, text_emb, mem)
    raw_qvec = text_emb.encode_text(query)
    mem_ctx = [d for d, _score in mem_index.search(raw_qvec, topk=5)]  # top memory docs

    # Clarify if ambiguous, using retrieved memory as context
    if cfg.get("enable_clarifier", True) and clarifier.is_ambiguous(query, {"memory": mem_ctx}):
        qs = clarifier.questions(query, {"memory": mem_ctx})[:2]
        if qs:
            print("Clarifying questions:")
            answers = {}
            for q in qs:
                answers[q] = input(f"- {q} ")
            query = clarifier.incorporate_answers(query, answers)
            if cfg.get("enable_memory_write", True):
                mem.upsert_from_answers(answers)
                mem.save()
                # Rebuild memory index after new facts are saved
                mem_index = _build_memory_index(cfg, text_emb, mem)
                mem_ctx = [d for d, _ in mem_index.search(text_emb.encode_text(query), topk=5)]

    # Augment with FAISS-retrieved memory docs
    ctx = {"memory": mem_ctx}
    aug = augmenter.expand(query, ctx)
    final_query = aug.get("expanded_query", query)

    # Embed & retrieve images
    qvec = text_emb.encode_text(final_query)
    index = VectorIndex.from_config(cfg)
    retriever = Retriever(index, cfg["metadata_csv"], distance=cfg.get("distance", "cosine"))
    results = retriever.search(qvec, topk=topk)

    print("Top results:")
    for r in results:
        print(f"{r['score']:.4f}	{r['path']}")

    # Flexible presentation: open N images and/or generate an HTML gallery
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
    # default True; user can disable with --no-gallery
    p_s.add_argument("--no-gallery", dest="gallery", action="store_false",
                    help="Do not open an HTML gallery after search")
    p_s.set_defaults(gallery=True)

    args = parser.parse_args()
    cfg = load_config("config.json")

    if args.cmd == "status":
        cmd_status(cfg)
    elif args.cmd == "reindex":
        cmd_reindex(cfg, full=args.full)
    elif args.cmd == "search":
        cmd_search(cfg, args.query, topk=args.topk, show=args.show, gallery=args.gallery)


if __name__ == "__main__":
    main()