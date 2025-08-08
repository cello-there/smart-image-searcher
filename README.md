# Smart Image Searcher

Semantic search over your local photos using CLIP embeddings + FAISS, with a tiny RAG layer for query clarification and alias learning. Windows-friendly CLI, optional HTML gallery with thumbnails, and (experimental) image clustering.

---

## Status: what actually works

**Implemented**

* **Incremental (re)indexing** of images in `images/` into a FAISS (or numpy) vector index
* **Search** by natural language prompt (embeds the text and retrieves nearest images)
* **Auto-change detection on search**: if files were added/removed/modified since the last index, a quick incremental reindex runs before searching
* **HTML gallery** output (thumbnails in `data/thumbnails/`, click to open full image)
* **RAG-lite**: clarifying questions when queries are ambiguous, plus optional alias learning (e.g., “J” → “Jason”) with user confirmation
* **Clustering** (KMeans) of all images; optional `--move` to group files into cluster folders
* **Windows CMD-friendly commands** and outputs

**Not (yet) implemented / experimental**

* `organize` command with rule-based destination folders (preview/apply/undo)
* Delete tombstones & FAISS prune flag (current reindex rewrites metadata and rebuilds/updates vectors; practical deletes work, but explicit `--prune` isn’t shipped)
* FastAPI desktop/mobile API

---

## Requirements

```
Python 3.11+
numpy, pillow, pandas, jsonschema, python-dotenv, faiss-cpu, open-clip-torch, torch, requests, tenacity, tqdm
```

If `torch`/`open-clip-torch` fail to install on Windows, the code will fall back to deterministic stub embeddings (good for wiring tests; for real results install Torch + open-clip).

Install:

```bash
pip install -r requirements.txt
```

---

## Quick start

1. Put some images under `images/` (subfolders OK). Supported: `.jpg .jpeg .png .webp .bmp`.

2. Configure `config.json` (sample):

```json
{
  "device": "auto",
  "clip_model": "ViT-B-32",
  "index_path": "data/index.faiss",
  "metadata_csv": "data/image_index.csv",
  "memory_path": "data/rag_memory.jsonl",
  "memory_index_path": "data/memory.faiss",
  "images_root": "images",
  "topk": 12,
  "distance": "cosine",
  "thumbnail_max_px": 512,
  "enable_clarifier": true,
  "enable_rerank": false,
  "enable_memory_write": true,
  "llm_provider": "ollama",
  "llm_model": "mistral",
  "ollama_host": "http://localhost:11434"
}
```

3. **Full reindex** (first time):

```cmd
python main.py reindex --full
```

4. **Search** (with HTML gallery):

```cmd
python main.py search "photos of cats" --topk 24 --gallery
```

* Top-N will be printed with scores
* A gallery opens at `data/search_results.html`
* Thumbnails are generated under `data/thumbnails/` (reused on subsequent runs)

> Tip: When you add/remove images later, just run `search`—the tool auto-detects changes and reindexes incrementally. If things look off, force a rebuild:

```cmd
DEL /Q data\index.faiss data\index.faiss.npy data\index.faiss.ids 2>nul
python main.py reindex --full
```

---

## CLI Reference (Windows CMD examples)

### Status

```cmd
python main.py status
```

Shows whether the FAISS index, metadata CSV, memory file, and memory index exist.

### Reindex

```cmd
python main.py reindex            & rem incremental (new/changed only)
python main.py reindex --full     & rem rebuild from scratch
```

* Writes/updates: `data/index.faiss` (+ `*.ids`/`*.npy` sidecars when FAISS not available) and `data/image_index.csv`

### Search

```cmd
python main.py search "yellow mustard bottle on white background" --topk 50 --gallery
python main.py search "photos of J" --show 3 --gallery
```

* On first ambiguous queries, you may be asked up to 2 clarifying questions
* If the model proposes an alias link (e.g., “J” → “Jason”), you’ll be asked to confirm before it’s saved

### Cluster (KMeans)

```cmd
python main.py cluster --k 8 --dry-run
python main.py cluster --k 8 --move --out clusters
```

* `--dry-run` prints the cluster assignments
* With `--move`, files are copied/moved into `images/<out>/<cluster-id>/...` (confirm behavior in your local code)
* Searching continues to work after moving because searches embed current paths; if you move externally, run a quick `reindex`

> **Note:** The `organize` command (rules-driven folder moves) is not shipped yet; clustering is the current grouping tool.

---

## How it works (high-level)

* **Embeddings**: Images and text are embedded by CLIP (OpenCLIP). Cosine similarity (IP on normalized vectors) ranks images.
* **FAISS index**: Stored at `data/index.faiss` with sidecar files for ids. If FAISS isn’t available, a NumPy index is used.
* **Metadata CSV**: `data/image_index.csv` maps FAISS ids → image paths and stores file signatures (size & mtime) for incremental updates.
* **Auto-change detection**: `search` checks for added/removed/modified files and calls incremental reindex before querying.
* **RAG-lite**: A memory file (`data/rag_memory.jsonl`) stores entity/alias/context tidbits learned via clarifier answers. A tiny text FAISS (or NumPy) index helps surface relevant memory when expanding queries.
* **Gallery & thumbnails**: `data/search_results.html` links to file:// URIs. Thumbs are generated once and reused; you can change max size via `thumbnail_max_px`.

---

## Troubleshooting

* **Gallery shows broken images**: make sure the paths in `data/image_index.csv` exist; if you moved files outside the tool, run `reindex --full`.
* **Only one image shows up for everything**: delete `data/index.faiss*` and rebuild with `reindex --full` (usually stale/misaligned vectors).
* **Faiss install on Windows**: use `pip install faiss-cpu`. GPU FAISS on Windows is tricky; CPU is fine for local collections.
* **Torch/OpenCLIP install troubles**: you can temporarily comment them in `requirements.txt` and run with stub embeddings to validate the wiring.

---

## Roadmap (“next” branch)

Create and work on a `next` branch for these:

```cmd
git checkout -b next
```

**Sprint 1**

* `organize` command (preview/apply/undo) with a simple rule file, e.g.:

  * `dest: "{year}/{month}/{subject|Scenes}/{filename}"`
  * Subject from memory entity → `People/{name}` or `Pets/{name}`; else a simple scene label; else `Misc`
* Move log for undo: CSV with `src,dst,hash,timestamp`
* Duplicates: skip/rename strategies, basic file hash (xxhash/sha1)

**Sprint 2**

* Delete handling polish: tombstone missing rows and `reindex --prune` to drop them from FAISS cleanly when over a threshold
* Minimal FastAPI with endpoints:

  * `GET /search?q=...`
  * `POST /reindex` (full/incremental)
  * `POST /cluster/preview`, `POST /cluster/apply`

**Sprint 3**

* Desktop UI shell (Tauri/Electron): Search grid, Library stats, Organize preview
* “Watch folders” background task to auto reindex on add/remove
* Tagging pane; optional zero-shot CLIP tags

Open a PR when ready:

```cmd
git add -A
git commit -m "README: mark implemented features; add cluster & gallery; plan next"
git push -u origin next
```

---

## Credits

* OpenCLIP / FAISS teams
* Everyone who posted their weird cat pics for testing 🐈‍⬛
