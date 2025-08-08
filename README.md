# AI Image Searcher

Local, fast image search with CLIP embeddings + FAISS, optional LLM-powered query augmentation and lightweight memory for names/aliases ("Jason" → "J"). Now with safer reindexing, deleted-file pruning, and an experimental clustering workflow for quick albuming.

---

## ✨ Features

* **Semantic search** over your `images/` folder with CLIP (OpenCLIP ViT-B/32 by default).
* **Incremental reindex** (only new/changed files embedded) **+ auto health checks** with a one-flag **full rebuild** when the index/CSV drift.
* **Deleted-file handling**: when you run a full rebuild, entries for missing files are automatically pruned from both the CSV and the FAISS index.
* **LLM-assisted queries** (optional): clarify ambiguous queries, expand to better terms.
* **Lightweight memory**: remember entities (pets, people, objects) and **aliases** ("J" → "Jason").
* **HTML gallery** with on-the-fly **thumbnails** (no duplicates saved—falls back to originals if thumb fails).
* **(Experimental) Clustering**: group similar photos to speed up album creation. Preview clusters then move/copy to folders.

> ⚠️ LLM features are optional. If you don’t configure a provider, search still works with local embeddings.

---

## 🧰 Requirements

Install Python 3.10+ and then:

```bash
pip install -r requirements.txt
```

### Included libraries (high level)

* `faiss-cpu` – vector index
* `open-clip-torch` + `torch` – CLIP embeddings (falls back to deterministic stubs if not installed)
* `pillow`, `pandas`, `numpy`
* `tqdm`, `requests`, `tenacity`
* *(Optional)* `jsonschema` for config validation
* *(Experimental)* `scikit-learn` for clustering

See `requirements.txt` for exact versions.

---

## ⚙️ Configure

Edit `config.json`:

```jsonc
{
  "device": "auto",                  // "cuda" if you have it, otherwise "cpu"
  "clip_model": "ViT-B-32",
  "images_root": "images",          // folder you want to index
  "index_path": "data/index.faiss", // FAISS index (with .npy/.ids sidecars if FAISS missing)
  "metadata_csv": "data/image_index.csv",
  "memory_path": "data/rag_memory.jsonl",
  "memory_index_path": "data/memory.faiss",
  "topk": 12,
  "distance": "cosine",

  // LLMs (optional)
  "augment_queries": true,
  "enable_clarifier": true,
  "enable_memory_write": true,
  "llm_provider": "ollama",
  "llm_model": "mistral",
  "ollama_host": "http://localhost:11434",

  // UI
  "thumbnail_max_px": 512,
  "enable_rerank": false
}
```

> Tip (Windows): paths like `C:\\Users\\me\\Pictures` work fine.

---

## 🚀 Quickstart (Windows CMD examples)

1. **Index your images** (first run or after big changes):

```cmd
python main.py reindex --full
```

2. **Search**:

```cmd
python main.py search "photos of orange cat" --topk 50 --gallery
```

3. **Status** (sanity check files exist):

```cmd
python main.py status
```

---

## 🔁 Indexing Details

### Incremental reindex

```cmd
python main.py reindex
```

* Walks `images_root`, computes a light file signature for each image.
* Embeds **only** new/changed files and appends them to the vector index.
* Updates `data/image_index.csv` with the latest file signatures.

### Full rebuild (purge + rebuild)

```cmd
python main.py reindex --full
```

* Recomputes **all** embeddings from the current filesystem snapshot.
* **Prunes deleted files**: if a file was removed from disk, it disappears from the CSV and the rebuilt vector index.
* Use this when you mass-delete/move files or suspect index/CSV drift.

### Auto health checks

* During reindex, if the saved IDs in `index.faiss(.ids)` don’t align with `image_index.csv`, a **full rebuild** is triggered automatically (or you’ll be prompted, depending on your version).
* If you ever get weird search results, do a manual reset:

```cmd
del /q data\index.faiss 2>nul
del /q data\index.faiss.npy 2>nul
del /q data\index.faiss.ids 2>nul
python main.py reindex --full
```

---

## 🔍 Searching

```cmd
python main.py search "bread" --topk 100 --gallery --debug
```

* `--gallery` writes/opens `data/search_results.html` with thumbnails.
* `--show N` can open top-N results in your OS viewer.
* `--debug` logs memory usage, augmentation, the final query, and writes `data/debug/last_query.json`.

### Query augmentation & clarifier (optional)

If `enable_clarifier` is true, you might get up to 2 clarifying questions (e.g., "Which J?"), then your query gets rewritten (e.g., "photos of Jason (nicknamed J)").

### Memory & aliases (optional)

* The tool tries to **link nicknames** found in a query to known entities.
* If it proposes a link (e.g., `J → Jason`) it will ask to confirm and then **add the alias** to the entity.
* When you **decline creating a new entity** (e.g., typed a nickname), it will try to resolve it as an alias and offer to add it to an existing entity instead of saving junk.

---

## 🧪 Clustering (Experimental)

Group semantically similar photos (e.g., all the beach shots, all the cat photos) to speed up albuming.

**Preview clusters and write groups to folders**:

```cmd
python main.py cluster --k 8 --min 3 --dest "images\clusters" --copy
```

* `--k` number of clusters (try 6–12 to start)
* `--min` minimum items to consider a valid cluster
* `--dest` root folder to create subfolders like `cluster_0001/`
* `--copy` (default) copies files; use `--move` to move instead

> After previewing, you can rename cluster folders to something human (e.g., `Jason/`, `Mustard/`).

**Note:** Reindex after moving a lot of files so paths/signatures stay in sync:

```cmd
python main.py reindex --full
```

---

## 🧷 Thumbnails

* Stored under `data/thumbnails/` on demand; if thumbnailing fails, the gallery uses the original image.
* No extra storage churn for images already small enough.

---

## 🛠️ Troubleshooting

**I added new photos but search ignores them**

* Run `python main.py reindex` (incremental). If still off, run a full rebuild.

**Results look wrong / I only get one image for every query**

* Likely an out-of-sync FAISS/CSV. Do a full rebuild (see reset commands above).

**I deleted/moved files but they still appear in results**

* Run `python main.py reindex --full` to rebuild and prune.

**Torch/OpenCLIP won’t install on Windows**

* You can comment them out in `requirements.txt`. The app will use deterministic stub embeddings so you can still test the UX.

**Ollama/LLMs not installed**

* Set `augment_queries: false` and `enable_clarifier: false` or just leave provider unset. Search still works.

---

## 🧪 Dev & Tests

* Basic config smoke test:

```bash
pytest -q
```

* To inspect the last search pipeline end-to-end, open `data/debug/last_query.json` (written when `--debug` is passed to `search`).

---

## 🗺️ Roadmap

* Desktop/mobile app with cluster previews and **drag-to-recluster** UX.
* Smarter alias learning (regex + LLM) with fewer prompts.
* Multimodal memory (faces, pets) and per-entity visual descriptors.
* Background watcher to auto-reindex on file changes.
