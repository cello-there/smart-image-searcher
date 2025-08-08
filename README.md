# AI Image Searcher V3

Local‑first semantic image search using **CLIP** embeddings + **FAISS** with a small **RAG** layer for entities (pets, trips, nicknames). Runs entirely on your machine; no photos leave your box.

---

## Highlights

* **Semantic search**: “photos of my orange cat at night,” “yellow mustard bottle on white background.”
* **RAG memory**: remembers entities (e.g., *Fluffy* ↔ *Floof*), learns aliases with your approval.
* **LLM assists**: clarifies vague queries and expands them into better CLIP prompts.
* **Incremental indexing** with **auto‑detect of stale index** (see below).
* **Windows‑friendly**: examples use **CMD** (not PowerShell).

---

## Install

```bash
pip install -r requirements.txt
```

> If PyTorch / open-clip are heavy, you can still run on stub embeddings (slower/less accurate). Install `faiss-cpu` for speed.

---

## Configure (`config.json`)

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
  "augment_queries": true,
  "enable_clarifier": true,
  "enable_rerank": false,
  "enable_memory_write": true,
  "thumbnail_max_px": 512,
  "safe_mode_redact_names": false,
  "use_ivf": false,
  "use_opq": false,
  "llm_provider": "ollama",
  "llm_model": "mistral",
  "ollama_host": "http://localhost:11434",
  "auto_reindex": true  // optional: reindex on detected changes
}
```

**Notes**

* `images_root`: the folder to crawl. Subfolders are included.
* `auto_reindex`: when `true`, `search` will auto‑reindex if it detects new/changed files (based on signatures). If `false`, you’ll be prompted.

---

## Index your photos

### First build

```cmd
python main.py reindex
```

### Force a clean rebuild

```cmd
python main.py reindex --full
```

### Auto‑detect behavior

* On `search`, the app compares what’s on disk vs what’s in `data/image_index.csv`.
* If files were added/renamed/updated:

  * With `auto_reindex: true`, it will run an incremental reindex automatically.
  * Otherwise, you’ll see a prompt telling you to run `python main.py reindex`.

---

## Search

```cmd
python main.py search "photos of my orange cat"
python main.py search "yellow mustard bottle on white background" --gallery --show 3
```

* `--gallery` opens an HTML grid. `--show N` opens the top N images natively.
* The LLM will only ask clarifying questions when your intent is genuinely ambiguous.

---

## Memory & Aliases (RAG)

* Entities you confirm are stored in `data/rag_memory.jsonl`.
* When you type a nickname (e.g., `Floof`) and we believe it maps to an existing entity (`Fluffy`), you’ll be asked to confirm adding it as an alias.
* If you type a short form like `J` and decline creating a new entity, the system now **offers to attach it as an alias** to a likely existing entity (e.g., `Jason`).
* Memory is **opt‑in**; set `enable_memory_write` to control persistence.

**Memory example** (`rag_memory.jsonl`):

```jsonl
{"type":"entity","name":"Fluffy","category":"pet","kind":"cat","aliases":["Floof"],"attributes":{"color":"dark grey"},"persistent":true}
{"type":"entity","name":"Jason","category":"pet","kind":"cat","aliases":["J"],"attributes":{"color":"orange"},"persistent":true}
```

**Auto‑repair**

* On load, memory sanitizes aliases (splits comma/"and", removes sentence‑like junk) and repairs odd attribute keys (e.g., `color|attributes.note` → `color`).

---

## Useful Windows CMD snippets

**Check if a file is indexed**

```cmd
findstr /i /c:"download (4).jpg" data\image_index.csv
```

**Delete old FAISS artifacts (if things get weird)**

```cmd
del /q data\index.faiss 2>nul
rem (if numpy fallback exists)
del /q data\index.faiss.npy 2>nul

del /q data\index.faiss.ids 2>nul
python main.py reindex --full
```

**Confirm config paths**

```cmd
type config.json | findstr /i "metadata_csv index_path images_root"
```

---

## FAQ / Troubleshooting

**I added new photos but search doesn’t find them.**

* Run: `python main.py reindex` (or enable `auto_reindex`).
* Verify the file appears in `data/image_index.csv` with `findstr` (see snippet above).

**Gallery opens but images don’t load.**

* Thumbnails are written to `data/thumbnails/`. Make sure the original paths printed in the console are valid.

**LLM keeps asking needless questions.**

* That usually means the query is ambiguous (e.g., `photos of J`). Try adding one more hint, or answer the two quick clarifiers. The system won’t ask if memory already disambiguates it.

**Aliases attached to the wrong thing.**

* You can run the same alias again; when prompted, decline new entity and accept the suggested existing entity to attach the alias correctly.

**Performance is slow.**

* Install `faiss-cpu` and real CLIP (`torch`, `open-clip-torch`).

---

## How it works (short)

1. **Index** walks `images_root`, computes signatures, embeds new/changed files, and writes:

   * vectors → `data/index.*`
   * metadata → `data/image_index.csv`
2. **Memory** builds a tiny FAISS/numpy index over your docs to supply context to the LLM.
3. **Search** optionally clarifies, augments the query, embeds text, queries FAISS, optionally reranks, then prints/open results.

---

## Roadmap

* Optional web UI
* Face/person clustering
* Fine‑tuned re‑ranker

---

## License

MIT (images remain your property; nothing is uploaded).
