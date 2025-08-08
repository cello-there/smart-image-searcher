# AI Image Searcher V3

A local-first, modular image search system that combines **CLIP embeddings** (for images & text), a **FAISS** vector index, and a lightweight **RAG layer** over personal “memory docs.” The RAG layer uses a local or cloud **LLM** (Ollama Mistral by default) to: (1) expand queries, (2) ask clarifying questions when the intent is vague, (3) infer/normalize entities (like your pets or trip names), and (4) optionally update memory (aliases, attributes) with user consent.

---

## Why this exists

* **Find your photos faster** using semantic search: “photos of my orange cat at night,” “NYC trip skyline,” etc.
* **Be precise without hard‑coding:** The LLM does the heavy lifting—clarifying, normalizing terms, and mapping nicknames to canonical entities.
* **Stay local & private:** Run everything on your machine. Embeddings and indexes live in `data/`. No raw photos are uploaded.

---

## Features

* **CLIP-powered semantic search** over your image folders
* **FAISS index** (with numpy fallback) for fast retrieval
* **RAG memory** of user-specific entities (e.g., pets, trips) to personalize queries
* **LLM-driven query augmentation** (no brittle synonym lists)
* **Clarifying questions** only when needed, guided by memory context
* **Alias learning** (e.g., “Floof” → “Fluffy”) with opt-in persistence
* **Modular backends** so you can swap:

  * Embedders (OpenCLIP vs stubs)
  * LLM provider (Ollama / Anthropic / OpenAI)
  * Index strategy (FAISS or pure numpy)
* **HTML gallery** output and/or open top results in your default viewer
* **Incremental reindex** that only embeds changed/new files

---

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> On Windows, PyTorch can be heavy. If you hit install issues, you can still run the random-stub embedders; search will work but be less accurate until PyTorch/OpenCLIP are installed.

### 2) (Optional) Start an LLM locally with Ollama

* Install Ollama and pull a small model (e.g., `mistral`):

```bash
ollama pull mistral
```

* Ensure Ollama is running (default host `http://localhost:11434`).

### 3) Configure

Edit `config.json`:

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
  "thumbnail_max_px": 512,
  "enable_rerank": false,
  "enable_memory_write": true,
  "safe_mode_redact_names": false,
  "use_ivf": false,
  "use_opq": false,
  "llm_provider": "ollama",
  "llm_model": "mistral",
  "ollama_host": "http://localhost:11434"
}
```

> Point `images_root` to the folder containing your photos. Subfolders are handled automatically.

### 4) Index images

```bash
python main.py reindex
```

* First run will embed & index all supported images under `images_root`.
* Subsequent runs only pick up new/modified files (use `--full` for a full rebuild).

### 5) Search

```bash
python main.py search "photos of my cat at night" --gallery --show 3
```

* `--gallery` opens an HTML grid of results.
* `--show N` opens the top N images in your OS default viewer.

---

## How it works

### 1) Embedding & Indexing

* **ImageEmbedder** encodes images (OpenCLIP if available; otherwise a deterministic stub for development).
* **TextEmbedder** encodes queries into the same vector space.
* **VectorIndex** stores vectors & ids in FAISS (or numpy fallback) and supports nearest-neighbor search.
* **incremental\_reindex** walks `images_root`, computes a signature (name+size+mtime), embeds only changed/new files, and updates `data/index.*` plus `data/image_index.csv`.

### 2) Memory (RAG)

* User-specific “memory docs” live in `data/rag_memory.jsonl` (one JSON per line).
* **MemoryVectorIndex** builds a FAISS/numpy index over textual summaries of these docs for quick retrieval as RAG context.
* The **Clarifier** and **QueryAugmenter** read this context to:

  * Ask **clarifying questions** only when necessary
  * Infer canonical entities or **learn aliases** (with your consent)
  * Normalize queries (“kitty” → cat) without hard-coded lists

### 3) Retrieval

* We optionally clarify the query, augment it using memory context, then embed the final text and search the image index.
* If enabled, a simple **CLIP re-ranker** can re-score the top K results for better ordering.
* Finally, we print results and optionally open the **HTML gallery** and/or native image viewer.

---

## Memory & Aliases

* When you mention a new entity (e.g., `Fluffy` the cat), the system can ask a few targeted questions (species, aliases, attributes) and then save a structured entity doc.
* If you later search for a nickname (e.g., `Floof`), the system may propose linking it as an alias for `Fluffy` and, if you confirm, update the memory record.
* Memory writing is **opt‑in** (`enable_memory_write: true`) and always asks before persisting changes.

### Memory file format (example)

```jsonl
{"type":"entity","name":"fluffy","category":"pet","kind":"cat","aliases":["floof"],"attributes":{"color":"dark grey"},"user_owned":true,"persistent":true}
{"type":"entity","name":"jason","category":"pet","kind":"cat","aliases":["jas","J"],"attributes":{"color":"orange"},"persistent":true}
```

---

## Commands

```bash
# show health of artifacts
python main.py status

# (re)index images (incremental)
python main.py reindex
python main.py reindex --full   # force rebuild

# search
python main.py search "your query here" --gallery --show 5
```

---

## Configuration notes

* **llm\_provider / llm\_model**: choose between `ollama` (default), `anthropic`, or `openai`.
* For **Ollama**, ensure the model exists locally (`ollama pull mistral`).
* Set `enable_rerank: true` to use CLIP image embeddings to re-rank the top K.
* `safe_mode_redact_names: true` will try to avoid injecting entity names into the final query string.

---

## Troubleshooting

* **No results / weird ordering**: If running without PyTorch/OpenCLIP, you’re using stub embeddings—install Torch & `open-clip-torch`.
* **FAISS not installed**: We fall back to numpy search (slower). Install `faiss-cpu` for speed.
* **Main errors with prompts/schemas**: Ensure prompts in `rag/prompts.py` still match what `utils/llm.complete_json` expects (especially keys like `expanded_query`).
* **Windows path issues**: We normalize paths in the HTML gallery; if a link fails, open the printed path directly.

---

## Roadmap / Ideas

* Train a small **re-ranker** tuned to your collection
* Add **face/subject tagging** via lightweight local models
* Expose a simple **web UI** for search/history/memory editing
* Export/import memory as a single JSON for easy backup

---

## License

MIT (adjust as you like). All images remain your property and never leave your machine.
