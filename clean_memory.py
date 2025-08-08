# scripts/clean_memory.py
from pathlib import Path
import json

p = Path("data/rag_memory.jsonl")
if not p.exists():
    print("no memory file found")
    raise SystemExit

def keep(d):
    t = d.get("type")
    if t == "preference":
        v = str(d.get("value","")).strip().lower()
        return v not in {"", "no", "none", "n/a", "na", "null", "nil"}
    if t == "context":
        desc = (d.get("description") or "").strip().lower()
        bad = ["any photo at night","any photos at night","no preference","does not have a preference","just any photo","general photos"]
        return bool(desc) and not any(b in desc for b in bad)
    if t == "entity":
        return bool(d.get("name")) and bool(d.get("kind"))
    return False

docs = []
for line in p.read_text(encoding="utf-8").splitlines():
    try:
        d = json.loads(line)
        if keep(d):
            docs.append(d)
    except Exception:
        pass

p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + ("\n" if docs else ""), encoding="utf-8")
print(f"kept {len(docs)} docs")
