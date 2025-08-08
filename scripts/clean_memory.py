from pathlib import Path
import json

# Path to your memory file
p = Path("data/rag_memory.jsonl")

if not p.exists():
    print("No memory file found at", p)
    raise SystemExit(0)

TRIVIAL_NEGATIONS = {"", "no", "none", "n/a", "na", "null", "nil"}

BAD_CONTEXT_BITS = {
    "any photo at night",
    "any photos at night",
    "no preference",
    "does not have a preference",
    "just any photo",
    "general photos",
}

def keep(d: dict) -> bool:
    t = d.get("type")
    if t == "preference":
        v = str(d.get("value", "")).strip().lower()
        return v not in TRIVIAL_NEGATIONS and len(v) > 1
    if t == "context":
        desc = (d.get("description") or "").strip().lower()
        return bool(desc) and not any(bit in desc for bit in BAD_CONTEXT_BITS)
    if t == "entity":
        # require both a name and kind so we don't save partials
        return bool(d.get("name")) and bool(d.get("kind"))
    return False

# Read -> filter -> write
docs = []
for line in p.read_text(encoding="utf-8").splitlines():
    try:
        d = json.loads(line)
        if keep(d):
            docs.append(d)
    except Exception:
        pass

backup = p.with_suffix(".jsonl.bak")
backup.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + ("\n" if docs else ""), encoding="utf-8")
# swap backup into place to be safe
backup.replace(p)

print(f"kept {len(docs)} docs -> {p}")
