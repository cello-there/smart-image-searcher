from pathlib import Path
from PIL import Image

def _thumb(src: str, outdir: Path, max_px: int) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    src_p = Path(src)
    thumb_p = outdir / (src_p.stem + "_thumb.jpg")
    if thumb_p.exists():
        return thumb_p
    try:
        im = Image.open(src_p).convert("RGB")
        im.thumbnail((max_px, max_px))
        im.save(thumb_p, "JPEG", quality=85)
    except Exception:
        # If thumbnailing fails, fall back to original image
        return src_p
    return thumb_p


def write_gallery(results, cfg: dict) -> str:
    out_dir = Path("data")
    thumbs = out_dir / "thumbnails"
    max_px = int(cfg.get("thumbnail_max_px", 512))

    items = []
    for r in results:
        tpath = _thumb(r["path"], thumbs, max_px)
        items.append({
            "thumb": str(tpath),
            "full": r["path"],
            "score": r["score"],
        })

    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Image Search Results</title>",
        "<style>body{font-family:sans-serif;margin:16px} .grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));} .card{border:1px solid #ddd;border-radius:10px;padding:8px} img{width:100%;height:auto;border-radius:8px} .meta{font-size:12px;color:#555;margin-top:6px;word-break:break-all}</style>",
        "<h2>Image Search Results</h2>",
        "<div class='grid'>"
    ]
    for it in items:
        full_uri = Path(it["full"]).resolve().as_uri()
        thumb_uri = Path(it["thumb"]).resolve().as_uri()
        html.append(
            f"<a class='card' href='{full_uri}' target='_blank'>"
            f"<img src='{thumb_uri}' alt='thumb'/>"
            f"<div class='meta'>score: {it['score']:.4f}<br/>{it['full']}</div>"
            "</a>"
        )
    html.append("</div>")
    out = out_dir / "search_results.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return str(out.resolve())