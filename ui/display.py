import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
import os, sys, time, subprocess, webbrowser
from .gallery import write_gallery

def truncate_path(p: str, maxlen: int = 80) -> str:
    if len(p) <= maxlen:
        return p
    head = p[: maxlen // 2 - 2]
    tail = p[-maxlen // 2 + 1 :]
    return f"{head}…{tail}"


def _open_file_native(path: str, delay: float = 0.2):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        if delay:
            time.sleep(delay)
    except Exception as e:
        print(f"Could not open {path}: {e}")


def present_results(results, cfg: dict, show: int = 0, gallery: bool = False):
    """Flexible presenter: optionally open top-N in native viewer and/or an HTML gallery."""
    if show and results:
        for r in results[:show]:
            _open_file_native(r["path"])

    if gallery and results:
        html_path = write_gallery(results, cfg)
        webbrowser.open(Path(html_path).resolve().as_uri())
        print(f"\nOpened gallery: {html_path}")