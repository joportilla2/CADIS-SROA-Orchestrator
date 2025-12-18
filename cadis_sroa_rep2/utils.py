from __future__ import annotations

"""Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fingerprint_source(sources) -> str:
    """Compute a deterministic SHA256 fingerprint of source code.

    Parameters
    ----------
    sources:
        Either a package directory (str/Path) or an iterable of file paths.

    Notes
    -----
    * Deterministic: paths are sorted; each file is hashed by path + raw bytes.
    * Designed for Q1-grade reproducibility (stable across runs).
    """
    # Normalize inputs to a list of files
    if isinstance(sources, (str, Path)):
        root = Path(sources)
        files = sorted(root.rglob("*.py"))
    else:
        files = sorted(Path(p) for p in sources)

    h = hashlib.sha256()
    for f in files:
        if f.is_file():
            h.update(str(f).encode("utf-8"))
            h.update(b"\n")
            h.update(f.read_bytes())
            h.update(b"\n")
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def compute_checksums(paths: List[str], root: str | None = None) -> Dict[str, str]:
    """Compute SHA256 for a list of files.

    If root is provided, keys are relative paths from root (POSIX-style).
    Otherwise keys are basenames (legacy behavior).
    """
    out: Dict[str, str] = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        key = os.path.basename(p)
        if root is not None:
            try:
                key = str(Path(p).resolve().relative_to(Path(root).resolve())).replace(os.sep, "/")
            except Exception:
                key = os.path.basename(p)
        out[key] = sha256_file(p)
    return out


def stable_json(obj) -> str:
    """Deterministic JSON serialization (sorted keys, compact)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
