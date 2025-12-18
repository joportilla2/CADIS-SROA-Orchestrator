import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# v5 uses a flat layout (package lives at repo root).
# Ensure the repo root is importable for tests without requiring installation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
