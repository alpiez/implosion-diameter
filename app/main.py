"""Entry point (kept tiny: the logic lives in blast.py, which is precompiled to .pyc)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blast  # noqa: E402

blast.main()
