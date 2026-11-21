"""Entry point (kept tiny: the logic lives in precompiled modules).

runner.py: static selection (blast.py), then in-container verification (verify.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402

runner.main()
