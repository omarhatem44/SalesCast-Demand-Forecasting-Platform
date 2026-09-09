"""Make the platform importable in tests without installing the package.

The source lives under src/, and modules import as `forecasting.*`, so src/ has
to be on sys.path before collection. pytest imports conftest.py first, which is
why the path insert lives here rather than in the test module.
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))