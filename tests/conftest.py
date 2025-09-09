"""
Ensure the local package under this workspace's `src/` is imported first during tests.

This avoids accidentally picking up another installed copy of
`swissclim_evaluations` elsewhere on PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the workspace's src/ at the beginning of sys.path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
