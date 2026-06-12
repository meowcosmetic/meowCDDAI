"""Pytest config for meowCDDAI unit tests.

Ensures the meowCDDAI package root is importable so tests can do
`from job_repository import ...` and `from config import Config`
regardless of the directory pytest is launched from.
"""

import os
import sys

# Insert the meowCDDAI root (parent of this tests/ dir) at the front of sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
