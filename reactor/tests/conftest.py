"""Flat model modules are imported just as in a Reactor workspace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
