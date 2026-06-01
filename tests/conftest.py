"""Pytest conftest — adds src/ to sys.path so test imports work from any
invocation directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
