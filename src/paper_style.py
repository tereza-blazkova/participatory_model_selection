"""
The paper style for the pipeline's own plot modules — the same file the paper
figures use (figures/scripts/paper_style.py), loaded by path.

It lives under figures/ so the draw scripts can reach it without importing
anything from src/ (the wall that lets figures/ re-render on its own). This
shim is the only way src/ gets it; no src/ module sets rcParams itself.
"""
import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "figures" / "scripts" / "paper_style.py"

if "paper_style" in sys.modules:
    _style = sys.modules["paper_style"]
else:
    _spec = importlib.util.spec_from_file_location("paper_style", _PATH)
    _style = importlib.util.module_from_spec(_spec)
    sys.modules["paper_style"] = _style
    _spec.loader.exec_module(_style)

METHOD_COLOR  = _style.METHOD_COLOR
METHOD_MARKER = _style.METHOD_MARKER
TEXTWIDTH_IN  = _style.TEXTWIDTH_IN
apply_style   = _style.apply_style
canvas        = _style.canvas
