import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "RamanPL_2D"
author = "Hao Yu"
release = "0.6.5"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

autosummary_generate = True
html_theme = "furo"
html_title = "RamanPL_2D documentation"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock optional dependencies so autodoc works in clean environments
autodoc_mock_imports = ["ramanspy", "renishawWiRE", "sklearn"]
