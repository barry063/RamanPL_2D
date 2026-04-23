from setuptools import setup
from pathlib import Path

_here = Path(__file__).resolve().parent
# README lives one level up (repo root) during a normal source build.
# When building wheel from sdist, _here.parent is a temp dir; fall back
# to empty string so the build doesn't fail.
for _p in [_here.parent / "README.md", _here / "README.md"]:
    if _p.exists():
        _readme = _p.read_text(encoding="utf-8")
        break
else:
    _readme = ""

setup(
    long_description=_readme,
    long_description_content_type="text/markdown",
)
