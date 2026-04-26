# Release checklist

Use this checklist to validate a release candidate before tagging.
This is the operational gate for v0.5.0 and all subsequent releases.

---

## Pre-tag checks

- [ ] Version string is consistent across `src/pyproject.toml`, `src/setup.py`, `src/ramanpl/__init__.py`, and `CITATION.cff`
- [ ] `CHANGELOG` has an entry for this version
- [ ] `README.md` does not reference removed or renamed files

## Build verification

```bash
# Build wheel and sdist
pip install build
python -m build . --outdir dist/
```

- [ ] Wheel builds without error
- [ ] Sdist builds without error
- [ ] `dist/` contains exactly one `.whl` and one `.tar.gz`

## Clean-install smoke

```bash
# Install built wheel into a fresh environment and confirm import
pip install dist/RamanPL_2D-*.whl
python -c "import ramanpl; print(ramanpl.__version__)"
```

- [ ] Wheel installs cleanly in a fresh environment
- [ ] `ramanpl.__version__` matches the release tag
- [ ] All `__all__` names resolve without error

## Test suite

```bash
pip install -e .
pip install pytest
pytest tests/ --ignore=tests/test_mapping_backend_parity.py \
              --ignore=tests/test_preprocessing_backend_resolution.py -v
```

- [ ] Base test suite passes

## Packaging smoke tests

```bash
pytest tests/test_packaging_smoke.py -v
```

- [ ] All packaging smoke tests pass

## Notebook smoke

```bash
pip install nbformat nbconvert ipykernel
python -m ipykernel install --user --name python3
pytest tests/test_notebook_smoke.py -v
```

- [ ] Canonical backend notebooks execute without error

## Benchmark smoke

```bash
pytest tests/test_release_benchmark_smoke.py -v
```

- [ ] Benchmark harness runs and emits structurally valid output

## Repository state

- [ ] `src/` contains only package and build content (no stray notebooks or data files)
- [ ] No unintended files in the sdist or wheel (inspect with `unzip -l dist/*.whl`)
- [ ] No planned breaking API or backend changes before this tag

---

Once all boxes are checked, tag the release:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```
