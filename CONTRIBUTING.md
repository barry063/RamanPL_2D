# Contributing to RamanPL_2D

Thank you for your interest in contributing.

## Contribution flow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Add or update tests for any behavioural change.
3. Run the test suite: `pytest tests/`
4. Open a pull request against `main` with a clear description of the change and its motivation.
5. Add a CHANGELOG entry under the appropriate section.

## Scientific behaviour

This package is used for reproducible spectroscopy analysis. **Do not change scientific or algorithmic behaviour** (fitting logic, preprocessing semantics, peak models, backend outputs) without:
- an explicit motivation in the PR description
- updated or new tests that document the behavioural contract

## Coding expectations

- Match the existing code style.
- Keep changes minimal and traceable to the stated goal.
- Ensure canonical example notebooks still execute cleanly after your change.
- Do not introduce new dependencies without discussion.

## Issues and feature requests

- **Bug reports**: use the GitHub issue template. Include a minimal reproducer, the version, and expected vs actual behaviour.
- **Feature requests**: open an issue before implementing, especially for new peak models, baseline methods, or backend changes. This avoids wasted effort on work that may not fit the project scope.

## Questions

Open a GitHub Discussion or email yuhao19980603@gmail.com.
