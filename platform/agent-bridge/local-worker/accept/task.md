This repository is a small Python package `invstats` with pytest tests. The tests currently fail.

1. Read README.md, every file in invstats/ and tests/.
2. Fix the two bugs in the package code (one in invstats/parsing.py, one in invstats/report.py) so the behaviour
   matches the docstrings.
3. Add a function `top_n(items, n)` to invstats/report.py that returns the n items with the highest value
   (qty * price), highest first, and export it from invstats/__init__.py (import and `__all__`).
4. Add one line to README.md describing `top_n`.
5. Run the tests with `.venv/bin/python -m pytest -q` and keep fixing until all tests pass.
Do not edit anything under tests/. Do not commit. Finish with a short report: files changed and the final pytest summary line.
