# invstats

Tiny inventory statistics package (local-worker acceptance fixture).

- `invstats.parsing.parse_csv(text)` reads `name,qty,price` CSV text (first line is a header) into `Item`s.
- `invstats.report` computes totals and summaries over a list of items.

Run the tests with `.venv/bin/python -m pytest -q`.
