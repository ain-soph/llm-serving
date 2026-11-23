#!/bin/zsh
# verify.sh REPO: independent check of a worker's result (CPU only): visible tests, hidden tests, tests/ untouched, diff.
set -u
repo=${1:?usage: verify.sh REPO}; here=${0:A:h}
cd $repo; export PYTHONDONTWRITEBYTECODE=1; find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
echo "== git status"; git status --short
echo "== tests/ untouched"; git diff --quiet HEAD -- tests && echo "yes" || echo "NO: tests/ modified"
echo "== visible tests"; .venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3
echo "== hidden tests"; PYTHONPATH=$repo .venv/bin/python -B -m pytest -q -p no:cacheprovider --rootdir=$repo $here/hidden_test.py 2>&1 | tail -3
echo "== diff"; git diff HEAD
