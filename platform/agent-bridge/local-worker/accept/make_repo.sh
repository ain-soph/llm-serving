#!/bin/zsh
# make_repo.sh DEST: fresh git copy of the acceptance fixture with a local .venv (pytest), outside the worker sandbox.
set -eu
dest=${1:?usage: make_repo.sh DEST}
[[ ! -e $dest ]] || { echo "exists: $dest"; exit 2; }
cp -R ${0:A:h}/fixture $dest
cd $dest
git init -q && git add -A && git -c user.name=fixture -c user.email=fixture@localhost commit -qm "fixture"
uv venv -q --python 3.12 .venv && VIRTUAL_ENV=$dest/.venv uv pip install -q pytest
echo "$dest ready ($(.venv/bin/python -m pytest --version 2>&1))"
