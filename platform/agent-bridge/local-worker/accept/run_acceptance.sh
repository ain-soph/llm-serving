#!/bin/zsh
# run_acceptance.sh BASE_DIR [HARNESS...]: the local-worker acceptance test, every harness on the same task,
# each on a fresh copy of the fixture repo, all inside ONE exclusive gpu-lock session (exclusive because the
# summaries report prefill/decode speeds, AGENTS.md rule 1). The share is released in the EXIT trap, before the
# CPU-only verification (verify.sh) runs.
set -u
base=${1:?usage: run_acceptance.sh BASE_DIR [claude codex]}; shift
harnesses=($@); (( $#harnesses )) || harnesses=(claude codex)
here=${0:A:h}; worker=${here:h}/run; lock=$HOME/llm-serving/bin/gpu-lock
name=local_worker_accept_$$; per_run=${ACCEPT_TIMEOUT:-30m}
mkdir -p $base
for h in $harnesses; do $here/make_repo.sh $base/repo-$h || exit 3; done

held=0 hb=
cleanup() {
  trap - EXIT INT TERM
  [[ -n $hb ]] && kill $hb 2>/dev/null
  (( held )) && { echo "[accept] $($lock release $name | tail -1)"; held=0; }
  $lock status | tail -2
}
trap cleanup EXIT
trap 'exit 143' INT TERM

while true; do
  $lock acquire $name all; rc=$?
  (( rc == 0 )) && break
  (( rc == 1 )) || { echo "[accept] gpu-lock refused ($rc)"; exit 75; }
done
held=1; echo "[accept] $(date +%T) exclusive share $name acquired"
( while sleep 600; do $lock heartbeat $name >/dev/null; done ) & hb=$!

for h in $harnesses; do
  echo "[accept] $(date +%T) harness $h"
  $worker --under-lock $name --harness $h --variant normal --cwd $base/repo-$h --timeout $per_run --label accept \
    --task-file $here/task.md > $base/summary-$h.json; rc=$?
  echo "[accept] $(date +%T) $h exit $rc"
done
cleanup   # release before the CPU-only checks

for h in $harnesses; do
  echo "================ verify $h"; $here/verify.sh $base/repo-$h > $base/verify-$h.txt 2>&1; sed -n '1,20p' $base/verify-$h.txt
done
