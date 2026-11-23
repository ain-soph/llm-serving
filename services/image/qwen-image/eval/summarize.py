"""Per-pair and per-category win rates from runs/judge-*.jsonl (tie = half a vote each; '?' votes left out)."""
import collections, json, sys
from pathlib import Path
runs = Path(__file__).resolve().parent / "runs"
CATS = ["text", "count", "spatial", "short", "style", "edit"]
for path in sorted(runs.glob("judge-*.jsonl")):
    va, vb = path.stem[len("judge-"):].split("-vs-")
    per = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for line in open(path):
        row = json.loads(line)
        for v in row["votes"]:
            for key in (row["cat"], "all"):
                c = per[key]
                if v["winner"] == va: c[0] += 1
                elif v["winner"] == vb: c[1] += 1
                elif v["winner"] == "tie": c[0] += 0.5; c[1] += 0.5
                else: c[2] += 1
    def rate(k):
        a, b, q = per[k]
        return f"{100 * b / (a + b):.0f}%" if a + b else "-"
    cells = " | ".join(rate(k) for k in CATS)
    a, b, q = per["all"]
    print(f"| {va} vs **{vb}** | {rate('all')} ({b:g}/{a + b:g}) | {cells} | {q} |")
