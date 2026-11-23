"""Pairwise judge with the chat model (vision): for every prompt, variant A vs B in both orders.
Usage: judge.py BASE_URL RUNS_DIR VARIANT_A VARIANT_B  -> RUNS_DIR/judge-A-vs-B.jsonl + a summary line."""
import base64, io, json, os, random, sys, urllib.request
from pathlib import Path
from PIL import Image

base, runs, va, vb = sys.argv[1].rstrip("/"), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
here = Path(__file__).resolve().parent
key = (Path(os.environ.get("LLM_API_KEY_FILE") or Path(__file__).resolve().parents[4] / "platform/llama-swap/.api_key")).read_text().strip()
MODEL = "qwen3.8-flash-next"


def data_url(path, side=768):
    im = Image.open(path).convert("RGB"); im.thumbnail((side, side)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(case, first, second):
    for _ in range(2):
        v = ask_once(case, first, second)
        if v["winner"] != "?":
            return v
    return v


def ask_once(case, first, second):
    content = []
    if case.get("images"):
        content.append({"type": "text", "text": "Source image(s) the edit was applied to:"})
        content += [{"type": "image_url", "image_url": {"url": data_url(here / p)}} for p in case["images"]]
        task = f'The user asked to edit the source image(s) with this instruction: "{case["prompt"]}".'
    else:
        task = f'The user asked for this image: "{case["prompt"]}".'
    content += [{"type": "text", "text": "Image A:"}, {"type": "image_url", "image_url": {"url": data_url(first)}},
                {"type": "text", "text": "Image B:"}, {"type": "image_url", "image_url": {"url": data_url(second)}},
                {"type": "text", "text": task + " Which image fulfils the request better? Judge, in this order: every "
                 "requested element present (objects, counts, positions, exact text, the edit applied while the rest is "
                 "kept); then overall image quality. Answer ONLY with JSON: "
                 '{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}'}]
    # Flash-Next thinks first (reasoning_content); 400 tokens left the answer empty on ~half the calls
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 3000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    for start in reversed([i for i, c in enumerate(text) if c == "{"]):
        try:
            obj = json.loads(text[start:text.index("}", start) + 1])
            if obj.get("winner") in ("A", "B", "tie"):
                return obj
        except (ValueError, json.JSONDecodeError):
            continue
    return {"winner": "?", "reason": text[:120]}


out = runs / f"judge-{va}-vs-{vb}.jsonl"
# resumable: keep prompts whose two votes are both valid, judge the rest again
kept = {}
if out.exists():
    for line in open(out):
        row = json.loads(line)
        if all(v["winner"] != "?" for v in row["votes"]):
            kept[row["id"]] = row
rows = []
score = {va: 0.0, vb: 0.0, "?": 0}
for line in open(here / "prompts.jsonl"):
    case = json.loads(line)
    a, b = runs / va / f"{case['id']}.png", runs / vb / f"{case['id']}.png"
    if not (a.exists() and b.exists()):
        continue
    if case["id"] in kept:
        rows.append(kept[case["id"]])
        for v in kept[case["id"]]["votes"]:
            if v["winner"] == "tie":
                score[va] += 0.5; score[vb] += 0.5
            else:
                score[v["winner"]] += 1
        continue
    votes = []
    for first, second, names in ((a, b, (va, vb)), (b, a, (vb, va))):
        v = ask(case, first, second)
        winner = {"A": names[0], "B": names[1]}.get(v["winner"], v["winner"])
        votes.append({"winner": winner, "reason": v.get("reason", "")})
        if winner in (va, vb):
            score[winner] += 1
        elif winner == "tie":
            score[va] += 0.5; score[vb] += 0.5
        else:
            score["?"] += 1
    rows.append({"id": case["id"], "cat": case["cat"], "votes": votes})
with open(out, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"pair": f"{va} vs {vb}", "score": score}, ensure_ascii=False))
