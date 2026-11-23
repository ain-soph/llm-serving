"""Generate the evaluation set through the API for one variant (model ID + prompt_enhance).
Usage: run_eval.py BASE_URL MODEL ENHANCE OUTDIR  -> OUTDIR/<id>.png + OUTDIR/results.jsonl (resumable)."""
import base64, io, json, os, sys, time, urllib.request, uuid
from pathlib import Path

base, model, enhance, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], Path(sys.argv[4])
here = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
key = (Path(os.environ.get("LLM_API_KEY_FILE") or Path(__file__).resolve().parents[4] / "platform/llama-swap/.api_key")).read_text().strip()
done = {json.loads(l)["id"] for l in open(out / "results.jsonl")} if (out / "results.jsonl").exists() else set()
SEED = 11


def post(path, body, ctype):
    req = urllib.request.Request(base + path, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


for line in open(here / "prompts.jsonl"):
    case = json.loads(line)
    if case["id"] in done:
        continue
    fields = {"model": model, "prompt": case["prompt"], "size": "1024x1024", "seed": SEED, "prompt_enhance": enhance}
    t = time.time()
    try:
        if case.get("images"):
            b = uuid.uuid4().hex
            parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
            for p in case["images"]:
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image[]"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                             + (here / p).read_bytes() + b"\r\n")
            resp = post("/v1/images/edits", b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}")
        else:
            resp = post("/v1/images/generations", json.dumps(fields).encode(), "application/json")
        item = resp["data"][0]
        (out / f"{case['id']}.png").write_bytes(base64.b64decode(item["b64_json"]))
        rec = {"id": case["id"], "seconds": round(time.time() - t, 1), "revised_prompt": item.get("revised_prompt")}
    except Exception as e:  # keep going; the error is recorded
        body = getattr(e, "read", lambda: b"")()
        rec = {"id": case["id"], "seconds": round(time.time() - t, 1), "error": f"{e} {body[:200]!r}"}
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False)[:200], flush=True)
