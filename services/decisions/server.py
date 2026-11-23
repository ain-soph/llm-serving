"""OpenAI Decisions API (POST /v1/decisions) in front of SystemOne decision models on llama-swap (2026-11-16).

  python3 server.py [--port 8095] [--upstream http://127.0.0.1:8081]

A stateless translator, run as the LaunchAgent com.local.decisions; the gateway (Caddy :8080) sends /v1/decisions here.
Each request is rewritten to SystemOne and posted to llama-swap's /v1/systemone, which picks the model by its `model`
field and loads it on demand, so any SystemOne model added to llama-swap works here unchanged. The client's
Authorization header is passed through: llama-swap does the authentication, this process holds no key.

OpenAI -> SystemOne (OpenAI spec: developers.openai.com/api/reference/resources/decisions/methods/create)
  input       string -> state; message array -> text, one "role: text" block per message (input_json parts as JSON,
              tool calls and outputs as "tool call NAME: ARGS" / "tool output: OUTPUT"); input_image parts ->
              SystemOne "images" (base64 data URLs only), allowed only for models whose llama-swap entry declares
              image input (capabilities.in), since a text model would silently ignore them. Images in a question's
              instructions: 400 (SystemOne images belong to the whole request).
  predicate   -> noul; probability = P(true).
  choice      -> choice; criteria {value: description}, options in the given order (a value is the text the model
              reads; boolean values are shown as true/false); probabilities come back in the given order.
  score       -> score; criteria = levels lowest first, each "label: description" (or the one that is set);
              score = probability-weighted level index from 0, as both specs define it.
  questions   ids q0, q1, ... upstream; answers are returned in request order with each question's name.
Not supported (400): image URLs other than data URLs, prompt_cache_options. Accepted and ignored:
prompt_cache_key, safety_identifier.
Upstream SystemOne answers have no refusal, so none is returned. confidence is the upstream value.
"""
import argparse, json, logging, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("decisions")


class RequestError(ValueError):
    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


def text_of(content, where, images=None):
    """Text of a string or a list of content parts (input_text / output_text / refusal / input_json); input_image
    data URLs are appended to images (None: images are not allowed here)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError(f"{where} must be a string or an array of content parts", where)
    out = []
    for i, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text"):
            out.append(str(part.get("text", "")))
        elif kind == "refusal":
            out.append(str(part.get("refusal", "")))
        elif kind == "input_json":
            out.append(json.dumps(part.get("data"), ensure_ascii=False, indent=2))
        elif kind == "input_image":
            if images is None:
                raise RequestError("images are only supported in input, not in question instructions", f"{where}[{i}]")
            url = part.get("image_url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise RequestError("only base64 data URLs (data:image/...;base64,...) are supported for images",
                                   f"{where}[{i}].image_url")
            images.append(url)
        else:
            raise RequestError(f"unsupported content part type {kind!r}", f"{where}[{i}].type")
    return "\n".join(out)


def state_of(inp, images):
    if isinstance(inp, str):
        return inp
    if not isinstance(inp, list) or not inp:
        raise RequestError("input must be a string or a non-empty array of input items", "input")
    blocks = []
    for i, item in enumerate(inp):
        if not isinstance(item, dict):
            raise RequestError("each input item must be an object", f"input[{i}]")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "developer", "system", "assistant"):
                raise RequestError(f"unsupported role {role!r}", f"input[{i}].role")
            blocks.append(f"{role}: {text_of(item.get('content', ''), f'input[{i}].content', images)}")
        elif kind in ("function_call", "custom_tool_call"):
            args = item.get("arguments") if kind == "function_call" else item.get("input")
            blocks.append(f"tool call {item.get('name', '')}: {args}")
        elif kind in ("function_call_output", "custom_tool_call_output"):
            blocks.append(f"tool output: {text_of(item.get('output', ''), f'input[{i}].output', images)}")
        else:
            raise RequestError(f"unsupported input item type {kind!r}", f"input[{i}].type")
    return "\n\n".join(blocks)


def to_systemone(body):
    """(SystemOne request, per-question metadata) for an OpenAI Decisions request."""
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestError("model is required", "model")
    if "input" not in body:
        raise RequestError("input is required", "input")
    if body.get("prompt_cache_options") is not None:
        raise RequestError("prompt_cache_options is not supported", "prompt_cache_options")
    questions = body.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 400:
        raise RequestError("questions must be an array of 1 to 400 questions", "questions")
    out, meta = {}, []
    for i, q in enumerate(questions):
        p = f"questions[{i}]"
        if not isinstance(q, dict):
            raise RequestError("each question must be an object", p)
        kind, name = q.get("type"), q.get("name")
        instructions = text_of(q["instructions"], f"{p}.instructions") if q.get("instructions") is not None else None
        sq = {"type": kind, "instructions": instructions}
        if kind == "predicate":
            sq["type"], values = "noul", None
        elif kind == "choice":
            choices = q.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise RequestError("choices must be an array of 2 to 255 choices", f"{p}.choices")
            values, criteria = [], {}
            for j, c in enumerate(choices):
                v = c.get("value") if isinstance(c, dict) else None
                if not isinstance(v, (str, bool)):
                    raise RequestError("choice value must be a string or a boolean", f"{p}.choices[{j}].value")
                key = ("true" if v else "false") if isinstance(v, bool) else v
                if key in criteria:
                    raise RequestError("choice values must be unique (as the model reads them)", f"{p}.choices[{j}].value")
                criteria[key] = c.get("description")
                values.append(v)
            sq["criteria"] = criteria
        elif kind == "score":
            levels = q.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise RequestError("levels must be an array of 2 to 10 levels", f"{p}.levels")
            values, criteria = [], []
            for j, lv in enumerate(levels):
                label = lv.get("label") if isinstance(lv, dict) else None
                if not isinstance(label, str):
                    raise RequestError("level label is required", f"{p}.levels[{j}].label")
                desc = lv.get("description")
                criteria.append(f"{label}: {desc}" if desc else label)
                values.append(label)
            sq["criteria"] = criteria
        else:
            raise RequestError("question type must be predicate, choice or score", f"{p}.type")
        out[f"q{i}"] = sq
        meta.append((kind, name, values))
    images = []
    req = {"model": body["model"], "state": state_of(body["input"], images), "questions": out}
    if images:
        req["images"] = images
    return req, meta


def to_openai(model, res, meta):
    answers = []
    for i, (kind, name, values) in enumerate(meta):
        a = res["answers"][f"q{i}"]
        if kind == "predicate":
            answers.append({"type": "predicate", "name": name, "probability": a["noul"]})
        elif kind == "choice":
            keys = [("true" if v else "false") if isinstance(v, bool) else v for v in values]
            best = keys.index(a["choice"])
            answers.append({"type": "choice", "name": name, "choice": values[best], "confidence": a.get("confidence"),
                            "probabilities": [{"value": v, "probability": a["probabilities"][k]}
                                              for v, k in zip(values, keys)]})
        else:
            answers.append({"type": "score", "name": name, "score": a["score"], "confidence": a.get("confidence"),
                            "probabilities": [{"label": lab, "value": j, "probability": a["probabilities"][str(j)]}
                                              for j, lab in enumerate(values)]})
    n = int((res.get("usage") or {}).get("input_tokens") or 0)
    return {"object": "decision", "model": model, "answers": answers,
            "usage": {"input_tokens": n, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": n}}


_vision, _vision_lock = {}, threading.Lock()


def accepts_images(upstream, model, auth):
    """Whether llama-swap lists `model` with image input (architecture.input_modalities), cached for 60 s."""
    with _vision_lock:
        hit = _vision.get(model)
        if hit and time.monotonic() - hit[0] < 60:
            return hit[1]
    req = urllib.request.Request(upstream + "/v1/models", headers={"Authorization": auth} if auth else {})
    models = json.load(urllib.request.urlopen(req, timeout=30)).get("data", [])
    with _vision_lock:
        for m in models:
            _vision[m.get("id")] = (time.monotonic(), "image" in ((m.get("architecture") or {}).get("input_modalities") or []))
        return _vision.get(model, (0, False))[1]


def error(message, kind="invalid_request_error", param=None, code=None):
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s " + fmt, self.headers.get("X-Real-IP") or self.client_address[0], *args)

    def reply(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        self.reply(404, error("not found", "not_found_error"))

    def do_POST(self):
        if self.path.split("?")[0] not in ("/v1/decisions", "/decisions"):
            return self.reply(404, error("not found", "not_found_error"))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            upstream, meta = to_systemone(body)
        except json.JSONDecodeError as e:
            return self.reply(400, error(f"invalid JSON: {e}"))
        except RequestError as e:
            return self.reply(400, error(str(e), param=e.param))
        headers = {"Content-Type": "application/json"}
        for h in ("Authorization", "X-Forwarded-For", "X-Real-IP"):   # key check and client IP stay llama-swap's
            if self.headers.get(h):
                headers[h] = self.headers[h]
        if upstream.get("images"):
            try:
                if not accepts_images(self.server.upstream, upstream["model"], headers.get("Authorization")):
                    return self.reply(400, error(f"model {upstream['model']} does not accept image input", param="input"))
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                return self.reply(502, error(f"upstream unavailable: {e}", "server_error"))
        req = urllib.request.Request(self.server.upstream + "/v1/systemone", method="POST", headers=headers,
                                     data=json.dumps(upstream, ensure_ascii=False).encode())
        try:
            res = json.load(urllib.request.urlopen(req, timeout=1800))
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                detail = json.loads(raw)
                detail = detail.get("error", detail)
                msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            except (ValueError, AttributeError):
                msg = raw.strip() or e.reason
            kind = {401: "authentication_error", 404: "not_found_error"}.get(e.code, "invalid_request_error"
                                                                                   if e.code < 500 else "server_error")
            return self.reply(e.code, error(f"upstream: {msg}", kind, code=e.code))
        except (urllib.error.URLError, TimeoutError) as e:
            return self.reply(502, error(f"upstream unavailable: {e}", "server_error"))
        try:
            out = to_openai(body["model"], res, meta)
        except (KeyError, ValueError, TypeError) as e:
            return self.reply(502, error(f"unexpected upstream answer: {e!r}", "server_error"))
        self.reply(200, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--upstream", default="http://127.0.0.1:8081", help="llama-swap")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [decisions] %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.upstream = args.upstream.rstrip("/")
    log.info("listening on http://%s:%d -> %s/v1/systemone", args.host, args.port, server.upstream)
    server.serve_forever()


if __name__ == "__main__":
    main()
