#!/usr/bin/python3
"""Summarize one local-worker run as JSON (stdout): harness events, engine speed lines, network attempts, git diff.

Counts come from the harness's own event stream (Claude Code stream-json or Codex exec --json) and from the Strata
engine log written during the run ("prompt P tokens = R reused + N read in X ms (pp tok/s), G generated in Y ms
(tg tok/s)" per request). The worker's own claims are kept only as `final_message`: verify the work yourself.
"""
import argparse
import json
import os
import re
import subprocess

ENGINE_REQUEST = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<new>\d+)(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), (?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")
# Tool results that mean the model's call itself was malformed (unknown tool, bad/missing arguments), not that a
# well-formed command failed.
FORMAT_ERR = re.compile(r"InputValidationError|invalid (tool|input|arguments)|unknown tool|No such tool|"
                        r"failed to parse|is not valid JSON|required parameter|missing (required )?(field|parameter)",
                        re.I)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"_unparsed": line[:300]})
    return out


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def claude_stats(events):
    msg_ids, tools, errors, fmt_errors, final, result = [], {}, [], [], "", {}
    for e in events:
        t = e.get("type")
        if t == "assistant":
            m = e.get("message", {})
            if m.get("id") and m["id"] not in msg_ids:
                msg_ids.append(m["id"])
            for c in m.get("content", []) or []:
                if c.get("type") == "tool_use":
                    tools[c.get("name")] = tools.get(c.get("name"), 0) + 1
                elif c.get("type") == "text" and c.get("text", "").strip():
                    final = c["text"]
        elif t == "user":
            for c in (e.get("message", {}).get("content") or []):
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    txt = text_of(c.get("content"))[:300]
                    errors.append(txt)
                    if FORMAT_ERR.search(txt):
                        fmt_errors.append(txt)
        elif t == "result":
            result = e
    return {
        "model_responses": len(msg_ids),
        "num_turns": result.get("num_turns"),
        "tool_calls": sum(tools.values()), "tool_calls_by_name": tools,
        "tool_errors": len(errors), "tool_error_samples": errors[:8],
        "tool_format_errors": len(fmt_errors), "tool_format_error_samples": fmt_errors[:5],
        "permission_denials": len(result.get("permission_denials") or []),
        "harness_result": {k: result.get(k) for k in ("subtype", "is_error", "duration_ms", "duration_api_ms",
                                                      "terminal_reason", "stop_reason")},
        "usage": result.get("usage", {}) and {k: result["usage"].get(k) for k in ("input_tokens", "output_tokens",
                                                                                 "cache_read_input_tokens")},
        "final_message": (result.get("result") or final)[:4000],
    }


def codex_stats(events):
    tools, errors, fmt_errors, final, usage, turns, errs = {}, [], [], "", {}, 0, []
    for e in events:
        t = e.get("type", "")
        if t == "turn.completed":
            turns += 1
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        elif t in ("error", "turn.failed"):
            errs.append(json.dumps(e)[:300])
        elif t == "item.completed":
            it = e.get("item", {})
            kind = it.get("type") or it.get("item_type")
            if kind == "agent_message":
                final = it.get("text", "") or final
            elif kind in ("command_execution", "file_change", "mcp_tool_call", "web_search", "patch_apply"):
                tools[kind] = tools.get(kind, 0) + 1
                failed = it.get("status") == "failed" or (kind == "command_execution" and it.get("exit_code") not in (0, None))
                if failed:
                    txt = (it.get("aggregated_output") or json.dumps(it))[-300:]
                    errors.append(txt)
                    if FORMAT_ERR.search(txt):
                        fmt_errors.append(txt)
            elif kind == "error":
                txt = it.get("message", "")[:300]
                if txt.startswith("Model metadata for"):    # harmless: Codex does not know the local model
                    continue
                errs.append(txt)
                if FORMAT_ERR.search(txt):
                    fmt_errors.append(txt)
    return {
        "model_responses": None, "num_turns": turns,
        "tool_calls": sum(tools.values()), "tool_calls_by_name": tools,
        "tool_errors": len(errors), "tool_error_samples": errors[:8],
        "tool_format_errors": len(fmt_errors), "tool_format_error_samples": fmt_errors[:5],
        "harness_errors": errs[:8], "usage": usage, "final_message": final[:4000],
    }


def engine_stats(path, offset):
    reqs = []
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            f.seek(int(offset or 0))
            for line in f.read().decode("utf-8", "replace").splitlines():
                m = ENGINE_REQUEST.search(line)
                if m:
                    d = {k: float(v) for k, v in m.groupdict().items()}
                    reqs.append(d)
    if not reqs:
        return {"requests": 0}
    new = sum(r["new"] for r in reqs)
    read_ms = sum(r["read"] for r in reqs)
    gen = sum(r["gen"] for r in reqs)
    gen_ms = sum(r["gen_ms"] for r in reqs)
    big = [r for r in reqs if r["new"] >= 1024]
    return {
        "requests": len(reqs),
        "prompt_tokens_total": int(sum(r["prompt"] for r in reqs)), "prompt_tokens_new": int(new),
        "prompt_tokens_reused": int(sum(r["reused"] for r in reqs)),
        "prompt_first": int(reqs[0]["prompt"]), "prompt_max": int(max(r["prompt"] for r in reqs)),
        "generated_tokens": int(gen),
        "prefill_tok_s_aggregate": round(new / read_ms * 1000, 1) if read_ms else None,
        "prefill_tok_s_requests_ge_1024_new": round(sum(r["new"] for r in big) / sum(r["read"] for r in big) * 1000, 1)
        if big else None,
        "decode_tok_s_aggregate": round(gen / gen_ms * 1000, 1) if gen_ms else None,
        "decode_tok_s_min_max": [min(r["tg"] for r in reqs), max(r["tg"] for r in reqs)],
        "engine_busy_s": round((read_ms + gen_ms) / 1000, 1),
    }


def git_state(cwd):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return ""
    if not run("rev-parse", "--is-inside-work-tree").strip():
        return None
    return {"status": run("status", "--short").splitlines()[:50], "diffstat": run("diff", "--stat", "HEAD").strip()[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    for a in ("harness", "variant", "model", "cwd", "run-dir", "status", "engine-log"):
        ap.add_argument("--" + a, default="")
    for a in ("engine-offset", "t-start", "t-lock", "t-ready", "t-end"):
        ap.add_argument("--" + a, type=int, default=0)
    a = ap.parse_args()
    code, _, state = a.status.partition(":")
    events = read_jsonl(os.path.join(a.run_dir, "events.jsonl"))
    stats = claude_stats(events) if a.harness == "claude" else codex_stats(events)
    if a.harness == "codex" and not stats["final_message"]:
        p = os.path.join(a.run_dir, "last-message.txt")
        if os.path.exists(p):
            stats["final_message"] = open(p, errors="replace").read()[:4000]
    net = read_jsonl(os.path.join(a.run_dir, "net-attempts.jsonl"))
    eng = engine_stats(a.engine_log, a.engine_offset)
    if stats.get("model_responses") is None:
        stats["model_responses"] = eng.get("requests")
    out = {
        "status": state or "unknown", "exit_code": int(code or 0),
        "harness": a.harness, "variant": a.variant, "model": a.model, "cwd": a.cwd, "run_dir": a.run_dir,
        "seconds": {"wait_for_gpu": (a.t_lock - a.t_start) if a.t_lock else None,
                    "model_ready": (a.t_ready - a.t_lock) if a.t_ready and a.t_lock else None,
                    "harness": (a.t_end - a.t_ready) if a.t_ready else None,
                    "total": a.t_end - a.t_start},
        **stats,
        "engine": eng,
        "external_network_attempts": len(net), "external_network_samples": net[:10],
        "unparsed_event_lines": sum(1 for e in events if "_unparsed" in e),
        "git": git_state(a.cwd) if a.cwd else None,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
