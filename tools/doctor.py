"""Check the deployment (this checkout is ~/llm-serving) without starting models or changing system services."""
import argparse
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import install_agents
import install_serving
import llm_collab_bridge

ROOT = Path(__file__).resolve().parents[1]
LSW = ROOT / "platform/llama-swap"
# The serving glue's archive-restore: its manifest checks decide "archived".
ARCHIVE_RESTORE = Path(os.environ.get("ARCHIVE_RESTORE", ROOT / "bin/archive-restore"))
SEARCH_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


def serving_env():
    """The environment llama-swap runs with: platform/llama-swap/env.sh (machine values, GPU pool)."""
    r = subprocess.run(["/bin/zsh", "-c", 'source "$1" >/dev/null && env -0', "doctor", str(LSW / "env.sh")],
                       env=dict(os.environ, LLM_SERVING=str(ROOT)), capture_output=True, timeout=60)
    if r.returncode:
        raise SystemExit(f"FAIL platform/llama-swap/env.sh: {r.stderr.decode(errors='replace')[-300:]}")
    return dict(item.split("=", 1) for item in r.stdout.decode().split("\0") if "=" in item)


def archived(models, name):
    """Why <models>/<name> may be absent: a manifest <name>.archive exists and the local copy is missing or
    incomplete (same rule as bin/regress, using archive-restore's load_manifest / problems). None otherwise."""
    shortcut, local = models / f"{name}.archive", models / name
    if not shortcut.is_file():
        return None
    if not os.path.lexists(local):
        return "missing"
    if not ARCHIVE_RESTORE.is_file():   # without the tool only "missing" can be told apart
        return None
    import importlib.util
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("archive_restore", str(ARCHIVE_RESTORE))
    spec = importlib.util.spec_from_loader("archive_restore", loader)
    ar = importlib.util.module_from_spec(spec)
    loader.exec_module(ar)
    bad = ar.problems(local, ar.load_manifest(shortcut)[2])
    return f"incomplete ({len(bad)}: {', '.join(bad[:2])})" if bad else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-models", action="store_true")
    a = p.parse_args()
    # Rendered templates, .local/serving.env, llama-swap key (tools/install_serving.py --check): WARN only.
    install_serving.doctor()
    errors = []
    for name in ("mlx.core", "mlx_lm", "h3_image_mlx.api", "mflux", "mlx_audio", "torchvision", "mlx_whisper", "mlx_qwen3_asr", "numpy", "soundfile",
                 "jinja2", "regex", "yaml", "requests", "psutil"):
        try:
            importlib.import_module(name)
            print(f"OK Python import: {name}")
        except Exception as e:
            errors.append(f"Python import {name}: {type(e).__name__}")
    env_file = ROOT / ".local/serving.env"
    if not env_file.is_file():
        raise SystemExit("FAIL missing .local/serving.env; run tools/install_serving.py first")
    env = dict(serving_env(), LLAMA_API_KEY="configuration-validation")
    search = env.get("PATH", "") + ":" + SEARCH_PATH
    binaries = {name: shutil.which(name, path=search) for name in ("llama-swap", "caddy", "ffmpeg", "llama-server")}
    values = install_serving.load_values(Path.home())
    binaries.update({"strata": values["STRATA_EXE"], "strata-vision": values["STRATA_VISION_EXE"],
                     "strata-hy4-metal": values["HY4_RELEASE"] + "/build-hy4/bin/strata-hy4-metal"})
    for name, path in binaries.items():
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            errors.append(f"Missing executable: {name}")
        else:
            print(f"OK executable: {name}")
    key = LSW / ".api_key"
    if not key.is_file() or not key.read_text().strip() or key.stat().st_mode & 0o077:
        errors.append("API key platform/llama-swap/.api_key must be nonempty and private (mode 0600)")
    checks = [([binaries["llama-swap"], "-config", str(LSW / "config.yaml"), "-config-dir", str(LSW / "conf.d"),
                "-validate"], "llama-swap config"),
              ([binaries["caddy"], "validate", "--config", str(ROOT / "platform/gateway/Caddyfile"), "--adapter",
                "caddyfile", "--envfile", str(env_file)], "Caddy config"),
              ([binaries["ffmpeg"], "-hide_banner", "-h", "encoder=libx264"], "FFmpeg libx264")]
    for argv, name in checks:
        if not argv[0]:
            continue
        try:
            r = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=60)
            missing_encoder = name == "FFmpeg libx264" and "Encoder libx264" not in r.stdout
            if r.returncode or missing_encoder:
                errors.append(f"{name} failed: {(r.stderr or r.stdout)[-500:]}")
            else:
                print(f"OK {name}")
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"{name}: {type(e).__name__}")
    # elephant-1 imports the official OneJev engine from a checkout next to its server (docs/decisions.md)
    if (ROOT / "services/decisions/OneJev/qev/engine.py").is_file():
        print("OK OneJev checkout")
    else:
        errors.append("Missing services/decisions/OneJev checkout (elephant-1); see docs/decisions.md")
    if not a.skip_models:
        required = json.loads((ROOT / "models.required.json").read_text())
        models = Path(env["LLM_MODELS"])
        cache = {}
        for kind, names, present in (("directory", required["directories"], Path.is_dir),
                                     ("file", required["files"], Path.is_file)):
            for name in names:
                top = name.split("/")[0]   # the unit that is archived: the top-level model directory
                if top not in cache:
                    cache[top] = archived(models, top)
                if cache[top]:   # archived on purpose (<models>/<top>.archive): its entry restores it on demand
                    print(f"SKIP(archived) model {kind}: {name} ({top} {cache[top]})")
                elif not present(models / name):
                    errors.append(f"Missing model {kind}: {name}")
    # Agent rules and the local worker are optional for serving: report drift, do not fail.
    try:
        claude = install_agents.install_claude_md(Path.home() / ".claude/CLAUDE.md", None, Path.home(), check=True)[0]
        print(("OK" if claude == "unchanged" else "WARN") + f" agent rules in ~/.claude/CLAUDE.md: {claude}"
              + ("" if claude == "unchanged" else "; run tools/install_agents.py"))
    except ValueError as e:
        print(f"WARN agent rules in ~/.claude/CLAUDE.md: {e}")
    worker = install_agents.link_worker(ROOT / "platform/agent-bridge/local-worker", check=True)
    print(("OK" if worker in ("unchanged", "in place") else "WARN") + f" local worker: {worker}")
    for ok, message in llm_collab_bridge.status(Path.home()):
        print(("OK " if ok else "WARN ") + message + ("" if ok else "; run tools/install_agents.py"))
    if errors:
        for message in errors:
            print("FAIL", message)
        raise SystemExit(1)
    print("Deployment inputs checked. No inference was run.")


if __name__ == "__main__":
    main()
