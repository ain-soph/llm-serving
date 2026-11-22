"""Check deployment inputs without starting models or changing system services."""
import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path

import install_agents
import llm_collab_bridge
from service import environment

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / ".local/config/machine.json")
    p.add_argument("--skip-models", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    errors = []
    for name in ("mlx.core", "mlx_lm", "h3_image_mlx.api", "mflux", "mlx_audio", "torchvision", "mlx_whisper", "mlx_qwen3_asr", "numpy", "soundfile",
                 "jinja2", "regex", "yaml", "requests", "psutil"):
        try:
            importlib.import_module(name)
            print(f"OK Python import: {name}")
        except Exception as e:
            errors.append(f"Python import {name}: {type(e).__name__}")
    for name, path in c["binaries"].items():
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            errors.append(f"Missing executable: {name}")
        else:
            print(f"OK executable: {name}")
    key = Path(c["state"]) / "api_key"
    if not key.is_file() or not key.read_text().strip() or key.stat().st_mode & 0o077:
        errors.append("API key must be nonempty and private (mode 0600)")
    env = dict(environment(c), LLAMA_API_KEY="configuration-validation")
    b, config = c["binaries"], Path(c["config"])
    checks = [([b["llama-swap"], "-config", str(config / "config.yaml"), "-config-dir", str(config / "conf.d"),
                "-validate"], "llama-swap config"),
              ([b["caddy"], "validate", "--config", str(config / "Caddyfile"), "--adapter", "caddyfile"], "Caddy config"),
              ([b["ffmpeg"], "-hide_banner", "-h", "encoder=libx264"], "FFmpeg libx264")]
    for argv, name in checks:
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
    if (ROOT / "decisions/OneJev/qev/engine.py").is_file():
        print("OK OneJev checkout")
    else:
        errors.append("Missing decisions/OneJev checkout (elephant-1); see docs/decisions.md")
    if not a.skip_models:
        required = json.loads((ROOT / "models.required.json").read_text())
        for name in required["directories"]:
            if not (Path(c["models"]) / name).is_dir():
                errors.append(f"Missing model directory: {name}")
        for name in required["files"]:
            if not (Path(c["models"]) / name).is_file():
                errors.append(f"Missing model file: {name}")
    # Agent rules and the local worker are optional for serving: report drift, do not fail.
    try:
        claude = install_agents.install_claude_md(Path.home() / ".claude/CLAUDE.md", None, Path.home(), check=True)[0]
        print(("OK" if claude == "unchanged" else "WARN") + f" agent rules in ~/.claude/CLAUDE.md: {claude}"
              + ("" if claude == "unchanged" else "; run tools/install_agents.py"))
    except ValueError as e:
        print(f"WARN agent rules in ~/.claude/CLAUDE.md: {e}")
    worker = install_agents.link_worker(Path.home() / "llm-serving/platform/agent-bridge/local-worker", check=True)
    print(("OK" if worker == "unchanged" else "WARN") + f" local worker link: {worker}")
    for ok, message in llm_collab_bridge.status(Path.home()):
        print(("OK " if ok else "WARN ") + message + ("" if ok else "; run tools/install_agents.py"))
    if errors:
        for message in errors:
            print("FAIL", message)
        raise SystemExit(1)
    print("Deployment inputs checked. No inference was run.")


if __name__ == "__main__":
    main()
