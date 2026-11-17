"""Render host-local configuration; never edit checked-in files with host paths."""
import argparse
import ipaddress
import json
import os
import platform
import plistlib
import re
import secrets
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LABELS = {name: f"app.llm-serving.{name}" for name in ("router", "gateway", "files")}


def sysctl(key, default=""):
    p = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else default


def hardware():
    return {"system": platform.system(), "arch": platform.machine(),
            "macos": platform.mac_ver()[0], "chip": sysctl("machdep.cpu.brand_string"),
            "memory_mb": int(sysctl("hw.memsize", "0")) // (1024 * 1024),
            "wired_limit_mb": int(sysctl("iogpu.wired_limit_mb", "0"))}


def executable(name, explicit=None):
    if explicit:
        p = Path(explicit).expanduser().absolute()
    else:
        managed = ROOT / ".local/bin" / name
        p = managed if managed.exists() else Path(shutil.which(name) or managed)
    return str(p)


def replace(value, mapping, shell=False):
    if isinstance(value, str):
        return re.sub(r"@([A-Z_]+)@", lambda m: shlex.quote(mapping[m[1]]) if shell else mapping[m[1]], value)
    if isinstance(value, list):
        return [replace(v, mapping, shell) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, mapping, shell or k in ("cmd", "macros")) for k, v in value.items()}
    return value


def load_template(path):
    # Unquoted @ placeholders are not YAML scalars. Quote just scalar lines;
    # folded command strings and strings inside double quotes stay untouched.
    text = path.read_text()
    text = re.sub(r"^(\s*path:\s*)(@[^\n]+)$", lambda m: m[1] + json.dumps(m[2]), text, flags=re.M)
    # cmd can also be a single, unfolded command.
    text = re.sub(r"^(\s*cmd:\s*)(@[^\n]+)$", lambda m: m[1] + json.dumps(m[2]), text, flags=re.M)
    return yaml.safe_load(text)


def make_caddy(config):
    trusted = ""
    if config["trusted_proxies"]:
        trusted = "servers {\n trusted_proxies static " + " ".join(config["trusted_proxies"]) + "\n client_ip_headers X-Real-IP\n}\n"
    return """{
 admin off
 auto_https off
 persist_config off
 %s
}
(upstream) {
 reverse_proxy 127.0.0.1:8081 {
  flush_interval -1
  header_up X-Forwarded-For {client_ip}
 }
}
(files) {
 forward_auth 127.0.0.1:8081 {
  uri /v1/models
 }
 reverse_proxy 127.0.0.1:8090 {
  flush_interval -1
  header_up X-Forwarded-For {client_ip}
 }
}
%s:8080 {
 @video_render {
  method POST
  path /v1/videos /v1/videos/*
 }
 @video_read {
  path /v1/videos /v1/videos/* /media/*
 }
 handle @video_render {
  rewrite * /upstream/minimax-h3-video{uri}
  import upstream
 }
 handle @video_read {
  import files
 }
 handle {
  import upstream
 }
}
""" % (trusted, config["listen"])


def render(config, output):
    output.mkdir(parents=True, exist_ok=True)
    state = Path(config["state"])
    m = {"ROOT": str(ROOT), "PYTHON": str(ROOT / ".venv/bin/python"),
         "MODELS": config["models"], "STATE": str(state), "CONFIG": str(output),
         "STRATA_SOURCE": config["sources"]["strata"],
         "STRATA_EXE": config["binaries"]["strata"],
         "VISION_EXE": config["binaries"]["strata-vision"],
         "LLAMA_SERVER": config["binaries"]["llama-server"]}
    for p in sorted((ROOT / "templates").rglob("*")):
        if not p.is_file():
            continue
        dest = output / p.relative_to(ROOT / "templates")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            dest.write_text(json.dumps(replace(json.loads(p.read_text()), m), indent=2) + "\n")
        else:
            data = replace(load_template(p), m)
            if p.name == "config.yaml" and not config["preload"]:
                data.pop("hooks", None)
            dest.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (output / "Caddyfile").write_text(make_caddy(config))
    for name, label in LABELS.items():
        doc = {"Label": label, "ProgramArguments": [m["PYTHON"], str(ROOT / "tools/service.py"), name,
                 "--config", str(output / "machine.json")],
               "WorkingDirectory": str(ROOT), "RunAtLoad": True, "KeepAlive": True,
               "ThrottleInterval": 30,
               "StandardOutPath": str(state / "logs" / f"{name}.log"),
               "StandardErrorPath": str(state / "logs" / f"{name}.log")}
        (output / (label + ".plist")).write_bytes(plistlib.dumps(doc))
    (output / "machine.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--state", type=Path, default=Path.home() / "Library/Application Support/llm-serving")
    p.add_argument("--output", type=Path, default=ROOT / ".local/config")
    p.add_argument("--listen", default="127.0.0.1", help="IP address; use 0.0.0.0 for LAN access")
    p.add_argument("--public-base", default="http://127.0.0.1:8080")
    p.add_argument("--fetch-host", action="append", default=[])
    p.add_argument("--trusted-proxy", action="append", default=[])
    p.add_argument("--pool-mb", type=int, help="Measured pool; otherwise use current wired limit minus 4872 MiB")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--strata-source", type=Path)
    p.add_argument("--h3-source", type=Path)
    p.add_argument("--strata-exe")
    p.add_argument("--vision-exe")
    p.add_argument("--nemo-exe")
    for name in ("llama-server", "llama-swap", "caddy", "ffmpeg", "ffprobe"):
        p.add_argument("--" + name)
    a = p.parse_args()
    ipaddress.IPv4Address(a.listen)
    for proxy in a.trusted_proxy:
        ipaddress.ip_network(proxy, strict=False)
    from urllib.parse import urlsplit
    parsed = urlsplit(a.public_base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        p.error("--public-base must be an HTTP(S) URL without credentials")
    for host in a.fetch_host:
        if not re.fullmatch(r"[A-Za-z0-9.-]+", host):
            p.error("--fetch-host accepts a hostname without a scheme, path or port")
    hw = hardware()
    if hw["system"] != "Darwin" or hw["arch"] != "arm64":
        p.error("This deployment supports macOS on Apple Silicon")
    pool = a.pool_mb if a.pool_mb is not None else hw["wired_limit_mb"] - 4872
    if pool <= 0:
        p.error("No usable wired limit detected; supply --pool-mb after measuring the machine")
    # Do not silently apply measured 96-GiB-M5 settings to a smaller GPU budget.
    if pool < 71000:
        p.error("Current chat configuration needs a pool >=71000 MiB. Use a separately validated smaller-model profile.")
    state = a.state.expanduser().absolute()
    out = a.output.expanduser().absolute()
    for sub in ("logs", "media/videos", "media/images", "strata", "caddy"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    key = state / "api_key"
    if not key.exists():
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    if not key.read_text().strip():
        p.error("Existing api_key is empty; fill it locally before configuring")
    key.chmod(0o600)
    lock = json.loads((ROOT / "native.lock.json").read_text())
    sources = {name: str(ROOT / ".local/src" / f"{name}-{v['revision'][:12]}")
               for name, v in lock["sources"].items()}
    if a.strata_source:
        sources["strata"] = str(a.strata_source.expanduser().absolute())
    if a.h3_source:
        sources["h3"] = str(a.h3_source.expanduser().absolute())
    binaries = {name: executable(name, getattr(a, name.replace("-", "_")))
                for name in ("llama-server", "llama-swap", "caddy", "ffmpeg", "ffprobe")}
    binaries.update({"strata": a.strata_exe or str(Path(sources["strata"]) / "build-metal-serving/strata"),
                     "strata-vision": a.vision_exe or str(Path(sources["strata"]) / "build-vision/bin/strata-vision"),
                     "h3": str(Path(sources["h3"]) / "h3"),
                     "nemo": a.nemo_exe or str(Path(sources["nemo"]) / "build/metal-server/bin/nemo-speech")})
    config = {"root": str(ROOT), "python": str(ROOT / ".venv/bin/python"),
              "models": str(a.models.expanduser().absolute()), "state": str(state), "config": str(out),
              "listen": a.listen, "public_base": a.public_base.rstrip("/"), "fetch_hosts": a.fetch_host,
              "trusted_proxies": a.trusted_proxy, "pool_mb": pool, "preload": a.preload,
              "sources": sources, "binaries": binaries, "hardware": hw}
    render(config, out)
    print(f"Rendered configuration: {out}; credentials stay in the state directory (mode 0600)")
    print("No system services were changed. Run tools/install.py --help for installation.")


if __name__ == "__main__":
    main()
