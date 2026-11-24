"""Install the machine-specific part of the serving deployment: rendered templates, env file, API key.

This checkout is the deployment: clone it to ~/llm-serving and the scripts, llama-swap config (config.yaml,
conf.d/*.yaml with ${env.NAME} macros) and the Caddyfile ({$NAME}) are already where they run
(docs/serving-restore.md). What differs per machine is rendered here from a machine-local values JSON that is
never committed (default .local/serving-values.json):
  env       <checkout>/.local/serving.env (mode 0600): LLM_SERVING, LLM_MODELS, STRATA_RELEASE, HY4_RELEASE,
            LLM_TRUSTED_PROXY, H3_VIDEO_SHARE_BASE, H3_VIDEO_FETCH_HOSTS; sourced by platform/llama-swap/env.sh, read by
            Caddy --envfile
  template  the two Strata engine configs (services/text/strata/configs/*.json.in -> *.json), the Hy4-Flash engine
            config (services/text/hy4-flash/configs/*.json.in) and the LaunchAgents
            (platform/launchd/*.plist.in -> ~/Library/LaunchAgents); the gpu-lock watchdog plist only with --render
  generated ~/llm-serving/platform/llama-swap/.api_key: random, mode 0600, only when absent; never overwritten or printed
  verbatim  bin/ scripts and the llama-swap / gateway files: "in place" when this checkout is <home>/llm-serving,
            copied when installing into another home (tests, a second checkout)

A missing file is created. An existing file that differs is only reported unless --update is given; then it is
copied to <home>/llm-serving/.backup/<relative path>.<time>-install-serving first. --check writes nothing and
exits 1 when anything is missing or differs. Never loads, unloads or restarts anything (LaunchAgents are written,
and the launchctl command to load them is printed) and needs no GPU.
"""
import argparse
import difflib
import ipaddress
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
DEFAULT_VALUES = ROOT / ".local/serving-values.json"
S, LA = "llm-serving/", "Library/LaunchAgents/"
KEY = S + "platform/llama-swap/.api_key"
ENV_FILE = S + ".local/serving.env"
SERVING_ENV = """\
# Written by tools/install_serving.py from .local/serving-values.json; machine-local, never committed.
LLM_SERVING=@SERVING_ROOT@
LLM_MODELS=@MODELS@
STRATA_RELEASE=@STRATA_RELEASE@
HY4_RELEASE=@HY4_RELEASE@
LLM_TRUSTED_PROXY=@TRUSTED_PROXY@
H3_VIDEO_SHARE_BASE=@SHARE_BASE@
H3_VIDEO_FETCH_HOSTS=@FETCH_HOSTS@
"""
VERBATIM = [f"bin/{n}" for n in ("archive-restore", "gpu-lock", "gpu-lock-run", "gpu-lock-watchdog", "gpu-pool",
                                 "gpu-top", "health-check", "flashnext-prefetch", "lsw-apply", "lsw-inflight", "memguard", "model-archive",
                                 "regress")] \
    + [f"platform/llama-swap/{n}" for n in ("serve.sh", "start.sh", "stop.sh")]
VERBATIM_DATA = ["platform/llama-swap/env.sh", "platform/llama-swap/config.yaml", "platform/gateway/Caddyfile"] \
    + sorted(f"platform/llama-swap/conf.d/{p.name}" for p in (ROOT / "platform/llama-swap/conf.d").glob("*.yaml"))
STRATA = "services/text/strata/configs/"
HY4 = "services/text/hy4-flash/configs/"
AGENTS = ("llama-server", "llm-gateway", "decisions", "h3-video-files", "health", "flashnext-prefetch")
# destination relative to the target home -> (source relative to this checkout, kind, mode)
FILES = {S + p: (p, "verbatim", 0o755) for p in VERBATIM}
FILES.update({S + p: (p, "verbatim", 0o644) for p in VERBATIM_DATA})
FILES.update({S + STRATA + n: (STRATA + n + ".in", "template", 0o644)
              for n in ("strata-ad427-c262k-p2-compact.json", "strata-uncensored-c262k-p2-compact.json")})
FILES.update({S + HY4 + n: (HY4 + n + ".in", "template", 0o644) for n in ("hy4-flash-sq64-c128k.json",)})
FILES.update({LA + f"com.local.{n}.plist": (f"platform/launchd/com.local.{n}.plist.in", "template", 0o644)
              for n in AGENTS})
FILES[ENV_FILE] = (None, "env", 0o600)
# Rendered only with --render: the gpu-lock watchdog stays disabled (user decision 2026-11-15), so its plist is never
# written where launchd would load it.
RENDER_ONLY = {LA + "com.local.gpu-lock-watchdog.plist": ("platform/launchd/com.local.gpu-lock-watchdog.plist.in",
                                                           "template", 0o644)}
PATH_VALUES = ("MODELS", "STRATA_RELEASE", "STRATA_EXE", "STRATA_VISION_EXE", "HY4_RELEASE")
VALUE_NAMES = PATH_VALUES + ("TRUSTED_PROXY", "SHARE_BASE", "FETCH_HOSTS")
PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")
# Values go unquoted into llama-swap `cmd` lines, JSON strings, plists and the env file: plain characters only.
SAFE_PATH = re.compile(r"/[A-Za-z0-9_./+-]*")
SAFE_URL = re.compile(r"https?://[A-Za-z0-9.-]+(?::\d+)?(?:/[A-Za-z0-9_./-]*)?")
SAFE_HOSTS = re.compile(r"(?:[A-Za-z0-9.-]+(?:,[A-Za-z0-9.-]+)*)?")


def serving_root(home):
    return Path(home) / "llm-serving"


def default_values(home):
    """Values for a machine restored from this repo alone: Strata built by tools/native.py, weights in ~/models,
    no trusted proxy in front of the gateway. HY4_RELEASE (the release whose build-hy4/bin/strata-hy4-metal and
    server.py run Hy4-Flash) is not here: when the values file does not name it, it is STRATA_RELEASE."""
    lock = json.loads((ROOT / "native.lock.json").read_text())
    release = ROOT / ".local/src" / f"strata-{lock['sources']['strata']['revision'][:12]}"
    return {"MODELS": str(Path(home) / "models"), "STRATA_RELEASE": str(release),
            "STRATA_EXE": str(release / "build-metal-serving/strata"),
            "STRATA_VISION_EXE": str(release / "build-vision/bin/strata-vision"),
            "TRUSTED_PROXY": "127.0.0.1", "SHARE_BASE": "http://127.0.0.1:8080", "FETCH_HOSTS": ""}


def load_values(home, path=None):
    """Defaults, overridden by the values file (when present); "~" means the target home."""
    values = default_values(home)
    path = Path(path) if path else DEFAULT_VALUES
    if path.is_file():
        given = json.loads(path.read_text())
        unknown = set(given) - set(VALUE_NAMES)
        if unknown:
            raise ValueError(f"unknown keys in {path.name}: {', '.join(sorted(unknown))}")
        values.update(given)
    elif path != DEFAULT_VALUES:
        raise FileNotFoundError(f"values file not found: {path}")
    values.setdefault("HY4_RELEASE", values["STRATA_RELEASE"])
    out = {}
    for name, value in values.items():
        if not isinstance(value, str) or (not value and name != "FETCH_HOSTS"):
            raise ValueError(f"{name} must be a non-empty string")
        if name == "SHARE_BASE":
            if not SAFE_URL.fullmatch(value):
                raise ValueError(f"{name} must be an http(s) URL without query or special characters")
            out[name] = value.rstrip("/")
            continue
        if name == "FETCH_HOSTS":
            if not SAFE_HOSTS.fullmatch(value):
                raise ValueError(f"{name} must be comma-separated host names")
            out[name] = value
            continue
        if name == "TRUSTED_PROXY":
            try:
                out[name] = str(ipaddress.IPv4Address(value))
            except ValueError:
                raise ValueError(f"{name} must be one IPv4 address") from None
            continue
        if value == "~" or value.startswith("~/"):
            value = str(Path(home)) + value[1:]
        if not os.path.isabs(value):
            raise ValueError(f"{name} must be an absolute path or start with ~/")
        out[name] = value.rstrip("/")
    out["SERVING_ROOT"] = str(serving_root(home))
    for name in PATH_VALUES + ("SERVING_ROOT",):
        if not SAFE_PATH.fullmatch(out[name]):
            raise ValueError(f"{name} contains characters other than letters, digits and _ . / + -")
    return out


def render_text(text, values):
    def sub(m):
        if m[1] not in values:
            raise KeyError(f"unknown placeholder @{m[1]}@")
        return values[m[1]]
    return PLACEHOLDER.sub(sub, text)


def rendered(values, files=None):
    """{destination relative to home: (bytes, mode)} for every installed file."""
    out = {}
    for dest, (src, kind, mode) in (files or FILES).items():
        if kind == "env":
            data = render_text(SERVING_ENV, values).encode()
        else:
            data = (SOURCE / src).read_bytes()
            if kind == "template":
                data = render_text(data.decode(), values).encode()
        out[dest] = (data, mode)
    return out


def in_place(home, rel):
    """True when the destination is this checkout's own file (the normal case: the checkout is ~/llm-serving)."""
    src = FILES.get(rel, (None, None, None))[0]
    if FILES.get(rel, (None, "x"))[1] != "verbatim":
        return False
    dest = Path(home) / rel
    return dest.exists() and dest.resolve() == (SOURCE / src).resolve()


def state(path, data, mode):
    if not path.is_file():
        return "missing"
    if path.read_bytes() != data:
        return "differs"
    if (path.stat().st_mode & 0o777) != mode:
        return "mode"
    return "unchanged"


def key_state(path):
    """Presence and mode only; the key itself is never read."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    mode = path.stat().st_mode & 0o777
    return "ok" if mode == 0o600 else f"mode {mode:o}"


def write_atomic(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def backup(path, home):
    """Copy to ~/llm-serving/.backup/<path relative to ~/llm-serving, or to home outside it>.<time>-install-serving."""
    root = serving_root(home)
    rel = path.relative_to(root) if path.is_relative_to(root) else path.relative_to(home)
    base = root / ".backup" / rel.parent / f"{rel.name}.{time.strftime('%Y%m%d-%H%M%S')}-install-serving"
    dest, n = base, 1
    while dest.exists():
        n += 1
        dest = base.with_name(f"{base.name}-{n}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def make_key(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32) + "\n")


def launchctl_hint(path, replaced):
    label = path.stem
    steps = f"launchctl bootout gui/$(id -u)/{label}; " if replaced else ""
    return f"to load: {steps}launchctl bootstrap gui/$(id -u) {path}"


def install(home, values, mode="install", update=False, show_diff=False, log=print):
    """mode: install | dry-run | check. Returns the number of problems (check) or items left alone (install)."""
    home = Path(home)
    problems = 0
    for rel, (data, fmode) in rendered(values).items():
        path = home / rel
        if in_place(home, rel):
            log(f"in place   {rel}")
            continue
        s = state(path, data, fmode)
        if s == "unchanged":
            log(f"unchanged  {rel}")
            continue
        if mode == "check":
            problems += 1
            log(f"{s:<10} {rel}")
            if s == "differs" and show_diff:
                log("".join(difflib.unified_diff(path.read_text(errors="replace").splitlines(True),
                                                 data.decode().splitlines(True), str(path), "rendered")))
            continue
        if s == "differs" and not update:
            problems += 1
            log(f"drift      {rel} (differs from the repo; kept; --update replaces it after a backup)")
            continue
        action = {"missing": "create", "differs": "update", "mode": "chmod"}[s]
        if mode == "dry-run":
            log(f"would {action:<6} {rel}" + (" (backup first)" if s == "differs" else ""))
            continue
        if s == "mode":
            os.chmod(path, fmode)
        else:
            saved = backup(path, home) if s == "differs" else None
            write_atomic(path, data, fmode)
            if saved:
                log(f"backup     {saved}")
        log(f"{action + 'd':<10} {rel}")
        if rel.startswith(LA) and s != "mode":
            log("           " + launchctl_hint(path, s == "differs"))
    key = home / KEY
    k = key_state(key)
    if k == "ok":
        log(f"present    {KEY} (mode 600)")
    elif mode == "check":
        problems += 1
        log(f"{k:<10} {KEY}")
    elif k == "missing":
        if mode == "dry-run":
            log(f"would generate {KEY} (mode 600)")
        else:
            make_key(key)
            log(f"generated  {KEY} (mode 600)")
    elif k == "empty":
        problems += 1
        log(f"empty      {KEY}: fill it locally (not overwritten)")
    else:
        if mode == "dry-run":
            log(f"would chmod 600 {KEY} ({k})")
        else:
            os.chmod(key, 0o600)
            log(f"chmodded   {KEY} to 600 (content kept)")
    return problems


def doctor(home=None, values_path=None, log=print):
    """Read-only report for tools/doctor.py: OK/WARN lines, never fails the deployment check."""
    home = Path(home or Path.home())
    try:
        files = rendered(load_values(home, values_path))
    except (OSError, ValueError, KeyError) as e:
        log(f"WARN serving glue: {e}")
        return
    for rel, (data, fmode) in files.items():
        if in_place(home, rel):
            continue
        s = state(home / rel, data, fmode)
        log(("OK" if s == "unchanged" else "WARN") + f" serving ~/{rel}: "
            + ("matches the repo" if s == "unchanged" else s + "; see tools/install_serving.py --check"))
    k = key_state(home / KEY)
    log(("OK" if k == "ok" else "WARN") + f" llama-swap key ~/{KEY}: " + ("present, mode 600" if k == "ok" else k))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", type=Path, default=Path.home(),
                   help="target home: files go to HOME/llm-serving and HOME/Library/LaunchAgents")
    p.add_argument("--values", type=Path, help=f"machine-local values JSON (default {DEFAULT_VALUES.relative_to(ROOT)})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    g.add_argument("--check", action="store_true", help="write nothing; exit 1 if anything is missing or differs")
    g.add_argument("--render", type=Path, metavar="DIR",
                   help="only render the templates and the env file into DIR (paths relative to home); no scripts, no key")
    p.add_argument("--update", action="store_true", help="replace differing files (after a backup)")
    p.add_argument("--diff", action="store_true", help="with --check: show unified diffs of differing files")
    a = p.parse_args(argv)
    home = a.home.expanduser().absolute()
    try:
        values = load_values(home, a.values)
    except (OSError, ValueError) as e:
        p.error(str(e))
    if a.render:
        templates = {rel: v for rel, v in {**FILES, **RENDER_ONLY}.items() if v[1] in ("template", "env")}
        for rel, (data, fmode) in rendered(values, templates).items():
            write_atomic(a.render / rel, data, fmode)
            print(f"rendered   {rel}")
        return 0
    mode = "check" if a.check else "dry-run" if a.dry_run else "install"
    print(f"{mode}: {home}")
    problems = install(home, values, mode, update=a.update, show_diff=a.diff)
    if mode == "check":
        print("OK: installed serving glue matches the repo" if not problems else f"{problems} difference(s)")
        return 1 if problems else 0
    if problems:
        print(f"WARN: {problems} item(s) left as they are", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
