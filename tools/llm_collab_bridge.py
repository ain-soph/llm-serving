"""Install or update the llm-collab MCP bridge from platform/agent-bridge (used by tools/install_agents.py).

Bridge directory: <home>/llm-serving/platform/agent-bridge.
- release/           ain-soph/llm-collab at the commit pinned in platform/agent-bridge/pins.json
- runtime/upstream/  ain-soph/local-agent-bridge (branch serving) at its pinned commit
  A missing checkout is cloned; a clean one at another commit is fetched and checked out (detached) at the pin;
  one with local changes is reported and left alone. Locked dependencies: `npm ci --ignore-scripts`, skipped when
  node_modules already matches package-lock.json.
- workspace/         default collaboration checkout; only created when missing (clone of a local bare mirror
                     runtime/portable-base.git of this repository, pushes go to a nonexistent path).
- mcp, codex-select, llm-collab.json, llm-collab.env: tracked in platform/agent-bridge, so in the deployment they
  are "in place" (only their mode is set); another home gets copies, and a differing file is backed up under
  <home>/llm-serving/.backup/ (same relative path, .<time>-install-agents suffix) before it is replaced.
- state/ and llm-collab.local.env are never touched.
MCP server "llm-collab" -> <bridge>/mcp is registered through each client's own CLI (Claude Code user scope,
Codex in $CODEX_HOME or <home>/.codex, Antigravity) when the CLI is installed and the entry differs. Current
entries are read from the client config files, not through the CLIs (`claude mcp get` starts the server).
"""
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "platform/agent-bridge"
SERVER = "llm-collab"
# Template name, installed name, mode. The launcher keeps the name of the private wrapper it replaced, so client
# registrations that point at <bridge>/mcp stay valid.
MANAGED = (("mcp", "mcp", 0o700), ("codex-select", "codex-select", 0o700),
           ("llm-collab.json", "llm-collab.json", 0o600), ("llm-collab.env", "llm-collab.env", 0o600))
EXTRA_PATH = ("/opt/homebrew/bin", "/usr/local/bin")


class BridgeError(Exception):
    pass


def bridge_dir(home):
    return Path(home) / "llm-serving/platform/agent-bridge"


def load_pins(sources=None):
    pins = json.loads((TEMPLATES / "pins.json").read_text())
    for name, url in (sources or {}).items():
        if name not in pins:
            raise BridgeError(f"unknown source {name!r}; use one of {', '.join(pins)}")
        pins[name]["url"] = url
    return pins


def _run(args, cwd=None, env=None, timeout=600):
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=env, timeout=timeout,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise BridgeError(f"{' '.join(map(str, args[:4]))} failed ({result.returncode}): {' | '.join(detail)}")
    return result.stdout.strip()


def _git(path, *args, timeout=600):
    return _run(["git", "-C", path, *args], timeout=timeout)


def _has_commit(path, commit):
    return subprocess.run(["git", "-C", path, "cat-file", "-e", f"{commit}^{{commit}}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def sync_checkout(path, pin, apply):
    """Bring one Git checkout to its pinned commit. Returns the action taken (or planned)."""
    path, commit = Path(path), pin["commit"]
    if not path.exists():
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--quiet", "--single-branch", "--branch", pin["branch"], pin["url"], path])
            if not _has_commit(path, commit):
                _git(path, "fetch", "--quiet", pin["url"], commit)
            _git(path, "checkout", "--quiet", "--detach", commit)
        return f"cloned at {commit[:12]}"
    try:
        head = _git(path, "rev-parse", "HEAD")
    except BridgeError as e:
        raise BridgeError(f"{path} exists but is not a Git checkout") from e
    if head == commit:
        return "unchanged"
    if _git(path, "status", "--porcelain", "--untracked-files=no"):
        raise BridgeError(f"{path} has local changes at {head[:12]}; commit or stash them, then rerun")
    if apply:
        if not _has_commit(path, commit):
            _git(path, "fetch", "--quiet", pin["url"], pin["branch"])
        if not _has_commit(path, commit):
            _git(path, "fetch", "--quiet", pin["url"], commit)
        _git(path, "checkout", "--quiet", "--detach", commit)
    return f"updated {head[:12]} -> {commit[:12]}"


def deps_installed(path):
    """True when node_modules matches package-lock.json (npm's hidden lockfile has the same packages)."""
    try:
        locked = json.loads((Path(path) / "package-lock.json").read_text())["packages"]
    except (OSError, ValueError, KeyError):
        return False
    wanted = {k: (v.get("version"), v.get("integrity")) for k, v in locked.items() if k}
    if not wanted:
        return True
    try:
        hidden = json.loads((Path(path) / "node_modules/.package-lock.json").read_text())["packages"]
    except (OSError, ValueError, KeyError):
        return False
    return wanted == {k: (v.get("version"), v.get("integrity")) for k, v in hidden.items() if k}


def find_node(bridge):
    candidates = [os.environ.get("LLM_COLLAB_NODE"), bridge / "runtime/node/bin/node", shutil.which("node"),
                  *(Path(d) / "node" for d in EXTRA_PATH)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


def _tool_env(node):
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(node.parent), env.get("PATH", ""), *EXTRA_PATH])
    return env


def install_deps(path, node, apply):
    if deps_installed(path):
        return "unchanged"
    if apply:
        if node is None:
            raise BridgeError("Node.js 22+ not found (brew install node, or set LLM_COLLAB_NODE)")
        npm = node.parent / "npm"
        npm = npm if npm.exists() else shutil.which("npm", path=_tool_env(node)["PATH"])
        if not npm:
            raise BridgeError(f"npm not found next to {node}")
        _run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=path, env=_tool_env(node), timeout=1800)
    return "npm ci"


def ensure_workspace(bridge, apply):
    workspace, base = bridge / "workspace", bridge / "runtime/portable-base.git"
    if workspace.exists():
        return "unchanged"
    if apply:
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--quiet", "--bare", ROOT, base])
        _run(["git", "clone", "--quiet", base, workspace])
        _git(workspace, "remote", "set-url", "--push", "origin", bridge / "runtime/no-push-target")
    return "created (clone of a local mirror of this repository)"


def link_node(bridge, node, apply):
    """Record the node used here as <bridge>/node for the launcher, unless the bridge has its own runtime/node."""
    link = bridge / "node"
    if (bridge / "runtime/node/bin/node").exists() or node is None:
        return "unchanged" if not link.is_symlink() else "skipped (runtime/node present; stale link left)"
    target = Path(os.path.realpath(node))
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return "unchanged"
    if link.exists() and not link.is_symlink():
        return "skipped (not a symlink; left alone)"
    if apply:
        bridge.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    return f"linked to {target}"


def backup(path, backup_dir, home):
    try:
        relative = Path(path).resolve().relative_to(Path(home).resolve())
    except ValueError:
        relative = Path(Path(path).name)
    dest = Path(backup_dir) / relative.parent / f"{relative.name}.{time.strftime('%Y%m%d-%H%M%S')}-install-agents"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def render(bridge, backup_dir, home, apply):
    """Copy the managed files into the bridge directory. Returns [(name, action, backup path or None)]."""
    results = []
    for template, name, mode in MANAGED:
        target, data = bridge / name, (TEMPLATES / template).read_bytes()
        if target.is_file() and target.resolve() == (TEMPLATES / template).resolve():
            same_mode = (target.stat().st_mode & 0o777) == mode
            if apply and not same_mode:
                target.chmod(mode)
            results.append((name, "in place" if same_mode else "mode fixed", None))
            continue
        current = target.read_bytes() if target.is_file() else None
        same_mode = current is not None and (target.stat().st_mode & 0o777) == mode
        if current == data and same_mode:
            results.append((name, "unchanged", None))
            continue
        action = "created" if current is None else ("updated" if current != data else "mode fixed")
        saved = None
        if apply:
            if current is not None:
                saved = backup(target, backup_dir, home)
            bridge.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{name}.install-agents.tmp")
            tmp.write_bytes(data)
            tmp.chmod(mode)
            os.replace(tmp, target)
        results.append((name, action, saved))
    return results


# Client registrations ---------------------------------------------------------------------------------------

def _client_env(home):
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *EXTRA_PATH, str(Path(home) / ".local/bin")])
    if Path(home).resolve() != Path.home().resolve():
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(Path(home) / ".codex")
        env.pop("CLAUDE_CONFIG_DIR", None)
    # Bridge-delegated turns run with CODEX_HOME (and possibly CLAUDE_CONFIG_DIR) inside the bridge directory, e.g.
    # <bridge>/state/codex. The Bridge rebuilds that config.toml from allowlisted keys on every delegated start, and
    # an llm-collab entry there would let delegated turns call the coordinator recursively. Never read or write a
    # registration there: fall back to the interactive client's default location under <home>.
    for name, default in (("CODEX_HOME", Path(home) / ".codex"), ("CLAUDE_CONFIG_DIR", None)):
        if env.get(name) and _inside(env[name], bridge_dir(home)):
            if default is None:
                env.pop(name)
            else:
                env[name] = str(default)
    env.setdefault("CODEX_HOME", str(Path(home) / ".codex"))
    return env


def _inside(path, directory):
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(directory))
        return True
    except ValueError:
        return False


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _claude_current(home, env):
    config_dir = env.get("CLAUDE_CONFIG_DIR")
    entry = _read_json(Path(config_dir) / ".claude.json" if config_dir else Path(home) / ".claude.json")
    entry = (entry.get("mcpServers") or {}).get(SERVER)
    return entry and (entry.get("command"), list(entry.get("args") or []))


def _codex_current(home, env):
    try:
        config = tomllib.loads((Path(env.get("CODEX_HOME") or Path(home) / ".codex") / "config.toml").read_text())
    except (OSError, ValueError):
        return None
    entry = (config.get("mcp_servers") or {}).get(SERVER)
    return entry and (entry.get("command"), list(entry.get("args") or []))


def _agy_current(home, env):
    entry = (_read_json(Path(home) / ".gemini/config/mcp_config.json").get("mcpServers") or {}).get(SERVER)
    if entry and entry.get("disabled"):
        return ("<disabled>", [])
    return entry and (entry.get("command"), list(entry.get("args") or []))


def _claude_commands(launcher, exists):
    remove = [["mcp", "remove", SERVER, "-s", "user"]] if exists else []
    return remove + [["mcp", "add", "--scope", "user", "--transport", "stdio", SERVER, "--", launcher]]


CLIENTS = (
    # label, CLI name, reader, commands(launcher, exists)
    ("Claude Code", "claude", _claude_current, _claude_commands),
    ("Codex", "codex", _codex_current, lambda launcher, exists: [["mcp", "add", SERVER, "--", launcher]]),
    ("Antigravity", "agy", _agy_current,
     lambda launcher, exists: [["mcp", "add", SERVER, launcher]] + ([["mcp", "enable", SERVER]] if exists else [])),
)


def register(home, launcher, apply, bins=None):
    """Point each installed client's llm-collab entry at the launcher. Returns [(label, action)]."""
    env, results, wanted = _client_env(home), [], (str(launcher), [])
    for label, cli, current_of, commands in CLIENTS:
        current = current_of(home, env)
        if current == wanted:
            results.append((label, "unchanged"))
            continue
        binary = (bins or {}).get(cli) or shutil.which(cli, path=env["PATH"])
        if not binary:
            results.append((label, f"skipped ({cli} CLI not found)"))
            continue
        if apply:
            if cli == "codex":  # codex refuses a CODEX_HOME that does not exist yet
                Path(env.get("CODEX_HOME") or Path(home) / ".codex").mkdir(parents=True, exist_ok=True)
            for args in commands(str(launcher), current is not None):
                _run([binary, *args], cwd=home, env=env, timeout=120)
            if current_of(home, env) != wanted:
                raise BridgeError(f"{label}: registration did not take effect")
        results.append((label, "registered" if current is None else f"updated (was {current[0]})"))
    return results


# Entry points -------------------------------------------------------------------------------------------------

def install(home, backup_dir, mode="apply", sources=None, bins=None, register_clients=True):
    """mode: apply | dry-run | check. Returns (report lines, pending) and raises BridgeError on failure."""
    apply, home = mode == "apply", Path(home)
    bridge, lines, pending = bridge_dir(home), [], False
    verb = "" if apply else "would be "

    def note(item, action, extra=""):
        nonlocal pending
        # "in place": the bridge directory is this checkout (~/llm-serving), so the file is already the template
        changed = action not in ("unchanged", "in place") and not action.startswith("skipped")
        pending |= changed
        shown = f"{verb}{action}" if changed else ("up to date" if action == "unchanged" else action)
        lines.append(f"bridge {item}: {shown}{extra}")

    pins = load_pins(sources)
    node = find_node(bridge)
    for name, pin in pins.items():
        path = bridge / pin["path"]
        note(f"{name} ({pin['path']})", sync_checkout(path, pin, apply))
        if path.exists() or not apply:
            note(f"{name} dependencies", install_deps(path, node, apply) if path.exists() else "npm ci")
    note("workspace", ensure_workspace(bridge, apply))
    note("node link", link_node(bridge, node, apply))
    for name, action, saved in render(bridge, backup_dir, home, apply):
        note(name, action, f" (previous file saved to {saved})" if saved else "")
    if apply:
        if node is None:
            raise BridgeError("Node.js 22+ not found (brew install node, or set LLM_COLLAB_NODE)")
        check = [node, bridge / pins["llm-collab"]["path"] / "bin/llm-collab.mjs", "--config", bridge / "llm-collab.json",
                 "--check"]
        lines.append(f"bridge config check: {_run(check, cwd=bridge, env=_tool_env(node), timeout=60)}")
    if register_clients:
        for label, action in register(home, bridge / "mcp", apply, bins):
            note(f"MCP registration in {label}", action)
    return lines, pending


def status(home):
    """Read-only doctor summary: [(ok, message)]. Never runs Git network commands or client CLIs."""
    home, out = Path(home), []
    bridge = bridge_dir(home)
    for name, pin in load_pins().items():
        path = bridge / pin["path"]
        try:
            head = _git(path, "rev-parse", "HEAD", timeout=10) if path.exists() else None
        except BridgeError:
            head = None
        ok = head == pin["commit"] and deps_installed(path)
        out.append((ok, f"llm-collab {name}: " + ("at pin, dependencies installed" if ok else
                   "missing" if head is None else f"at {head[:12]}, pin {pin['commit'][:12]}"
                   + ("" if deps_installed(path) else ", dependencies not installed"))))
    drift = [name for _t, name, _m in MANAGED if not ((bridge / name).is_file()
             and (bridge / name).read_bytes() == (TEMPLATES / _t).read_bytes())]
    out.append((not drift, "llm-collab launcher: " + ("matches platform/agent-bridge" if not drift
                                                       else "differs or missing: " + ", ".join(drift))))
    env, wanted = _client_env(home), (str(bridge / "mcp"), [])
    for label, _cli, current_of, _commands in CLIENTS:
        current = current_of(home, env)
        out.append((current == wanted, f"llm-collab MCP in {label}: " + (
            "registered" if current == wanted else "not registered" if current is None else f"points to {current[0]}")))
    return out
