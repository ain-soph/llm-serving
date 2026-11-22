"""llm-collab bridge installer: rendering, backups, idempotence, registration. Temp directories only, no network."""
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import llm_collab_bridge as bridge_mod

FAKE_CLI = textwrap.dedent('''\
    #!{python}
    import json, os, pathlib, sys
    name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
    with open({log!r}, "a") as log:
        log.write(json.dumps([name, *args]) + "\\n")
    home = pathlib.Path(os.environ["HOME"])
    if name == "claude":
        f = home / ".claude.json"
        d = json.loads(f.read_text()) if f.exists() else {{}}
        if args[:2] == ["mcp", "remove"]:
            d.get("mcpServers", {{}}).pop(args[2], None)
        else:
            i = args.index("--")
            d.setdefault("mcpServers", {{}})[args[i - 1]] = {{"type": "stdio", "command": args[i + 1], "args": [], "env": {{}}}}
        f.write_text(json.dumps(d))
    elif name == "codex":
        f = pathlib.Path(os.environ["CODEX_HOME"]) / "config.toml"
        if not f.parent.is_dir():
            sys.exit("CODEX_HOME missing")
        f.write_text('[mcp_servers.{{}}]\\ncommand = "{{}}"\\n'.format(args[2], args[4]))
    elif name == "agy":
        f = home / ".gemini/config/mcp_config.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        d = json.loads(f.read_text()) if f.exists() else {{}}
        if args[1] == "add":
            d.setdefault("mcpServers", {{}})[args[2]] = {{"command": args[3], "disabled": False}}
        else:
            d["mcpServers"][args[2]]["disabled"] = False
        f.write_text(json.dumps(d))
''')


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_source(path, branch, entry):
    path.mkdir(parents=True)
    git(path, "init", "--quiet", "--initial-branch", branch)
    (path / "package.json").write_text('{"name": "x", "version": "0.0.0"}\n')
    (path / "package-lock.json").write_text('{"lockfileVersion": 3, "packages": {"": {"name": "x"}}}\n')
    (path / entry).parent.mkdir(parents=True, exist_ok=True)
    (path / entry).write_text("// stub\n")
    git(path, "add", ".")
    git(path, "-c", "user.name=t", "-c", "user.email=t", "-c", "commit.gpgsign=false",
        "commit", "--quiet", "-m", "init")
    return git(path, "rev-parse", "HEAD")


def snapshot(root):
    """Content and mode of every file and link under root (Git object stores included)."""
    out = {}
    for directory, _dirs, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            rel = str(path.relative_to(root))
            if path.is_symlink():
                out[rel] = ("link", os.readlink(path))
            elif "/.git/" not in f"/{rel}" or rel.endswith(("HEAD", "config")):
                out[rel] = (path.stat().st_mode, path.read_bytes())
    return out


class BridgeInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="llm collab ")
        base = Path(self.tmp.name)
        self.home, self.log = base / "home", base / "cli.log"
        self.home.mkdir()
        sources = base / "sources"
        self.pins = {
            "llm-collab": {"url": str(sources / "llm-collab"), "branch": "main", "path": "release",
                           "commit": make_source(sources / "llm-collab", "main", "bin/llm-collab.mjs")},
            "local-agent-bridge": {"url": str(sources / "bridge"), "branch": "serving", "path": "runtime/upstream",
                                   "commit": make_source(sources / "bridge", "serving", "src/collaboration-bridge.mjs")},
        }
        bins = base / "bin"
        bins.mkdir()
        self.bins = {}
        for name in ("claude", "codex", "agy"):
            (bins / name).write_text(FAKE_CLI.format(python=sys.executable, log=str(self.log)))
            (bins / name).chmod(0o755)
            self.bins[name] = str(bins / name)
        self.node = bins / "node"
        self.node.write_text("#!/bin/sh\necho Configuration OK.\n")
        self.node.chmod(0o755)
        self.bridge = bridge_mod.bridge_dir(self.home)
        self.backups = self.home / "llm-serving/.backup"
        patches = [patch.object(bridge_mod, "load_pins", lambda sources=None: json.loads(json.dumps(self.pins))),
                   patch.object(bridge_mod, "find_node", lambda bridge: self.node)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, mode="apply"):
        return bridge_mod.install(self.home, self.backups, mode, bins=self.bins)

    def cli_calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_render_backup_and_idempotence(self):
        self.bridge.mkdir(parents=True)
        (self.bridge / "mcp").write_text("#!/bin/sh\n# private hand-edited wrapper\n")
        (self.bridge / "state").mkdir()
        (self.bridge / "state/bridge-1.json").write_text("{}")
        lines, pending = self.install()
        self.assertTrue(pending)
        for template, name, mode in bridge_mod.MANAGED:
            target = self.bridge / name
            self.assertEqual(target.read_bytes(), (bridge_mod.TEMPLATES / template).read_bytes(), name)
            self.assertEqual(target.stat().st_mode & 0o777, mode, name)
        saved = list((self.backups / "llm-serving/platform/agent-bridge").glob("mcp.*-install-agents"))
        self.assertEqual(len(saved), 1)
        self.assertIn("private hand-edited wrapper", saved[0].read_text())
        self.assertEqual((self.bridge / "state/bridge-1.json").read_text(), "{}")
        self.assertEqual(sorted(os.listdir(self.bridge / "state")), ["bridge-1.json"])
        for name, pin in self.pins.items():
            self.assertEqual(git(self.bridge / pin["path"], "rev-parse", "HEAD"), pin["commit"])
        self.assertTrue(git(self.bridge / "workspace", "remote", "get-url", "origin"))
        self.assertEqual(os.readlink(self.bridge / "node"), os.path.realpath(self.node))
        launcher = str(self.bridge / "mcp")
        claude = json.loads((self.home / ".claude.json").read_text())["mcpServers"]["llm-collab"]
        self.assertEqual(claude["command"], launcher)
        self.assertIn(launcher, (self.home / ".codex/config.toml").read_text())
        agy = json.loads((self.home / ".gemini/config/mcp_config.json").read_text())["mcpServers"]["llm-collab"]
        self.assertEqual(agy["command"], launcher)
        self.assertTrue(any("config check: Configuration OK" in line for line in lines))

        before, calls = snapshot(self.home), len(self.cli_calls())
        lines, pending = self.install()
        self.assertFalse(pending, lines)
        self.assertEqual(snapshot(self.home), before)
        self.assertEqual(len(self.cli_calls()), calls, "second run must not invoke client CLIs")
        self.assertEqual(len(list(self.backups.rglob("*-install-agents"))), 1)
        lines, pending = self.install("check")
        self.assertFalse(pending, lines)

    def test_check_and_dry_run_write_nothing(self):
        self.install()
        (self.bridge / "llm-collab.env").write_text("AGY_MODEL=other\n")
        before = snapshot(self.home)
        for mode in ("check", "dry-run"):
            lines, pending = self.install(mode)
            self.assertTrue(pending)
            self.assertIn("bridge llm-collab.env: would be updated", lines)
            self.assertEqual(snapshot(self.home), before)
        self.install()
        self.assertFalse(self.install("check")[1])

    def test_registration_updates_foreign_entry(self):
        (self.home / ".claude.json").write_text(json.dumps({"mcpServers": {"llm-collab": {"command": "/old/mcp"}}}))
        lines, _ = self.install()
        self.assertIn("bridge MCP registration in Claude Code: updated (was /old/mcp)", lines)
        calls = [json.loads(c) for c in self.cli_calls()]
        self.assertIn(["claude", "mcp", "remove", "llm-collab", "-s", "user"], calls)

    def test_pin_update_and_dirty_checkout(self):
        self.install()
        source = Path(self.pins["llm-collab"]["url"])
        (source / "README.md").write_text("new\n")
        git(source, "add", ".")
        git(source, "-c", "user.name=t", "-c", "user.email=t", "-c", "commit.gpgsign=false",
            "commit", "--quiet", "-m", "next")
        old, self.pins["llm-collab"]["commit"] = self.pins["llm-collab"]["commit"], git(source, "rev-parse", "HEAD")
        lines, _ = self.install()
        self.assertIn(f"bridge llm-collab (release): updated {old[:12]} -> {self.pins['llm-collab']['commit'][:12]}",
                      lines)
        (self.bridge / "release/package.json").write_text("{}\n")
        self.pins["llm-collab"]["commit"] = old
        with self.assertRaisesRegex(bridge_mod.BridgeError, "local changes"):
            self.install()

    def test_status_after_install(self):
        self.install()
        with patch.dict(os.environ, {"CODEX_HOME": str(self.home / ".codex")}):
            results = bridge_mod.status(self.home)
        self.assertTrue(all(ok for ok, _ in results), results)


class TemplateTests(unittest.TestCase):
    def test_templates_are_path_free(self):
        pattern = re.compile(r"/(?:Users|home)/[^\s/]+/|@[A-Za-z0-9.-]+\.[a-z]{2,}|api_key\s*=")
        for path in bridge_mod.TEMPLATES.iterdir():
            self.assertIsNone(pattern.search(path.read_text()), path.name)
        config = json.loads((bridge_mod.TEMPLATES / "llm-collab.json").read_text())
        for key in ("bridgeRoot", "workspace", "stateDirectory", "productionRoot"):
            self.assertFalse(Path(config[key]).is_absolute(), key)
        pins = json.loads((bridge_mod.TEMPLATES / "pins.json").read_text())
        for pin in pins.values():
            self.assertRegex(pin["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(pin["url"].startswith("https://github.com/ain-soph/"))

    def test_shell_syntax(self):
        for name in ("launcher.sh", "codex-select"):
            subprocess.run(["sh", "-n", str(bridge_mod.TEMPLATES / name)], check=True)

    def test_codex_select_prefers_daemon_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cask = base / "Caskroom/codex/9.8.7/bin"
            cask.mkdir(parents=True)
            (cask / "codex").write_text("#!/bin/sh\necho cask\n")
            (cask / "codex").chmod(0o755)
            (base / "bin").mkdir()
            (base / "bin/codex").symlink_to(cask / "codex")
            daemon = base / "home/.codex/packages/app-server-daemon/releases/9.8.7-test-arch/bin"

            def run(**extra):
                env = {"PATH": f"{base / 'bin'}:/usr/bin:/bin", "HOME": str(base / "home"), **extra}
                return subprocess.run(["sh", str(bridge_mod.TEMPLATES / "codex-select")], env=env, check=True,
                                      capture_output=True, text=True).stdout.strip()

            self.assertEqual(run(), "cask")
            daemon.mkdir(parents=True)
            (daemon / "codex").write_text("#!/bin/sh\necho daemon\n")
            (daemon / "codex").chmod(0o755)
            self.assertEqual(run(), "daemon")
            self.assertEqual(run(LLM_COLLAB_CODEX_BIN=str(cask / "codex")), "cask")


class InstallAgentsCliTests(unittest.TestCase):
    def test_home_override_without_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            command = [sys.executable, str(ROOT / "tools/install_agents.py"), "--home", str(home), "--no-bridge"]
            self.assertEqual(subprocess.run(command + ["--dry-run"], capture_output=True).returncode, 0)
            self.assertEqual(list(home.iterdir()), [])
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertTrue((home / ".claude/CLAUDE.md").is_file())
            self.assertTrue((home / "llm-serving/platform/agent-bridge/local-worker").is_symlink())
            self.assertEqual(subprocess.run(command + ["--check"], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
