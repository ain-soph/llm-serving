import contextlib
import hashlib
import io
import json
import plistlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import install_serving as inst
import privacy_check


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = inst.main([str(a) for a in argv])
    return rc, out.getvalue()


class InstallServingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="serving-home.")
        self.home = Path(self._tmp.name)
        self.values = self.home / "values.json"
        self.values.write_text(json.dumps({"MODELS": "~/weights", "STRATA_RELEASE": "~/strata",
                                           "STRATA_EXE": "~/strata/build/strata",
                                           "STRATA_VISION_EXE": "/opt/vision/strata-vision"}))
        self.root = self.home / "llm-serving"
        self.key = self.home / inst.KEY

    def tearDown(self):
        self._tmp.cleanup()

    def files(self):
        return sorted(p.relative_to(self.home) for p in self.home.rglob("*") if p.is_file())

    def test_dry_run_install_check_and_key_kept(self):
        rc, out = run("--home", self.home, "--values", self.values, "--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(self.files(), [Path("values.json")])
        self.assertIn("would generate", out)
        self.assertEqual(run("--home", self.home, "--values", self.values, "--check")[0], 1)
        rc, out = run("--home", self.home, "--values", self.values)
        self.assertEqual(rc, 0)
        for rel, (_, kind, mode) in inst.FILES.items():
            self.assertEqual((self.home / rel).stat().st_mode & 0o777, mode, rel)
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
        key = self.key.read_bytes()
        self.assertGreaterEqual(len(key.strip()), 32)
        self.assertNotIn(key.strip().decode(), out)
        first = hashlib.sha256(key).hexdigest()
        rc, out = run("--home", self.home, "--values", self.values)
        self.assertEqual(rc, 0)
        self.assertEqual(first, hashlib.sha256(self.key.read_bytes()).hexdigest())
        self.assertNotIn("generated", out)
        self.assertEqual(run("--home", self.home, "--values", self.values, "--check")[0], 0)
        self.assertFalse((self.root / ".backup").exists())

    def test_verbatim_and_rendered_content(self):
        run("--home", self.home, "--values", self.values)
        self.assertEqual(inst.FILES[inst.S + "bin/gpu-top"], ("bin/gpu-top", "verbatim", 0o755))
        for rel, (src, kind, _) in inst.FILES.items():
            data = (self.home / rel).read_bytes()
            if kind == "verbatim":
                self.assertEqual(data, (inst.SOURCE / src).read_bytes())
            else:
                self.assertNotRegex(data.decode(), r"@[A-Z_]+@")
        conf = yaml.safe_load((self.root / "platform/llama-swap/conf.d/strata.yaml").read_text())
        cmd = conf["models"]["qwen3.8-flash-next-uncensored"]["cmd"].split()
        self.assertEqual(cmd[0], str(self.root / ".venv/bin/python"))
        self.assertEqual(cmd[1], str(self.home / "strata/terminal/serve/server.py"))
        self.assertEqual(cmd[cmd.index("--config") + 1],
                         str(self.root / "services/text/strata/configs/strata-uncensored-c262k-p2-compact.json"))
        engine = json.loads((self.root / "services/text/strata/configs/strata-ad427-c262k-p2-compact.json").read_text())
        self.assertEqual(engine["exe"], str(self.home / "strata/build/strata"))
        self.assertEqual(engine["vision"]["exe"], "/opt/vision/strata-vision")
        self.assertTrue(engine["tokenizer"].startswith(str(self.home / "weights") + "/"))

    def test_drift_is_kept_unless_update_then_backed_up(self):
        run("--home", self.home, "--values", self.values)
        conf = self.root / "platform/llama-swap/conf.d/strata.yaml"
        conf.write_text("# edited by hand\n")
        rc, out = run("--home", self.home, "--values", self.values, "--check", "--diff")
        self.assertEqual(rc, 1)
        self.assertIn("differs", out)
        rc, out = run("--home", self.home, "--values", self.values)
        self.assertEqual(rc, 0)
        self.assertIn("drift", out)
        self.assertEqual(conf.read_text(), "# edited by hand\n")
        rc, out = run("--home", self.home, "--values", self.values, "--update")
        self.assertEqual(rc, 0)
        backups = list((self.root / ".backup/platform/llama-swap/conf.d").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertRegex(backups[0].name, r"^strata\.yaml\.\d{8}-\d{6}-install-serving$")
        self.assertEqual(backups[0].read_text(), "# edited by hand\n")
        self.assertEqual(run("--home", self.home, "--values", self.values, "--check")[0], 0)

    def test_existing_key_never_overwritten(self):
        self.key.parent.mkdir(parents=True)
        self.key.write_text("existing-secret\n")
        self.key.chmod(0o644)
        self.assertEqual(run("--home", self.home, "--values", self.values, "--check")[0], 1)
        rc, out = run("--home", self.home, "--values", self.values)
        self.assertEqual(rc, 0)
        self.assertNotIn("existing-secret", out)
        self.assertEqual(self.key.read_text(), "existing-secret\n")
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
        self.key.write_text("")
        rc, out = run("--home", self.home, "--values", self.values)
        self.assertEqual(self.key.read_text(), "")
        self.assertIn("empty", out)

    def test_values_defaults_and_validation(self):
        with patch.object(inst, "DEFAULT_VALUES", self.home / "absent.json"):
            v = inst.load_values(self.home)
        lock = json.loads((ROOT / "native.lock.json").read_text())
        rev = lock["sources"]["strata"]["revision"][:12]
        self.assertEqual(v["STRATA_RELEASE"], str(ROOT / ".local/src" / f"strata-{rev}"))
        self.assertEqual(v["MODELS"], str(self.home / "models"))
        self.assertEqual(v["SERVING_ROOT"], str(self.root))
        bad = self.home / "bad.json"
        bad.write_text(json.dumps({"MODEL": "~/x"}))
        with self.assertRaises(ValueError):
            inst.load_values(self.home, bad)
        bad.write_text(json.dumps({"MODELS": "relative/path"}))
        with self.assertRaises(ValueError):
            inst.load_values(self.home, bad)
        bad.write_text(json.dumps({"MODELS": "~/model weights"}))
        with self.assertRaises(ValueError):
            inst.load_values(self.home, bad)
        with self.assertRaises(FileNotFoundError):
            inst.load_values(self.home, self.home / "missing.json")

    def test_render_only_writes_templates(self):
        out = self.home / "rendered"
        rc, _ = run("--home", self.home, "--values", self.values, "--render", out)
        self.assertEqual(rc, 0)
        expected = sorted(Path(r) for r, (_, kind, _) in inst.FILES.items() if kind == "template")
        self.assertEqual(sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file()), expected)
        self.assertFalse(self.root.exists())
        self.assertFalse((self.home / "Library").exists())

    def test_doctor_reports_without_key(self):
        lines = []
        inst.doctor(self.home, self.values, log=lines.append)
        self.assertTrue(all(line.startswith("WARN") for line in lines))
        run("--home", self.home, "--values", self.values)
        lines = []
        inst.doctor(self.home, self.values, log=lines.append)
        self.assertTrue(all(line.startswith("OK") for line in lines), lines)
        self.assertNotIn(self.key.read_text().strip(), "\n".join(lines))

    def test_launch_agents_written_not_loaded(self):
        rc, out = run("--home", self.home, "--values", self.values)
        agents = self.home / "Library/LaunchAgents"
        self.assertEqual(sorted(p.name for p in agents.iterdir()),
                         ["com.local.llama-server.plist", "com.local.llm-gateway.plist"])
        for plist in agents.iterdir():
            doc = plistlib.loads(plist.read_bytes())
            self.assertEqual(doc["Label"], plist.stem)
            self.assertIn(f"launchctl bootstrap gui/$(id -u) {plist}", out)
        swap = plistlib.loads((agents / "com.local.llama-server.plist").read_bytes())
        self.assertEqual(swap["ProgramArguments"][1], str(self.root / "platform/llama-swap/serve.sh"))
        # the watchdog stays disabled: rendered next to its script, never into LaunchAgents
        watchdog = plistlib.loads((self.root / "bin/com.local.gpu-lock-watchdog.plist.disabled").read_bytes())
        self.assertEqual(watchdog["ProgramArguments"][-1], str(self.root / "bin/gpu-lock-watchdog"))
        self.assertFalse(hasattr(inst, "subprocess"))  # it can print launchctl commands, never run them
        # an edited LaunchAgent is backed up under ~/llm-serving/.backup/Library/LaunchAgents before --update
        (agents / "com.local.llama-server.plist").write_text("old")
        rc, out = run("--home", self.home, "--values", self.values, "--update")
        self.assertIn("launchctl bootout gui/$(id -u)/com.local.llama-server", out)
        saved = list((self.root / ".backup/Library/LaunchAgents").iterdir())
        self.assertEqual([p.read_text() for p in saved], ["old"])

    def test_llama_swap_config_and_gateway(self):
        run("--home", self.home, "--values", self.values)
        main = yaml.safe_load((self.root / "platform/llama-swap/config.yaml").read_text())
        self.assertEqual(main["apiKeys"], ["${env.LLAMA_API_KEY}"])
        self.assertEqual(main["store"]["path"], str(self.root / "platform/llama-swap/activity.sqlite"))
        self.assertEqual(main["macros"]["h3-image-cmd"].split()[-1], str(self.home / "weights/minimax-h3-image"))
        caddy = (self.root / "platform/gateway/Caddyfile").read_text()
        self.assertIn("trusted_proxies static 127.0.0.1/32", caddy)
        self.values.write_text(json.dumps({"TRUSTED_PROXY": "127.0.0.2"}))
        self.assertIn("trusted_proxies static 127.0.0.2/32",
                      inst.rendered(inst.load_values(self.home, self.values))[inst.S + "platform/gateway/Caddyfile"][0].decode())
        self.values.write_text(json.dumps({"TRUSTED_PROXY": "127.0.0.0/8"}))
        with self.assertRaises(ValueError):
            inst.load_values(self.home, self.values)

    def test_published_files_pass_privacy_rules(self):
        for src, _, _ in inst.FILES.values():
            path = Path("serving") / src
            self.assertEqual(privacy_check.inspect_file(path, (inst.SOURCE / src).read_bytes()), [], src)
        for private in ("serving/platform/llama-swap/.api_key", ".local/serving-values.json", "serving-values.json"):
            self.assertIn("private/runtime file", privacy_check.inspect_file(Path(private), b"{}\n"))
        self.assertIn("unexpected file type", privacy_check.inspect_file(Path("serving/conf/tool"), b"#!/bin/sh\n"))


if __name__ == "__main__":
    unittest.main()
