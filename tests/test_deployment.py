import contextlib
import importlib.util
import io
import json
import os
import plistlib
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import configure
import privacy_check
from service import environment


class DeploymentTests(unittest.TestCase):
    def test_render_with_spaces_and_reconfiguration(self):
        with tempfile.TemporaryDirectory(prefix="llm deployment ") as directory:
            base = Path(directory)
            root = base / "source code"
            root.mkdir()
            shutil.copytree(ROOT / "templates", root / "templates")
            shutil.copy2(ROOT / "native.lock.json", root / "native.lock.json")
            models, state, output = base / "model weights", base / "private state", base / "host config"
            args = ["configure.py", "--models", str(models), "--state", str(state),
                    "--output", str(output), "--pool-mb", "83192"]
            hw = {"system": "Darwin", "arch": "arm64", "memory_mb": 98304,
                  "wired_limit_mb": 88064, "chip": "Apple M5 Ultra", "macos": "27.0"}
            with patch.object(configure, "ROOT", root), patch.object(configure, "hardware", return_value=hw), \
                 patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                configure.main()
                first_key = (state / "api_key").read_bytes()
                configure.main()
            self.assertEqual(first_key, (state / "api_key").read_bytes())
            self.assertEqual((state / "api_key").stat().st_mode & 0o777, 0o600)
            config = json.loads((output / "machine.json").read_text())
            for name, label in configure.LABELS.items():
                plist = plistlib.loads((output / (label + ".plist")).read_bytes())
                self.assertEqual(plist["ProgramArguments"][0], str(root / ".venv/bin/python"))
                self.assertEqual(plist["ProgramArguments"][-1], str(output / "machine.json"))
                self.assertNotIn("EnvironmentVariables", plist)
                self.assertNotIn(first_key.strip(), (output / (label + ".plist")).read_bytes())
            import yaml
            main = yaml.safe_load((output / "config.yaml").read_text())
            self.assertNotIn("hooks", main)
            self.assertEqual(main["store"]["path"], str(state / "activity.sqlite"))
            tokens = shlex.split(main["macros"]["h3-image-cmd"])
            self.assertEqual(tokens[0], str(root / ".venv/bin/python"))
            self.assertEqual(tokens[-1], str(models / "minimax-h3-image"))
            asr = yaml.safe_load((output / "conf.d/asr.yaml").read_text())
            args = shlex.split(asr["models"]["qwen3-asr-1.7b"]["cmd"])
            self.assertEqual(args[args.index("--aligner") + 1], str(models / "qwen3-forcedaligner-0.6b"))
            rerank = yaml.safe_load((output / "conf.d/rerank.yaml").read_text())
            command = shlex.split(rerank["macros"]["rerank-cmd"])
            self.assertEqual(command[:2], [str(root / ".venv/bin/python"), str(root / "rerank/server.py")])
            for size, variant in [("0.6b", "bf16"), ("4b", "bf16"), ("8b", "q8_0")]:
                args = shlex.split(rerank["models"]["qwen3-reranker-" + size]["cmd"])
                self.assertEqual(args[args.index("--model") + 1], str(models / "qwen3-reranker" / f"mlx-{size}-{variant}"))
            env = environment(config)
            self.assertEqual(env["H3_VIDEO_FETCH_HOSTS"], "")
            self.assertEqual(env["H3_VIDEO_LSW_URL"], "http://127.0.0.1:8081")
            self.assertEqual(config["trusted_proxies"], [])
            self.assertNotIn("trusted_proxies", (output / "Caddyfile").read_text())
            self.assertFalse(any("@MODELS@" in p.read_text() for p in output.rglob("*.yaml")))

    def test_small_pool_requires_validated_profile(self):
        hw = {"system": "Darwin", "arch": "arm64", "memory_mb": 32768, "wired_limit_mb": 24000}
        with patch.object(configure, "hardware", return_value=hw), \
             patch.object(sys, "argv", ["configure.py", "--models", "/tmp/models"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            configure.main()

    def test_privacy_scanner_checks_actual_content(self):
        name = Path("sample.py")
        self.assertEqual(privacy_check.inspect_file(name, b'URL = "http://127.0.0.1:8080"'), [])
        personal_path = b"/" + b"Users" + b"/example/weights"
        self.assertIn("absolute home path", privacy_check.inspect_file(name, personal_path))
        email = b"person" + b"@" + b"example.org"
        self.assertIn("email address", privacy_check.inspect_file(name, email))
        token = b"ghp_" + b"a" * 36
        self.assertIn("credential token", privacy_check.inspect_file(name, token))
        self.assertIn("private literal", privacy_check.inspect_file(name, b"value=private-value", [b"private-value"]))
        self.assertIn("private/runtime file", privacy_check.inspect_file(Path(".local/config/machine.json"), b"{}"))

    def test_media_stays_local_and_offline_by_default(self):
        path = ROOT / "h3-video/server.py"
        spec = importlib.util.spec_from_file_location("video_wrapper", path)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"H3_VIDEO_FETCH_HOSTS": "", "H3_VIDEO_SHARE_BASE": "http://127.0.0.1:8080"}):
            spec.loader.exec_module(mod)
        self.assertEqual(mod.FETCH_HOSTS, set())
        self.assertEqual(mod.FFMPEG, "ffmpeg")
        self.assertEqual(mod.FFPROBE, "ffprobe")


if __name__ == "__main__":
    unittest.main()
