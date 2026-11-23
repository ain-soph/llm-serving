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
import install_agents
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
            # pa1b (2026-11-18): one entry per size, each a stored per-class mix (mix.json); no -int8 entries
            self.assertEqual(list(rerank["models"]), ["qwen3-reranker-0.6b", "qwen3-reranker-4b", "qwen3-reranker-8b"])
            for size in ("0.6b", "4b", "8b"):
                args = shlex.split(rerank["models"][f"qwen3-reranker-{size}"]["cmd"])
                self.assertEqual(args[args.index("--model") + 1], str(models / "qwen3-reranker" / f"mix-{size}"))
                self.assertNotIn("--float-model", args)
                self.assertNotIn("--w8a8-classes", args)
            embed = yaml.safe_load((output / "conf.d/qwen3-embedding.yaml").read_text())
            self.assertEqual(shlex.split(embed["macros"]["qe-cmd"])[1], str(root / "embedding/qwen3/server.py"))
            self.assertEqual(list(embed["models"]), ["qwen3-embedding-0.6b", "qwen3-embedding-4b", "qwen3-embedding-8b",
                                                     "voyage-4-nano"])
            for model_id, directory in [("qwen3-embedding-0.6b", "zc-0.6b-fp16"), ("qwen3-embedding-4b", "zc-4b-fp16"),
                                        ("qwen3-embedding-8b", "mix-8b")]:
                args = shlex.split(embed["models"][model_id]["cmd"])
                self.assertEqual(args[args.index("--model") + 1], str(models / "qwen3-embedding" / directory))
            args = shlex.split(embed["models"]["voyage-4-nano"]["cmd"])
            self.assertEqual(args[args.index("--model") + 1], str(models / "voyage-4-nano/zc-bf16"))
            decide = yaml.safe_load((output / "conf.d/decisions.yaml").read_text())
            self.assertEqual(list(decide["models"]), ["pplx-decider-flash"])   # one entry: stored mix, text + images
            args = shlex.split(decide["models"]["pplx-decider-flash"]["cmd"])
            self.assertEqual(args[1], str(root / "decisions/decider/server.py"))
            self.assertEqual(args[args.index("--model") + 1], str(models / "pplx-decider-flash-mix8"))
            self.assertNotIn("--w8a8", args)
            self.assertNotIn("--no-vision", args)
            baselabs = yaml.safe_load((output / "conf.d/decisions-baselabs.yaml").read_text())
            self.assertEqual(list(baselabs["models"]), ["elephant-1"])   # one entry: the stored per-class mix
            args = shlex.split(baselabs["models"]["elephant-1"]["cmd"])
            self.assertEqual(args[1], str(root / "decisions/jev/server.py"))
            self.assertEqual(args[args.index("--model") + 1], str(models / "elephant-1-mix8"))
            self.assertNotIn("--quant", args)
            mammoth = yaml.safe_load((output / "conf.d/decisions-mammoth.yaml").read_text())
            self.assertEqual(list(mammoth["models"]), ["mammoth-1"])
            args = shlex.split(mammoth["models"]["mammoth-1"]["cmd"])
            self.assertEqual(args[args.index("--model") + 1], str(models / "mammoth-1-mix8"))
            qwen_pe = yaml.safe_load((output / "conf.d/qwen-image.yaml").read_text())["models"]["qwen-image-2.1-turbo-pe"]
            self.assertIn("QWEN_IMAGE_DEFAULT_ENHANCE=llm", qwen_pe["env"])
            tts = yaml.safe_load((output / "conf.d/tts.yaml").read_text())
            self.assertEqual(shlex.split(tts["macros"]["tts-cmd"])[1], str(root / "tts/server.py"))
            self.assertNotIn("cosyvoice3", tts["models"])
            args = shlex.split(tts["models"]["qwen3-tts-clone"]["cmd"])
            self.assertEqual(args[args.index("--model") + 1], str(models / "qwen3-tts/1.7B-Base"))
            self.assertIn("--q8", shlex.split(asr["models"]["qwen3-asr-1.7b-int8"]["cmd"]))
            health = plistlib.loads((output / "app.llm-serving.health.plist").read_bytes())
            self.assertEqual(health["StartInterval"], 600)
            self.assertNotIn("KeepAlive", health)
            self.assertNotIn("sov-transcribe-1", asr["models"])          # retired 2026-11-18
            self.assertNotIn("nemotron-3.5-transcribe", asr["models"])
            qwen = yaml.safe_load((output / "conf.d/qwen-image.yaml").read_text())
            command = shlex.split(qwen["macros"]["qwen-image-cmd"])
            self.assertEqual(command[:2], [str(root / ".venv/bin/python"), str(root / "qwen-image/server.py")])
            self.assertEqual(sorted(qwen["models"]), ["qwen-image-2.1", "qwen-image-2.1-turbo", "qwen-image-2.1-turbo-pe"])
            for model_id, variant in [("qwen-image-2.1-turbo", "turbo"), ("qwen-image-2.1", "base")]:
                entry = qwen["models"][model_id]
                args = shlex.split(entry["cmd"])
                self.assertEqual(args[args.index("--model") + 1], str(models / "qwen-image-2.1" / variant))
                self.assertIn(f"MFLUX_QWEN21_W8A8={models / 'qwen-image-2.1/w8a8' / variant}", entry["env"])
            self.assertNotIn("nemo", config["binaries"])                  # Nemotron retired 2026-11-18
            self.assertNotIn("nemo", config["sources"])
            env = environment(config)
            self.assertEqual(env["LLM_NEMO_EXE"], "")
            self.assertEqual(env["H3_VIDEO_FETCH_HOSTS"], "")
            self.assertEqual(env["H3_VIDEO_LSW_URL"], "http://127.0.0.1:8081")
            self.assertEqual(config["trusted_proxies"], [])
            self.assertNotIn("trusted_proxies", (output / "Caddyfile").read_text())
            self.assertFalse(any("@MODELS@" in p.read_text() for p in output.rglob("*.yaml")))

    def test_nemo_is_opt_in(self):
        # Nemotron was retired 2026-11-18; --nemo-exe brings it back (binary and lock source) for tools/nemo.sh.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            shutil.copytree(ROOT / "templates", root / "templates")
            shutil.copy2(ROOT / "native.lock.json", root / "native.lock.json")
            exe = base / "nemo-speech"
            args = ["configure.py", "--models", str(base / "models"), "--state", str(base / "state"),
                    "--output", str(base / "out"), "--pool-mb", "83192", "--nemo-exe", str(exe)]
            hw = {"system": "Darwin", "arch": "arm64", "memory_mb": 98304, "wired_limit_mb": 88064}
            with patch.object(configure, "ROOT", root), patch.object(configure, "hardware", return_value=hw), \
                 patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                configure.main()
            config = json.loads((base / "out/machine.json").read_text())
            self.assertEqual(config["binaries"]["nemo"], str(exe))
            self.assertIn("nemo", config["sources"])
            self.assertEqual(environment(config)["LLM_NEMO_EXE"], str(exe))

    def test_retrieval_models_are_required(self):
        # Every Qwen3 rerank / embedding directory the templates point at is checked by doctor (pa1b layout).
        import re
        required = json.loads((ROOT / "models.required.json").read_text())
        for name in ("rerank.yaml", "qwen3-embedding.yaml"):
            for directory in re.findall(r"--model @MODELS@/(\S+)", (ROOT / "templates/conf.d" / name).read_text()):
                self.assertIn(directory, required["directories"])
                if directory.startswith(("qwen3-reranker/", "qwen3-embedding/")):
                    marker = "mix.json" if Path(directory).name.startswith("mix-") else "w8a8.json"
                    self.assertIn(f"{directory}/model.safetensors", required["files"])
                    self.assertIn(f"{directory}/{marker}", required["files"])
        listed = [d for d in required["directories"] if d.startswith(("qwen3-reranker/", "qwen3-embedding/"))]
        self.assertEqual(listed, ["qwen3-reranker/mix-0.6b", "qwen3-reranker/mix-4b", "qwen3-reranker/mix-8b",
                                  "qwen3-embedding/zc-0.6b-fp16", "qwen3-embedding/zc-4b-fp16",
                                  "qwen3-embedding/mix-8b"])

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

    def test_privacy_rules_for_the_production_layout(self):
        ok = privacy_check.inspect_file
        self.assertEqual(ok(Path("bin/gpu-lock"), b"#!/bin/zsh\n"), [])
        self.assertEqual(ok(Path("platform/agent-bridge/mcp"), b"#!/bin/sh\n"), [])
        self.assertEqual(ok(Path("platform/agent-bridge/llm-collab.env"), b"X=1\n"), [])
        self.assertEqual(ok(Path("clients/mcp/server.py"), b"x = 1\n"), [])
        self.assertEqual(ok(Path("services/text/strata/configs/a.json.in"), b"{}\n"), [])
        self.assertEqual(ok(Path("platform/launchd/com.local.x.plist.in"), b"<plist/>\n"), [])
        self.assertIn("private/runtime file", ok(Path("clients/mcp/compose.yaml"), b"x: 1\n"))
        self.assertIn("private/runtime file", ok(Path(".local/serving.env"), b"X=1\n"))
        self.assertIn("private/runtime file", ok(Path("platform/agent-bridge/llm-collab.local.env"), b"X=1\n"))
        self.assertIn("unexpected file type", ok(Path("bin/tool"), b"no shebang\n"))
        self.assertIn("unexpected file type", ok(Path("services/x/tool.bin.in"), b"\n"))

    def test_agent_files_pass_privacy_rules(self):
        self.assertEqual(privacy_check.inspect_file(Path("agents/local-worker/run"), b"#!/bin/zsh\n"), [])
        self.assertEqual(privacy_check.inspect_file(Path("agents/local-worker/lib/sandbox.sb"), b"(version 1)"), [])
        self.assertIn("unexpected file type", privacy_check.inspect_file(Path("agents/local-worker/blob"), b"data"))
        self.assertIn("private/runtime file", privacy_check.inspect_file(Path("agents/local-worker/logs/x.json"), b"{}"))

    def test_claude_md_block_merges_without_overwriting(self):
        new = install_agents.block("rules v2")
        self.assertEqual(install_agents.merge(None, new), (new, "created"))
        own = "# my notes\nkeep me\n"
        text, action = install_agents.merge(own, new)
        self.assertEqual(action, "appended")
        self.assertTrue(text.startswith(own) and text.endswith(new))
        self.assertEqual(install_agents.merge(text, new), (text, "unchanged"))
        old = own + "\n" + install_agents.block("rules v1") + "tail\n"
        text, action = install_agents.merge(old, new)
        self.assertEqual((action, text), ("updated", own + "\n" + new + "tail\n"))
        with self.assertRaises(ValueError):
            install_agents.merge(install_agents.BEGIN + "\nhalf", new)

    def test_install_agents_backs_up_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            target, backups = home / ".claude/CLAUDE.md", home / "backup"
            target.parent.mkdir()
            target.write_text("personal rules\n")
            action, saved = install_agents.install_claude_md(target, backups, home)
            self.assertEqual(action, "appended")
            self.assertEqual(saved.read_text(), "personal rules\n")
            self.assertIn(install_agents.TEMPLATE.read_text().strip(), target.read_text())
            self.assertEqual(install_agents.install_claude_md(target, backups, home), ("unchanged", None))
            worker = home / "serving/local-worker"
            self.assertEqual(install_agents.link_worker(worker, check=True), "linked")
            self.assertFalse(worker.exists())
            self.assertEqual(install_agents.link_worker(worker), "linked")
            self.assertEqual(install_agents.link_worker(worker), "unchanged")
            other = home / "other"
            other.mkdir()
            self.assertEqual(install_agents.link_worker(other), "kept (existing directory)")

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
