import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import install_agents
import privacy_check


class DeploymentTests(unittest.TestCase):
    def test_conf_d_commands_point_into_this_checkout(self):
        # Every script a llama-swap entry starts from ${env.LLM_SERVING} is a tracked file of this repository.
        conf = ROOT / "platform/llama-swap"
        missing = []
        for f in [conf / "config.yaml", *sorted((conf / "conf.d").glob("*.yaml"))]:
            for rel in re.findall(r"\$\{env\.LLM_SERVING\}/(\S+\.(?:py|sh))\b", f.read_text()):
                if not (ROOT / rel).is_file():
                    missing.append(f"{f.name}: {rel}")
        self.assertEqual(missing, [])

    def test_retrieval_models_are_required(self):
        # Every Qwen3 rerank / embedding directory the config points at is checked by doctor (pa1b layout).
        required = json.loads((ROOT / "models.required.json").read_text())
        for name in ("rerank.yaml", "qwen3-embedding.yaml"):
            for directory in re.findall(r"--model \$\{env\.LLM_MODELS\}/(\S+)",
                                        (ROOT / "platform/llama-swap/conf.d" / name).read_text()):
                self.assertIn(directory, required["directories"])
                if directory.startswith(("qwen3-reranker/", "qwen3-embedding/")):
                    marker = "mix.json" if Path(directory).name.startswith("mix-") else "w8a8.json"
                    self.assertIn(f"{directory}/model.safetensors", required["files"])
                    self.assertIn(f"{directory}/{marker}", required["files"])
        listed = [d for d in required["directories"] if d.startswith(("qwen3-reranker/", "qwen3-embedding/"))]
        self.assertEqual(listed, ["qwen3-reranker/mix-0.6b", "qwen3-reranker/mix-4b", "qwen3-reranker/mix-8b",
                                  "qwen3-embedding/zc-0.6b-fp16", "qwen3-embedding/zc-4b-fp16",
                                  "qwen3-embedding/mix-8b"])

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
        self.assertEqual(privacy_check.inspect_file(Path("platform/agent-bridge/local-worker/run"), b"#!/bin/zsh\n"), [])
        self.assertEqual(privacy_check.inspect_file(Path("platform/agent-bridge/local-worker/lib/sandbox.sb"), b"(version 1)"), [])
        self.assertIn("unexpected file type", privacy_check.inspect_file(Path("platform/agent-bridge/local-worker/blob"), b"data"))
        self.assertIn("private/runtime file", privacy_check.inspect_file(Path("platform/agent-bridge/local-worker/logs/x.json"), b"{}"))

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
            # the deployment itself: the worker directory is this checkout's own
            self.assertEqual(install_agents.link_worker(install_agents.WORKER), "in place")

    def test_claude_md_old_marker_is_replaced_not_duplicated(self):
        new = install_agents.block("rules v2")
        old_block = install_agents.OLD_BEGINS[0] + "\nrules v1\n" + install_agents.END + "\n"
        text, action = install_agents.merge("mine\n\n" + old_block + "tail\n", new)
        self.assertEqual((action, text), ("updated", "mine\n\n" + new + "tail\n"))
        self.assertEqual(text.count("BEGIN llm-serving agents"), 1)
        self.assertEqual(install_agents.merge(text, new), (text, "unchanged"))

    def test_media_stays_local_and_offline_by_default(self):
        path = ROOT / "services/video/h3-video/server.py"
        spec = importlib.util.spec_from_file_location("video_wrapper", path)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"H3_VIDEO_FETCH_HOSTS": "", "H3_VIDEO_SHARE_BASE": "http://127.0.0.1:8080"}):
            spec.loader.exec_module(mod)
        self.assertEqual(mod.FETCH_HOSTS, set())
        self.assertEqual(mod.FFMPEG, "ffmpeg")
        self.assertEqual(mod.FFPROBE, "ffprobe")


if __name__ == "__main__":
    unittest.main()
