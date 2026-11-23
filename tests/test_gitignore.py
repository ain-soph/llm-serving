import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_gitignore


class GitignoreTests(unittest.TestCase):
    def test_gitignore_is_generated_from_public_paths(self):
        self.assertEqual((ROOT / ".gitignore").read_text(), gen_gitignore.generate((ROOT / "public-paths.txt").read_text()))

    def test_tracked_files_are_listed_and_not_ignored(self):
        git = ["git", "-C", str(ROOT)]
        tracked = subprocess.run(git + ["ls-files"], capture_output=True, text=True, check=True).stdout.split()
        if not tracked:
            self.skipTest("not a git checkout")
        entries = gen_gitignore.entries((ROOT / "public-paths.txt").read_text())
        dirs = tuple(e[:-2] for e in entries if e.endswith("/**"))
        files = {e for e in entries if not e.endswith("/**")}
        self.assertEqual([f for f in tracked if f not in files and not f.startswith(dirs)], [])
        ignored = subprocess.run(git + ["ls-files", "-ci", "--exclude-standard"], capture_output=True, text=True,
                                 check=True).stdout.split()
        self.assertEqual(ignored, [])

    def test_private_paths_stay_ignored(self):
        probe = ["AGENTS.md", "WORKLOG.md", ".local/serving.env", ".venv/bin/python", "logs/regress/x.txt",
                 "reports/x.md", "evals/golden/x/meta.json", "platform/llama-swap/.api_key",
                 "platform/llama-swap/activity.sqlite", "platform/agent-bridge/state/x.json",
                 "platform/agent-bridge/llm-collab.local.env", "services/text/strata/configs/strata-x.json",
                 "services/text/strata/repo/README.md", "services/rerank/rerank_cases.py", "docs/handoff/x.md",
                 "bin/__pycache__/x.pyc", "clients/mcp/compose.yaml", "upstream/x", ".backup/x"]
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *probe], capture_output=True, text=True)
        self.assertEqual(sorted(r.stdout.split()), sorted(probe))


if __name__ == "__main__":
    unittest.main()
