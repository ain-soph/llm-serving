"""Install the multi-agent rules into ~/.claude/CLAUDE.md and link the local worker to this checkout.

CLAUDE.md: the rules live between BEGIN/END markers. A missing file is created; an existing file keeps everything
outside the markers (the block is appended when absent, replaced when it differs). Any change to an existing file
is backed up first. Local worker: the deployment path becomes a symlink to agents/local-worker in this checkout;
an existing directory or foreign link there is left alone and reported. Never needs sudo; --check writes nothing.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "agents/claude-code/global-CLAUDE.md"
WORKER = ROOT / "agents/local-worker"
BEGIN = "<!-- BEGIN llm-serving agents: managed by tools/install_agents.py, edit agents/claude-code/global-CLAUDE.md -->"
END = "<!-- END llm-serving agents -->"


def block(template_text):
    return f"{BEGIN}\n{template_text.strip()}\n{END}\n"


def merge(existing, new_block):
    """Return (text, action) for an existing CLAUDE.md text (None = missing)."""
    if existing is None:
        return new_block, "created"
    start, end = existing.find(BEGIN), existing.find(END)
    if start >= 0 and end > start:
        stop = end + len(END)
        if existing[stop:stop + 1] == "\n":
            stop += 1
        if existing[start:stop] == new_block:
            return existing, "unchanged"
        return existing[:start] + new_block + existing[stop:], "updated"
    if start >= 0 or end >= 0:
        raise ValueError("unbalanced llm-serving markers; fix the file by hand")
    sep = "" if not existing else ("\n" if existing.endswith("\n") else "\n\n")
    return existing + sep + new_block, "appended"


def backup(path, backup_dir, home):
    try:
        relative = path.resolve().relative_to(home.resolve())
    except ValueError:
        relative = Path(path.name)
    dest = backup_dir / relative.parent / f"{relative.name}.{time.strftime('%Y%m%d-%H%M%S')}-install-agents"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def install_claude_md(target, backup_dir, home, check=False):
    existing = target.read_text() if target.exists() else None
    text, action = merge(existing, block(TEMPLATE.read_text()))
    if check or action == "unchanged":
        return action, None
    saved = backup(target, backup_dir, home) if existing is not None else None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".install-agents.tmp")
    tmp.write_text(text)
    os.replace(tmp, target)
    return action, saved


def link_worker(target, check=False):
    if target.is_symlink():
        return "unchanged" if target.resolve() == WORKER.resolve() else "kept (links elsewhere)"
    if target.exists():
        return "kept (existing directory)"
    if not check:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(WORKER)
    return "linked"


def main():
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claude-md", type=Path, default=home / ".claude/CLAUDE.md")
    p.add_argument("--worker-path", type=Path, default=home / "llm-serving/platform/agent-bridge/local-worker")
    p.add_argument("--backup-dir", type=Path, default=home / "llm-serving/.backup")
    p.add_argument("--no-worker", action="store_true", help="Only install the CLAUDE.md rules")
    p.add_argument("--check", action="store_true", help="Report what would change; exit 1 if anything would")
    a = p.parse_args()
    try:
        action, saved = install_claude_md(a.claude_md, a.backup_dir, home, a.check)
    except ValueError as e:
        p.error(f"{a.claude_md}: {e}")
    if a.check:
        print(f"CLAUDE.md rules: {'up to date' if action == 'unchanged' else 'would be ' + action}")
    else:
        print(f"CLAUDE.md rules: {action}" + (f" (previous file saved to {saved})" if saved else ""))
    pending = action != "unchanged"
    if not a.no_worker:
        state = link_worker(a.worker_path, a.check)
        print(f"local worker: {'would be linked' if a.check and state == 'linked' else state} ({a.worker_path} -> {WORKER})")
        if state.startswith("kept"):
            print("  Move its runtime state (logs/, claude-config/, codex-home/) into the checkout, remove it, then rerun.")
        pending |= state != "unchanged"
    raise SystemExit(int(a.check and pending))


if __name__ == "__main__":
    main()
