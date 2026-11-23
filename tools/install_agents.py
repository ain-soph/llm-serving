"""Install the multi-agent rules into ~/.claude/CLAUDE.md and link the local worker to this checkout.

CLAUDE.md: the rules live between BEGIN/END markers. A missing file is created; an existing file keeps everything
outside the markers (the block is appended when absent, replaced when it differs). Any change to an existing file
is backed up first; a block written by an older checkout (old BEGIN marker) is replaced, not duplicated. Local
worker: it lives in this checkout (platform/agent-bridge/local-worker), so in the deployment it is "in place";
another home gets a symlink to it, and an existing directory or foreign link there is left alone and reported. llm-collab bridge: see
tools/llm_collab_bridge.py (pinned checkouts, launcher, MCP registration; --no-bridge skips it).
Never needs sudo; --check and --dry-run write nothing. --home installs into another home directory (tests).
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import llm_collab_bridge

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "platform/agent-bridge/global-CLAUDE.md"
WORKER = ROOT / "platform/agent-bridge/local-worker"
BEGIN = "<!-- BEGIN llm-serving agents: managed by tools/install_agents.py, edit platform/agent-bridge/global-CLAUDE.md -->"
# Markers written by earlier layouts: recognised (and rewritten as BEGIN) so the block is replaced, not appended twice.
OLD_BEGINS = ("<!-- BEGIN llm-serving agents: managed by tools/install_agents.py, edit agents/claude-code/global-CLAUDE.md -->",)
END = "<!-- END llm-serving agents -->"


def block(template_text):
    return f"{BEGIN}\n{template_text.strip()}\n{END}\n"


def merge(existing, new_block):
    """Return (text, action) for an existing CLAUDE.md text (None = missing)."""
    if existing is None:
        return new_block, "created"
    start, end = -1, existing.find(END)
    for marker in (BEGIN, *OLD_BEGINS):
        start = existing.find(marker)
        if start >= 0:
            break
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
    if not target.is_symlink() and target.exists() and target.resolve() == WORKER.resolve():
        return "in place"
    if target.is_symlink():
        return "unchanged" if target.resolve() == WORKER.resolve() else "kept (links elsewhere)"
    if target.exists():
        return "kept (existing directory)"
    if not check:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(WORKER)
    return "linked"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", type=Path, help="Home directory to install into (default: yours); sets the defaults below")
    p.add_argument("--claude-md", type=Path, help="Default: <home>/.claude/CLAUDE.md")
    p.add_argument("--worker-path", type=Path, help="Default: <home>/llm-serving/platform/agent-bridge/local-worker")
    p.add_argument("--backup-dir", type=Path, help="Default: <home>/llm-serving/.backup")
    p.add_argument("--no-worker", action="store_true", help="Skip the local worker link")
    p.add_argument("--no-bridge", action="store_true", help="Skip the llm-collab bridge (checkouts, launcher, MCP)")
    p.add_argument("--no-register", action="store_true", help="Skip registering llm-collab in the MCP clients")
    p.add_argument("--source", action="append", default=[], metavar="NAME=URL",
                   help="Clone/fetch a pinned checkout from another URL or path (llm-collab, local-agent-bridge)")
    p.add_argument("--check", action="store_true", help="Report what would change; exit 1 if anything would")
    p.add_argument("--dry-run", action="store_true", help="Report what would change; exit 0")
    a = p.parse_args()
    home = (a.home or Path.home()).absolute()
    a.claude_md = a.claude_md or home / ".claude/CLAUDE.md"
    a.worker_path = a.worker_path or home / "llm-serving/platform/agent-bridge/local-worker"
    a.backup_dir = a.backup_dir or home / "llm-serving/.backup"
    report = a.check or a.dry_run
    try:
        action, saved = install_claude_md(a.claude_md, a.backup_dir, home, report)
    except ValueError as e:
        p.error(f"{a.claude_md}: {e}")
    if report:
        print(f"CLAUDE.md rules: {'up to date' if action == 'unchanged' else 'would be ' + action}")
    else:
        print(f"CLAUDE.md rules: {action}" + (f" (previous file saved to {saved})" if saved else ""))
    pending = action != "unchanged"
    if not a.no_worker:
        state = link_worker(a.worker_path, report)
        print(f"local worker: {'would be linked' if report and state == 'linked' else state} ({a.worker_path} -> {WORKER})")
        if state.startswith("kept"):
            print("  Move its runtime state (logs/, claude-config/, codex-home/) into the checkout, remove it, then rerun.")
        pending |= state not in ("unchanged", "in place")
    if not a.no_bridge:
        try:
            sources = dict(item.split("=", 1) for item in a.source)
            lines, bridge_pending = llm_collab_bridge.install(
                home, a.backup_dir, "check" if a.check else "dry-run" if a.dry_run else "apply", sources,
                register_clients=not a.no_register)
        except (ValueError, llm_collab_bridge.BridgeError) as e:
            print(f"llm-collab bridge: FAILED: {e}")
            raise SystemExit(1)
        print("\n".join(lines))
        pending |= bridge_pending
    raise SystemExit(int(a.check and pending))


if __name__ == "__main__":
    main()
