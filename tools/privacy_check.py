"""Audit tracked/staged public files. Report file names and rule names, never matched values."""
import argparse
import ipaddress
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "absolute home path": re.compile(rb"/(?:Users|home)/[^\s/]+/"),
    "email address": re.compile(rb"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})"),
    "credential-bearing URL": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
    "literal Bearer credential": re.compile(rb"Bearer\s+[A-Za-z0-9_=-]{24,}"),
}
FORBIDDEN = {".env", ".api_key", "api_key", "AGENTS.md", "WORKLOG.md"}
EXTENSIONS = {".py", ".md", ".json", ".toml", ".lock", ".sh", ".yaml", ".yml", ".txt"}


def inspect_file(path, data, extra_literals=()):
    problems = []
    if path.name in FORBIDDEN or any(p in {".venv", ".local", "logs", "outputs", "media", "models", "reports", "clients", "mcp"} for p in path.parts):
        problems.append("private/runtime file")
    if path.suffix not in EXTENSIONS and path.name not in {"Brewfile", "LICENSE", ".gitignore", ".python-version"}:
        problems.append("unexpected file type")
    if len(data) > 2_000_000 or b"\x00" in data:
        problems.append("binary or oversized file")
    for name, rule in RULES.items():
        if rule.search(data):
            problems.append(name)
    for raw in re.findall(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", data):
        try:
            ip = ipaddress.ip_address(raw.decode())
        except ValueError:
            continue
        if not ip.is_loopback and not ip.is_unspecified and ip.is_private:
            problems.append("private network address")
    if any(value and value in data for value in extra_literals):
        problems.append("private literal")
    return sorted(set(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-literals-file", type=Path,
                        help="Local-only newline-delimited values to forbid; contents are never printed")
    parser.add_argument("--check-history", action="store_true")
    a = parser.parse_args()
    literals = a.private_literals_file.read_bytes().splitlines() if a.private_literals_file else []
    for path in literals:
        if len(path) < 4:
            parser.error("Private literals must contain at least four bytes")
    # Read index blobs, so an ignored file accidentally added with -f is still checked.
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    failures = 0
    for raw in filter(None, files):
        name = raw.decode()
        data = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        reasons = inspect_file(Path(name), data, literals)
        if reasons:
            failures += 1
            print(name + ": " + ", ".join(reasons))
    if a.check_history:
        logs = subprocess.check_output(["git", "log", "--all", "--format=%an%n%ae%n%cn%n%ce%n%B"], cwd=ROOT)
        # GitHub's account-associated noreply identity is intentionally public.
        logs = re.sub(rb"\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com", b"github-noreply", logs)
        problems = inspect_file(Path("history.txt"), logs, literals)
        if problems:
            failures += 1
            print("commit metadata: " + ", ".join(problems))
    print(f"Audited {len([f for f in files if f])} indexed files; {failures} failures")
    if not any(files):
        parser.error("No indexed files; stage the intended public files first")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
