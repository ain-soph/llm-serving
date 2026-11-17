"""Install generated launchd services; optionally persist an explicit wired-memory limit."""
import argparse
import json
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

from configure import LABELS


def run(args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / ".local/config/machine.json")
    parser.add_argument("--activate", action="store_true", help="Start/restart the generated user services")
    parser.add_argument("--wired-limit-mb", type=int, help="Install a root-owned boot daemon; invokes sudo")
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    config = Path(c["config"])
    if a.wired_limit_mb is not None:
        limit = a.wired_limit_mb
        if not 0 < limit <= c["hardware"]["memory_mb"] - 8192:
            parser.error("Wired limit must leave at least 8192 MiB for the system")
        if c["pool_mb"] + 4872 > limit:
            parser.error("Configured GPU pool exceeds the requested wired limit; reconfigure the pool first")
    # Validate configs and binaries before writing outside the project.
    for name, value in c["binaries"].items():
        if not Path(value).is_file() or not os.access(value, os.X_OK):
            parser.error(f"Missing executable {name}; bootstrap/build/configure first")
    env = dict(os.environ, LLAMA_API_KEY="configuration-validation", LSW_GPU_POOL=str(c["pool_mb"]),
               LSW_GPU_EXCLUSIVE=str(c["pool_mb"]))
    subprocess.run([c["binaries"]["llama-swap"], "-config", str(config / "config.yaml"),
                    "-config-dir", str(config / "conf.d"), "-validate"], env=env, check=True)
    run([c["binaries"]["caddy"], "validate", "--config", config / "Caddyfile", "--adapter", "caddyfile"])
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    for name, label in LABELS.items():
        path = agents / (label + ".plist")
        source = config / path.name
        if path.exists() and path.read_bytes() != source.read_bytes():
            shutil.copy2(path, path.with_suffix(".plist.previous"))
        shutil.copy2(source, path)
        if a.activate:
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], capture_output=True)
            run(["launchctl", "bootstrap", f"gui/{os.getuid()}", path])
    if a.wired_limit_mb is not None:
        label = "app.llm-serving.wired-limit"
        staged = config / (label + ".plist")
        staged.write_bytes(plistlib.dumps({"Label": label, "RunAtLoad": True,
            "ProgramArguments": ["/usr/sbin/sysctl", "-w", f"iogpu.wired_limit_mb={a.wired_limit_mb}"]}))
        dest = Path("/Library/LaunchDaemons") / staged.name
        run(["sudo", "install", "-o", "root", "-g", "wheel", "-m", "644", staged, dest])
        subprocess.run(["sudo", "launchctl", "bootout", f"system/{label}"], capture_output=True)
        run(["sudo", "launchctl", "bootstrap", "system", dest])
    print("Services installed" + (" and activated" if a.activate else "; activate explicitly or at the next login"))
    print("LaunchAgents run in the logged-in user's session. This installer does not configure a reverse proxy on another machine.")


if __name__ == "__main__":
    main()
