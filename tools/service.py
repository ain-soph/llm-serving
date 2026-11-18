"""Run a service with the same Python environment and explicit local configuration."""
import argparse
import json
import os
from pathlib import Path


def environment(c):
    b, state = c["binaries"], Path(c["state"])
    # launchd does not inherit an interactive shell's PATH.
    paths = [str(Path(c["python"]).parent), str(Path(c["root"]) / ".local/bin")]
    paths += [str(Path(v).parent) for v in b.values()]
    paths += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    return dict(os.environ, PATH=os.pathsep.join(dict.fromkeys(paths)),
                LLM_ROOT=c["root"], LLM_MODELS_DIR=c["models"], LLM_STATE_DIR=c["state"],
                LLM_PYTHON=c["python"], LLM_NEMO_EXE=b["nemo"],
                FFMPEG=b["ffmpeg"], FFPROBE=b["ffprobe"], H3_FFMPEG=b["ffmpeg"], H3_FFPROBE=b["ffprobe"],
                H3_VIDEO_SOURCE=c["sources"]["h3"], H3_VIDEO_BIN=b["h3"],
                H3_VIDEO_LSW_URL="http://127.0.0.1:8081", H3_VIDEO_LSW_KEY_FILE=str(state / "api_key"),
                H3_VIDEO_FETCH_HOSTS=",".join(c["fetch_hosts"]), H3_VIDEO_SHARE_BASE=c["public_base"],
                H3_MEDIA_IMAGES=str(state / "media/images"),
                HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1",
                XDG_DATA_HOME=str(state / "caddy"), XDG_CONFIG_HOME=str(state / "caddy"),
                LSW_GPU_POOL=str(c["pool_mb"]), LSW_GPU_EXCLUSIVE=str(c["pool_mb"]))


def command(service, c):
    config, b = Path(c["config"]), c["binaries"]
    if service == "router":
        return [b["llama-swap"], "-config", str(config / "config.yaml"), "-config-dir", str(config / "conf.d"),
                "-listen", "127.0.0.1:8081"]
    if service == "gateway":
        return [b["caddy"], "run", "--config", str(config / "Caddyfile"), "--adapter", "caddyfile"]
    if service == "health":
        return [c["python"], str(Path(c["root"]) / "tools/health-check")]
    return [c["python"], str(Path(c["root"]) / "h3-video/server.py"), "--host", "127.0.0.1", "--port", "8090",
            "--models", str(Path(c["models"]) / "minimax-h3"), "--model-id", "minimax-h3-video", "--files-only",
            "--out-dir", str(Path(c["state"]) / "media/videos")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", choices=["router", "gateway", "files", "health"])
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    env = environment(c)
    if a.service == "router":
        env["LLAMA_API_KEY"] = (Path(c["state"]) / "api_key").read_text().strip()
        if not env["LLAMA_API_KEY"]:
            p.error("Local API key is empty")
    os.chdir(c["root"])
    cmd = command(a.service, c)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
