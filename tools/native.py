"""Build pinned native engines without modifying production checkouts."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
LOCK = json.loads((ROOT / "native.lock.json").read_text())


def run(args, cwd=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def checkout(name):
    spec = LOCK["sources"][name]
    path = LOCAL / "src" / f"{name}-{spec['revision'][:12]}"
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        run(["git", "init", path])
        run(["git", "remote", "add", "origin", spec["url"]], path)
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    if current.stdout.strip() != spec["revision"]:
        run(["git", "fetch", "--depth=1", "origin", spec["revision"]], path)
        run(["git", "checkout", "--detach", spec["revision"]], path)
    status = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=path, text=True)
    if status.strip():
        raise RuntimeError(f"Tracked changes in {name}; build a new pinned commit instead")
    if (path / ".gitmodules").exists():
        run(["git", "submodule", "update", "--init", "--recursive", "--depth=1"], path)
    return path


def metal_headers():
    dest = LOCAL / "sdk/metal-cpp"
    if (dest / "Metal/Metal.hpp").exists():
        return dest
    spec = LOCK["metal_cpp"]
    cache = LOCAL / "downloads/metal-cpp.zip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(spec["url"], cache)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != spec["sha256"]:
        raise RuntimeError("Metal headers checksum mismatch")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cache) as archive:
        for member in archive.infolist():
            target = (dest.parent / member.filename).resolve()
            if not target.is_relative_to(dest.parent.resolve()):
                raise RuntimeError("Unsafe path in Metal header archive")
        archive.extractall(dest.parent)
    if not (dest / "Metal/Metal.hpp").exists():
        raise RuntimeError("Metal header archive did not contain Metal/Metal.hpp")
    return dest


def cmake(source, build, flags, targets, jobs):
    run(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *flags])
    run(["cmake", "--build", build, "--parallel", jobs, "--target", *targets])


def build(name, jobs):
    src = checkout(name)
    commands = []
    if name == "strata":
        ggml = checkout("strata-ggml")
        headers = metal_headers()
        flags = [f"-DMETAL_CPP_DIR={headers}", f"-DSTRATA_GGML_DIR={ggml}"]
        cmake(src / "metal", src / "build-metal-serving", flags, ["strata"], jobs)
        vision_flags = [f"-DLLAMA_DIR={ggml}", "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_VISION_METAL=ON",
                        "-DSTRATA_PORTABLE=OFF", "-DGGML_OPENMP=OFF"]
        cmake(src / "tools/vision", src / "build-vision", vision_flags, ["strata-vision"], jobs)
        products = [src / "build-metal-serving/strata", src / "build-vision/bin/strata-vision"]
        commands = [flags, vision_flags]
    elif name == "h3":
        run(["make", "-j", jobs, "h3", "h3_pack_int8"], src)
        products = [src / "h3", src / "h3_pack_int8"]
        commands = [["make", "h3", "h3_pack_int8"]]
    else:
        run(["cmake", "--preset", "metal-server"], src)
        run(["cmake", "--build", "--preset", "metal-server", "--parallel", jobs], src)
        products = [src / "build/metal-server/bin/nemo-speech"]
        commands = [["cmake", "--preset", "metal-server"]]
    records = []
    for p in products:
        linked = subprocess.check_output(["otool", "-L", p], text=True)
        records.append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "dynamic_libraries": linked})
    versions = {}
    for tool, args in {"clang": ["xcrun", "clang", "--version"], "sdk": ["xcrun", "--show-sdk-version"],
                       "cmake": ["cmake", "--version"], "ninja": ["ninja", "--version"]}.items():
        versions[tool] = subprocess.check_output(args, text=True).splitlines()[0]
    manifest = {"source": LOCK["sources"][name], "architecture": platform.machine(),
                "macos": platform.mac_ver()[0], "tools": versions, "flags": commands, "products": records,
                "runtime_assets": "Keep the full pinned checkout and build directories, including shaders and dylibs"}
    (src / "serving-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {name}; provenance recorded in {src / 'serving-build.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["strata", "h3", "nemo", "all"])
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 12))
    a = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Native builds require macOS on Apple Silicon")
    if a.jobs <= 0:
        parser.error("--jobs must be positive")
    for tool in ("git", "cmake", "ninja", "make", "xcrun", "otool"):
        if not shutil.which(tool):
            parser.error(f"Missing {tool}; run bootstrap.sh and install Apple Command Line Tools")
    run(["xcrun", "--find", "clang"])
    # "all" = the engines in service; nemo (Nemotron ASR, retired 2026-11-18) only when named
    for name in (["strata", "h3"] if a.component == "all" else [a.component]):
        build(name, a.jobs)


if __name__ == "__main__":
    main()
