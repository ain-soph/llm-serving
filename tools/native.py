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


def run(args, cwd=None, env=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True, env=env)


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
    src = LOCAL / "build/indextts-2.5" if name == "indextts" else checkout(name)
    commands = []
    env = dict(os.environ)
    if name == "strata":
        # upstream layout: engine/metal (ggml vendored, Strata's own CPU quant layer), terminal/tools/vision (own tower)
        headers = metal_headers()
        flags = [f"-DMETAL_CPP_DIR={headers}", "-DSTRATA_OWN_CPU_QUANT=ON"]
        cmake(src / "engine/metal", src / "build-metal-serving", flags, ["strata"], jobs)
        vision_flags = ["-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_VISION_METAL=ON", "-DSTRATA_PORTABLE=OFF"]
        cmake(src / "terminal/tools/vision", src / "build-vision", vision_flags, ["strata-vision"], jobs)
        # Hy4-Flash Metal runner (hy4-flash entry): upstream's clang-only script, kernels embedded, into build-hy4/bin;
        # the script reads the sources from $HY4_DIR/wt, here a link to the checkout itself
        hy4 = src / "build-hy4"
        hy4.mkdir(exist_ok=True)
        if not (hy4 / "wt").exists():
            (hy4 / "wt").symlink_to(src)
        run(["bash", src / "engine/src/hy4flash/metal/build_mac.sh", "run"], src, env={**os.environ, "HY4_DIR": str(hy4)})
        products = [src / "build-metal-serving/strata", src / "build-vision/bin/strata-vision",
                    hy4 / "bin/strata-hy4-metal"]
        commands = [flags, vision_flags, ["HY4_DIR=build-hy4", "engine/src/hy4flash/metal/build_mac.sh", "run"]]
    elif name == "h3":
        run(["make", "-j", jobs, "h3", "h3_pack_int8"], src)
        products = [src / "h3", src / "h3_pack_int8"]
        commands = [["make", "h3", "h3_pack_int8"]]
    elif name == "indextts":
        package = ROOT / "services/tts/indextts"
        resolved = json.loads((package / "Package.resolved").read_text())
        spec = LOCK["sources"][name]
        pin, = (entry for entry in resolved["pins"] if entry["identity"] == "mlx-indextts2-swift")
        if pin["location"] != spec["url"] or pin["state"]["revision"] != spec["revision"]:
            raise RuntimeError("IndexTTS SwiftPM pin differs from native.lock.json")
        xcode = Path("/Applications/Xcode.app/Contents/Developer")
        if xcode.is_dir():
            env.setdefault("DEVELOPER_DIR", str(xcode))
        flags = ["swift", "build", "--package-path", str(package), "--scratch-path", str(src),
                 "--build-system", "swiftbuild", "--configuration", "release", "--product", "indextts-worker",
                 "--jobs", str(jobs), "--only-use-versions-from-resolved-file"]
        run(flags, env=env)
        products = [src / "out/Products/Release/indextts-worker"]
        commands = [flags]
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
        versions[tool] = subprocess.check_output(args, text=True, env=env).splitlines()[0]
    if name == "indextts":
        versions["swift"] = subprocess.check_output(["swift", "--version"], text=True, env=env).splitlines()[0]
    manifest = {"source": LOCK["sources"][name], "architecture": platform.machine(),
                "macos": platform.mac_ver()[0], "tools": versions, "flags": commands, "products": records,
                "runtime_assets": "Keep the full pinned checkout and build directories, including shaders and dylibs"}
    (src / "serving-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {name}; provenance recorded in {src / 'serving-build.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["strata", "h3", "indextts", "nemo", "all"])
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
    for name in (["strata", "h3", "indextts"] if a.component == "all" else [a.component]):
        build(name, a.jobs)


if __name__ == "__main__":
    main()
