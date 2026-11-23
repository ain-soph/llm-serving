"""Write the pre-quantized Decider checkpoint (2026-11-18): load the official checkpoint with the per-class W8A8 choice
quantized at load (as the server did), then save it (decider_mlx.MLXDecider.save_mix) so the server maps it zero-copy
and the official checkpoint need not stay on disk.

usage: decisions/decider/save_mix.py SOURCE OUT [--w8a8-names a,b,...] [--dtype float16] [--revision REPO@SHA]
Default classes: every W8A8 class except down_proj (its W8A8 was most of the error for 2-5% of the time,
reports/optimization-audit-20261117.md section 3). Rebuild: hf download perplexity-ai/pplx-decider-flash-v1.3
--local-dir SOURCE, run this, delete SOURCE.
"""
import argparse
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("out")
ap.add_argument("--w8a8-names", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--revision")
args = ap.parse_args()
sys.path.insert(0, str(Path(args.source) / "source/src"))     # the official pplx_decider package (see server.py)
from decider_mlx import MLXDecider   # noqa: E402

t = time.perf_counter()
d = MLXDecider(args.source, dtype=args.dtype, vision=True, w8a8=tuple(args.w8a8_names.split(",")))
load = time.perf_counter() - t
t = time.perf_counter()
out = d.save_mix(args.out, args.source, args.revision)
print(f"loaded + quantized in {load:.1f} s, saved {out} in {time.perf_counter() - t:.1f} s")
