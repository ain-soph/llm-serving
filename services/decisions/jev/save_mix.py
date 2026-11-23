"""Write a pre-quantized copy of a Base Labs checkpoint (2026-11-18): quantize at load exactly as the server does,
then save the result (jev_mlx.MLXJev.save_mix) so the server maps it zero-copy instead of re-quantizing the bfloat16
checkpoint on every load, and the bfloat16 checkpoint need not stay on disk.

usage: services/decisions/jev/save_mix.py SOURCE OUT --quant {q8,w8a8} [--w8a8-skip a,b] [--dtype bfloat16]
Rebuild: download the official checkpoint (hf download baselabs/<model> --local-dir SOURCE), run this, delete SOURCE.
"""
import argparse
import time

from jev_mlx import MLXJev

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("out")
ap.add_argument("--quant", required=True, choices=["q8", "w8a8"])
ap.add_argument("--w8a8-skip", default="")
ap.add_argument("--dtype", default="bfloat16")
args = ap.parse_args()
engine = MLXJev(args.source, dtype=args.dtype, quant=args.quant,
                w8a8_skip=[n for n in args.w8a8_skip.split(",") if n])
t = time.perf_counter()
out = engine.save_mix(args.out, args.source)
print(f"loaded + quantized in {engine.load_seconds:.1f} s, saved {out} in {time.perf_counter() - t:.1f} s")
