"""Quick GPU check of mlx_w8a8 (2026-11-17): W8A8Linear vs float matmul, zero-copy load == mx.load, GEMM speed.
  rerank/.venv-mlx/bin/python common/w8a8_selftest.py [MODEL_SAFETENSORS]"""
import sys, time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_w8a8 as w  # noqa: E402

mx.random.seed(0)
for dtype in (mx.bfloat16, mx.float16):
    x = mx.random.normal((300, 2560)).astype(dtype)
    W = (mx.random.normal((9728, 2560)) * 0.02).astype(dtype)
    layer = w.W8A8Linear(2560, 9728)
    layer.weight, layer.scale = w.quantize_weight(W)
    ref = (x.astype(mx.float32) @ W.astype(mx.float32).T)
    y = layer(x).astype(mx.float32)
    y2 = layer(x).astype(mx.float32)         # second call reuses the cached activation quantization
    w.release()
    print(f"{dtype}: W8A8 rel err {(mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item():.4f}, "
          f"float {dtype} rel err {(mx.linalg.norm((x @ W.T).astype(mx.float32) - ref) / mx.linalg.norm(ref)).item():.4f}, "
          f"cached call identical: {mx.array_equal(y, y2).item()}")

if len(sys.argv) < 2:
    sys.exit("usage: w8a8_selftest.py <w8a8 model.safetensors>")
f = Path(sys.argv[1])
a, b = w.load(f), mx.load(str(f))
print(f"zero-copy load == mx.load for all {len(a)} tensors of {f.parent.name}: "
      f"{all(mx.array_equal(a[k], b[k]).item() and a[k].dtype == b[k].dtype for k in b)}")

for rows, k, n in ((8192, 2560, 9728), (8192, 9728, 2560), (8192, 4096, 12288), (2048, 1024, 3072)):
    x = mx.random.normal((rows, k)).astype(mx.bfloat16)
    W = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    layer = w.W8A8Linear(k, n)
    layer.weight, layer.scale = w.quantize_weight(W)
    mx.eval(x, W, layer.parameters())
    def bench(fn, reps=20):
        mx.eval(fn()); w.release()
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(fn()); w.release()
        return (time.perf_counter() - t) / reps
    tf, tq = bench(lambda: x @ W.T), bench(lambda: layer(x))
    flops = 2 * rows * k * n
    print(f"{rows}x{k}x{n}: bf16 {tf * 1e3:.2f} ms ({flops / tf / 1e12:.0f} TFLOPS), "
          f"W8A8 incl. quantization {tq * 1e3:.2f} ms ({flops / tq / 1e12:.0f} TOPS), x{tf / tq:.2f}")
