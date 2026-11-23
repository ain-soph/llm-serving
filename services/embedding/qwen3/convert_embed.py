"""Serving copy of an embedding checkpoint for server.py (2026-11-16). Runs on the CPU (no GPU lock needed).

  .venv/bin/python embedding/qwen3/convert_embed.py HF_DIR OUT_DIR --fp16|--q8_0
--fp16: the BF16 weights stored as float16, so activations run in fp16 (10x closer to the float32 reference than
        bf16 at the same speed) without a load-time cast, which briefly held both copies (4B: 15 GiB peak).
--q8_0: exact llama.cpp Q8_0 in MLX 8-bit (fp16 scales) via rerank/q8_0_to_mlx.apply_q8_0, fp16 activations.
"""
import json, shutil, sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rerank"))
from q8_0_to_mlx import apply_q8_0   # noqa: E402

mx.set_default_device(mx.cpu)
src, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((src / "config.json").read_text())
weights = {}
for f in sorted(src.glob("*.safetensors")):
    weights.update(mx.load(str(f)))
weights = {k.removeprefix("model."): v for k, v in weights.items()}
extra = {k: weights.pop(k).astype(mx.float16) for k in [k for k in weights if k.startswith(("linear.", "lm_head"))]}
model = Qwen3Model(ModelArgs.from_dict(cfg))
model.load_weights(list(weights.items()))
model.set_dtype(mx.float16)
q8 = "--q8_0" in sys.argv
count = apply_q8_0(model) if q8 else 0
out.mkdir(parents=True, exist_ok=True)
params = dict(tree_flatten(model.parameters())) | extra
mx.save_safetensors(str(out / "model.safetensors"), params, metadata={"format": "mlx"})
if q8:
    cfg["quantization"] = {"group_size": 32, "bits": 8, "mode": "affine"}
cfg["torch_dtype"] = "float16"
(out / "config.json").write_text(json.dumps(cfg, indent=2))
for f in src.iterdir():
    if f.suffix != ".safetensors" and f.name not in ("config.json", "model.safetensors.index.json") and f.is_file():
        shutil.copy(f, out / f.name)
print(f"wrote {out}: float16" + (f", {count} Q8_0 linears / embeddings" if q8 else ""))
