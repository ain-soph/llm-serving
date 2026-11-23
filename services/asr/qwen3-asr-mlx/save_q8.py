"""Store Qwen3-ASR / Qwen3-ForcedAligner pre-quantized for zero-copy loading (2026-11-18, agent PA2).

  .venv/bin/python -I services/asr/qwen3-asr-mlx/save_q8.py SRC_DIR OUT_DIR [PLAN]
SRC_DIR is the official BF16 checkpoint (hf download Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ForcedAligner-0.6B); the service
reads <name>-q8 (platform/llama-swap/conf.d/asr.yaml). PLAN: '+'-joined asr_quant.QWEN keys, default the deployed
one (DEPLOYED, for both models). Quantizes on the GPU (the CPU kernels round differently), then writes
OUT_DIR/model.safetensors page-aligned (mx.save_safetensors(page_align=True), MLX module names, float16 floats) and
copies the config / tokenizer / preprocessor files. mlx-qwen3-asr reads the 8-bit modules from their .scales tensors
(8 bits, group 32), so no loader change is needed; set MLX_ZERO_COPY_LOAD=1 to map the file.
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w8a8"))
import asr_quant  # noqa: E402

DEPLOYED = "enc_q8+dec_q8+dec_q8_head+emb_q8"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    plan = sys.argv[3] if len(sys.argv) > 3 else DEPLOYED
    from mlx_qwen3_asr.load_models import load_model
    model, _ = load_model(str(src), dtype=mx.float16)
    counts = asr_quant.convert(model, asr_quant.plan(asr_quant.QWEN, plan), mx.float16)
    params = dict(tree_flatten(model.parameters()))
    mx.eval(params)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), params, metadata={"format": "mlx", "plan": plan},
                        page_align=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    cfg = json.loads((out / "config.json").read_text())
    cfg["quantization"] = {"group_size": asr_quant.mlx_q8.GROUP, "bits": 8}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "quant.json").write_text(json.dumps({"plan": plan, "converted": counts, "source": str(src)}, indent=1))
    print(json.dumps({"out": str(out), "plan": plan, "converted": counts,
                      "bytes": (out / "model.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
