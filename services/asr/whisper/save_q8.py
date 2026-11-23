"""Store Whisper large-v3 pre-quantized for zero-copy loading (2026-11-18, agent PA2).

  .venv/bin/python -I services/asr/whisper/save_q8.py SRC_DIR OUT_DIR [PLAN]
SRC_DIR is the FP16 MLX checkpoint (hf download mlx-community/whisper-large-v3-mlx); the service reads
whisper-large-v3-q8 (platform/llama-swap/conf.d/asr.yaml). PLAN: '+'-joined asr_quant.WHISPER keys using only q8 /
q8emb modes (8-bit weights, group 32, float16 activations), default the deployed one (DEPLOYED). Quantizes on the
GPU, writes OUT_DIR/weights.safetensors page-aligned and config.json with "quantization" (8 bits, group 32):
mlx_whisper.load_model rebuilds the 8-bit layers from their .scales tensors. Set MLX_ZERO_COPY_LOAD=1 to map the file.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w8a8"))
import asr_quant  # noqa: E402

DEPLOYED = "dec_q8+dec_xkv_q8+emb_q8"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    plan = sys.argv[3] if len(sys.argv) > 3 else DEPLOYED
    rules = asr_quant.plan(asr_quant.WHISPER, plan)
    if any(m == "w8a8" for _, m in rules):
        raise SystemExit("w8a8 layers cannot be rebuilt by mlx_whisper.load_model; use q8 / q8emb plans")
    from mlx_whisper.load_models import load_model
    model = load_model(str(src), dtype=mx.float16)
    counts = asr_quant.convert(model, rules, mx.float16)
    params = dict(tree_flatten(model.parameters()))
    mx.eval(params)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), params, metadata={"format": "mlx", "plan": plan},
                        page_align=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = {"group_size": asr_quant.mlx_q8.GROUP, "bits": 8}
    (out / "config.json").write_text(json.dumps(cfg, indent=4))
    (out / "quant.json").write_text(json.dumps({"plan": plan, "converted": counts, "source": str(src)}, indent=1))
    print(json.dumps({"out": str(out), "plan": plan, "converted": counts,
                      "bytes": (out / "weights.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
