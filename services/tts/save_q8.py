"""Store a Qwen3-TTS 1.7B variant pre-quantized for zero-copy loading (2026-11-18, agent PA2).

  .venv/bin/python -I services/tts/save_q8.py SRC_DIR OUT_DIR [PLAN]
PLAN: '+'-joined PLANS keys, default the deployed one (every key: talker+talker_emb+talker_head+cp+cp_heads).
SRC_DIR is the official BF16 checkpoint (hf download Qwen/Qwen3-TTS-12Hz-1.7B-<variant>). Quantizes on the GPU, writes OUT_DIR/model.safetensors page-aligned
(mx.save_safetensors(page_align=True), MLX names and layouts, bfloat16 floats) without the speech tokenizer, adds
"quantization" (8 bits, group 32) to config.json, copies the other files and hard-links the speech_tokenizer/ files
(one copy shared by every variant). mlx-audio rebuilds the 8-bit layers from their .scales tensors; set MLX_ZERO_COPY_LOAD=1 to map the file.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "asr/w8a8"))
import asr_quant  # noqa: E402  (generic per-path nn.Linear / nn.Embedding converter)

# Per-class plans (2026-11-18, agent PA2; also used by eval/tts_variant.py). "q8" = 8-bit weights, group 32 (W8A16).
PLANS = {
    # talker: the 28-layer Qwen3 LM, one step per 80 ms frame
    "talker": [(r"^talker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$", "q8")],
    "talker_emb": [(r"^talker\.model\.text_embedding$", "q8emb")],
    "talker_head": [(r"^talker\.(codec_head|text_projection\.linear_fc[12])$", "q8"),
                    (r"^talker\.model\.codec_embedding$", "q8emb")],
    # code predictor: 5-layer transformer run 15 times per frame (one step per residual codebook)
    "cp": [(r"^talker\.code_predictor\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$", "q8")],
    "cp_heads": [(r"^talker\.code_predictor\.(lm_head\.\d+|small_to_mtp_projection)$", "q8"),
                 (r"^talker\.code_predictor\.model\.codec_embedding\.\d+$", "q8emb")],
}
DEPLOYED = "talker+talker_emb+talker_head+cp+cp_heads"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    plan = sys.argv[3] if len(sys.argv) > 3 else DEPLOYED
    from mlx_audio.tts.utils import load_model
    model = load_model(str(src))
    counts = asr_quant.convert(model, sum((PLANS[k] for k in plan.split("+")), []), mx.bfloat16)
    params = {k: v for k, v in tree_flatten(model.parameters()) if not k.startswith("speech_tokenizer.")}
    mx.eval(params)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), params, metadata={"format": "mlx", "plan": plan},
                        page_align=True)
    for f in src.iterdir():
        if f.name in ("model.safetensors", "config.json") or f.name.startswith("."):
            continue
        dst = out / f.name
        if f.is_dir():  # speech_tokenizer/: one copy shared by every variant, so hard links again
            shutil.copytree(f, dst, copy_function=os.link, dirs_exist_ok=True)
        elif not dst.exists():
            shutil.copy2(f, dst)
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = {"group_size": 32, "bits": 8}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "quant.json").write_text(json.dumps({"plan": plan, "converted": counts, "source": str(src)}, indent=1))
    print(json.dumps({"out": str(out), "plan": plan, "converted": counts,
                      "bytes": (out / "model.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
