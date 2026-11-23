"""Write SOV-Transcribe-1 for mlx-audio with symmetric 8-bit decoder weights (Q8_0-style, bf16 scales).

  .venv/bin/python -I standby/asr/sov/quantize.py MODELS/sov-transcribe-1/official MODELS/sov-transcribe-1/mlx-q8
  .venv/bin/python -I standby/asr/sov/quantize.py MODELS/sov-transcribe-1/official MODELS/sov-transcribe-1/mlx-q4 --bits 4

Every decoder Linear, expert (SwitchLinear) and the tied embedding / output head is stored as blocks of 32 inputs
with d = max|w| / 127 and q = round(w / d) in [-127, 127], like llama.cpp Q8_0 (rerank/q8_0_to_mlx.py). MLX's 8-bit
affine layout holds it exactly as u = q + 128, scale = d, bias = -128 d. The scales stay bfloat16 here (the model
runs in bfloat16 activations; Gemma activations are too large for float16), so d is rounded to bfloat16 first and
q is computed against the rounded d; a block maximum can then land on +-128 and is clipped to 127 (< 0.4%).
mlx-audio's default 8-bit (affine min/max, group 64) has a per-group zero offset that does not cancel; on the
Qwen3-Reranker-8B it gave 5-9x the Q8_0 error. Audio encoder, projector and MoE routers stay as stored
(BF16 / FP32), as in the published quantizations. Needs ~30 GB of memory (the quantized model before saving).
--bits 4 writes MLX's standard 4-bit affine, group 64 (the recipe of nativ-community/SOV-Transcribe-1-4bit), for
comparison. Weights are read lazily from the BF16 files and quantized one module at a time.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.convert import load_config
from mlx_audio.lm.convert import quantize_model, save_config, save_model
from mlx_audio.stt import load

GROUP = 32


def q8(w):
    """(packed uint32, bf16 scales, bf16 biases) for w [..., in]."""
    shape = w.shape
    blocks = w.astype(mx.float32).reshape(-1, shape[-1] // GROUP, GROUP)
    d = (mx.max(mx.abs(blocks), axis=-1) / 127).astype(mx.bfloat16)
    inv = mx.where(d == 0, 0, 1 / d.astype(mx.float32))
    q = mx.clip(mx.round(blocks * inv[..., None]), -127, 127)
    u = (q + 128).astype(mx.uint32).reshape(blocks.shape[0], shape[-1] // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
    lead = shape[:-1]
    return (packed.reshape(*lead, shape[-1] // 4), d.reshape(*lead, shape[-1] // GROUP),
            (-128 * d.astype(mx.float32)).astype(mx.bfloat16).reshape(*lead, shape[-1] // GROUP))


def main():
    src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    bits = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
    t = time.time()
    model = load(str(src), lazy=True)
    predicate = model.model_quant_predicate
    originals = {p: m.weight for p, m in model.named_modules()
                 if hasattr(m, "to_quantized") and m.weight.shape[-1] % GROUP == 0 and predicate(p, m)}
    group = GROUP if bits == 8 else 64
    model, config = quantize_model(model, load_config(src), group, bits, quant_predicate=predicate)
    for path, module in model.named_modules():
        if path in originals:
            if bits == 8:
                module.weight, module.scales, module.biases = q8(originals[path])
            del originals[path]
            mx.eval(module.weight, module.scales, module.biases)
    assert not originals, f"not quantized: {list(originals)[:5]}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "processor_config.json", "preprocessor_config.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    count = sum(1 for k, _ in tree_flatten(model.parameters()) if k.endswith(".scales"))
    save_model(out, model, donate_model=True)
    config["model_type"] = "sovereignai_transcribe"
    if bits == 8:
        config["quantization_note"] = "symmetric Q8_0-style: u = q + 128, scale = d (bf16), bias = -128 d, group 32"
    save_config(config, config_path=out / "config.json")
    print(f"{count} quantized tensors written to {out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
