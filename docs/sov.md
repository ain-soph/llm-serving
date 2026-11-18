# SOV-Transcribe-1：权重准备与服务

服务 `asr/sov/server.py` 用 mlx-audio fork（commit 固定在 `uv.lock`）运行 Salesforce SOV-Transcribe-1（Apache-2.0；
Whisper large-v3 编码器 + Gemma 4 26B-A4B MoE 解码器），接口与其他语音转写模型相同：`POST /v1/audio/transcriptions`，
json / text / srt / vtt / verbose_json。路由模板：`templates/conf.d/asr.yaml` 的 `sov-transcribe-1`。
速度、内存、准确率：[性能与资源报告](performance.md#语音转写)。

- 只支持英文和 fr / de / es / it / pt / nl；中文音频会被译成英文或输出乱码，服务对 `language=zh` 返回 400（中文请用 `qwen3-asr-1.7b`）。
- 单次最长 30 分钟。模型本身没有时间戳：srt / vtt / verbose_json 按约 30 s 分段转写，再用 Qwen3-ForcedAligner-0.6B 对齐（与 `qwen3-asr-1.7b` 共用）。
- wired 峰值约 +32 GB，不能与聊天模型共存；加载时 llama-swap 会先卸下聊天模型。

## 准备步骤

```sh
M="$HOME/models/sov-transcribe-1"     # 换成实际的模型目录
hf download salesforce-sovereign-intelligence/SOV-Transcribe-1 --local-dir "$M/official"   # BF16，53.3 GB
.venv/bin/python -I asr/sov/quantize.py "$M/official" "$M/mlx-q8"                           # 29.8 GB
```

- `quantize.py` 把解码器写成对称 8-bit（每 32 个权重一组，d = max|w|/127，与 llama.cpp Q8_0 同样舍入，scale 用 BF16）；
  音频编码器和投影层保持 BF16，MoE 路由保持 FP32。没用 MLX 默认的 affine 8-bit：它每组有零点偏移，误差大得多。
- 转换需要约 30 GB 内存。确认 `mlx-q8` 可用后可以删除 `official/`（本机已删），需要重新转换时再下载。
- 对齐器 `qwen3-forcedaligner-0.6b` 与 `qwen3-asr-1.7b` 的准备方式相同。
