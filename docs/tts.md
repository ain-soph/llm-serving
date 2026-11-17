# Qwen3-TTS：权重准备与服务

服务 `tts/server.py` 用 mlx-audio fork（commit 固定在 `uv.lock`）运行 Qwen3-TTS 12Hz 1.7B（Apache-2.0，BF16），
接口 `POST /v1/audio/speech`（OpenAI 字段，音频边生成边发送）和 `GET /v1/audio/voices`。路由模板：`templates/conf.d/tts.yaml`。
速度与内存：[性能与资源报告](performance.md#语音合成qwen3-tts)。

| 模型 ID | 官方仓库 | 用法 |
| --- | --- | --- |
| `qwen3-tts` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 个预设音色（OpenAI 音色名映射到它们）；`instructions` 控制情感 / 风格 |
| `qwen3-tts-clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 声音克隆：`ref_audio`（3–20 s）+ `ref_text`（它的准确转写） |
| `qwen3-tts-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `instructions` 描述要生成的声音 |

扩展字段：`language`、`seed`、`temperature`（0 = 贪心）；`speed` 用 ffmpeg atempo（音调不变）。
长文本按段落、句子切成每段 ≤ 220 字逐段生成；每段最多生成 50 + 6 × 字数 帧，防止失控生成。

## 准备步骤

```sh
M="$HOME/models/qwen3-tts"      # 换成实际的模型目录
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir "$M/1.7B-CustomVoice"
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local-dir "$M/1.7B-Base"
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir "$M/1.7B-VoiceDesign"
# 可选：三个变体的语音编解码器（speech_tokenizer/model.safetensors，0.68 GB）相同，比对后可以硬链接成一份
cmp "$M/1.7B-Base/speech_tokenizer/model.safetensors" "$M/1.7B-CustomVoice/speech_tokenizer/model.safetensors" && \
  ln -f "$M/1.7B-CustomVoice/speech_tokenizer/model.safetensors" "$M/1.7B-Base/speech_tokenizer/model.safetensors"
```

权重原样使用（BF16），不需要转换。三个入口合计约 12.2 GB（编解码器硬链接后）。
