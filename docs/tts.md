# 语音合成

服务 `services/tts/server.py` 用 mlx-audio fork（commit 固定在 `uv.lock`）运行 Qwen3-TTS 12Hz 1.7B（Apache-2.0；2026-11-23 起用预量化的 8 位权重），
接口 `POST /v1/audio/speech`（OpenAI 字段，音频边生成边发送）和 `GET /v1/audio/voices`。路由模板：`platform/llama-swap/conf.d/tts.yaml`。
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
# 存成 8 位（服务用的版本）：在仓库根目录运行；speech_tokenizer/ 以硬链接放进 -q8 目录
for v in CustomVoice Base VoiceDesign; do
  .venv/bin/python -I services/tts/save_q8.py "$M/1.7B-$v" "$M/1.7B-$v-q8"
done
```

服务读 `1.7B-<变体>-q8`（`MLX_ZERO_COPY_LOAD=1` 零拷贝映射）：talker、码本预测器、各 embedding 和输出头都是 8 位权重
（W8A16，group 32），激活 BF16，语音解码器保持官方 FP32。和 BF16 比（同一会话独占测量）：实时率低 19–23%，MLX 峰值少 1.5–1.8 GB，
16 句 × 3 变体的 ASR 回写错误都是 0%（与 BF16 相同）。三个 `-q8` 目录合计 7.19 GB（BF16 原版 12.2 GB，只在重新生成时需要，
生成后可以删除）。要改回 BF16，把 `tts.yaml` 里的 `--model` 指回不带 `-q8` 的目录即可。

## Fish Audio Drama 3（`fish-drama-3`，2026-11-23）

同一个 `services/tts/server.py`，模型目录的 `config.json` 是 `fish_drama3` 时走 mlx-audio 的 `fish_drama3`。路由模板：`platform/llama-swap/conf.d/tts-drama.yaml`。
35B MoE（3B 激活），表现力强：语气提示直接写在文本里（`[whispers]`、`[laughs]`），`<|speaker:N|>` 切换说话人；长文本整段交给模型，
模型自己分块并保持同一音色。权重 39 GiB，放不进聊天模型旁边，所以是**独占入口**（加载时卸下其他模型）。许可：Fish Audio Community / Research License（非商用）。

- `voice`：`--voices` 里的内置提示音（JSON `{名字: {wav, text}}`，例如 `zh`、`en`）；OpenAI 音色名或不填 = 模型自选音色（随 seed 变）；
  也可以发 `ref_audio`（1–60 s）+ `ref_text`（必须）克隆。`instructions` 不支持（400）。
- 输出 44.1 kHz；`pcm` 按 OpenAI 的定义重采样到 24 kHz。
- 速度（M5 Ultra）：约 2.6 倍实时，首段音频约 0.17 s；锁定内存约 +41 GB。
- 正确性：官方参照实现只能在 CUDA 上运行，这里用 ASR 回写（16 句错误 0%）验证，没有逐样本对官方的误差。

```sh
hf download mlx-community/fish-audio-drama-3-8bit --revision fedc7cc9 --local-dir "$HOME/models/fish-audio-drama-3-8bit"   # 39 GiB
# 内置提示音（可选）：services/tts/voices/voices.json 写 {"zh": {"wav": "zh.wav", "text": "<转写>"}, ...}，wav 放同目录
```

## IndexTTS 2.5

`indextts-2.5` 使用 Swift MLX 运行 [IndexTeam IndexTTS 2.5 的 FP16 转换权重](https://huggingface.co/mlx-community/IndexTTS-2.5-fp16)，以 `POST /v1/audio/speech` 提供参考音频克隆。语言可设为 `zh`、`en`、`ja`、`es`、`ar` 或 `auto`。模型原生输出 22,050 Hz，支持 TTS 服务的 WAV、MP3、FLAC、Opus、AAC、PCM 与 SSE。

上传 1–15 秒参考音频，`ref_audio` 接受 Base64 或音频 data URL。可选的 `emotion` 接受情绪名称或八个权重，顺序为 happy、angry、sad、afraid、disgusted、melancholic、surprised、calm。权重总和最多 1，`emo_alpha` 控制强度，默认 0.6。选择 `speed` 或 `duration_factor` 之一，取值范围 0.5–2.0。发音标注写在原文中，如 `银<行|HANG2>`。可用 `seed` 固定随机种子。

```python
import base64
import os
from pathlib import Path

import requests

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8080/v1/audio/speech",
    headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
    json={
        "model": "indextts-2.5",
        "input": "明天上午九点在图书馆见面。",
        "ref_audio": reference,
        "language": "zh",
        "emotion": "calm",
        "response_format": "wav",
    },
    timeout=120,
)
response.raise_for_status()
Path("speech.wav").write_bytes(response.content)
```

```sh
hf download mlx-community/IndexTTS-2.5-fp16 --revision 65644cd70da15309ffeb74aa03f3686bb04e3eb1 --local-dir "${LLM_MODELS:-$HOME/models}/indextts-2.5-mlx"
.venv/bin/python tools/native.py indextts
```

模型许可为 bilibili Model Use License Agreement。Swift 构建产物与 Metal 资源包保存在 `.local/build/indextts-2.5/out/Products/Release/`。引擎使用 `native.lock.json` 与 `services/tts/indextts/Package.resolved` 的固定版本。llama-swap 在第一个请求时启动 worker，空闲 1800 秒后卸载。

## CosyVoice3（2026-11-18 下线）

`cosyvoice3`（Fun-CosyVoice3-0.5B）已下线：Qwen3-TTS 已经覆盖需要的方言，CosyVoice3 依赖停更的第三方 mlx-audio-plus，
还要单独的 Python 环境。方言请用：

- `qwen3-tts` 的预设音色 `eric`（四川话）、`dylan`（北京话）；
- `qwen3-tts-design`，在 `instructions` 里描述口音（例如粤语）。

`services/tts/server.py` 里的 CosyVoice3 分支（`--voices`、`--s3tokenizer`）保留，以后要恢复时重新下载权重：

```sh
hf download mlx-community/Fun-CosyVoice3-0.5B-2512-fp16 --local-dir "$HOME/models/cosyvoice3/mlx-fp16"
hf download mlx-community/S3TokenizerV3 --local-dir "$HOME/models/cosyvoice3/s3tokenizer-v3"
```

它需要 mlx-audio-plus（上游 4c9ec6a）和 transformers < 5，与统一环境不兼容；测量数据见[性能报告](performance.md#语音合成cosyvoice3已下线)。
