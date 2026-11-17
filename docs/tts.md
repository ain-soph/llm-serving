# 语音合成：Qwen3-TTS 与 CosyVoice3

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

## CosyVoice3（`cosyvoice3`）

Fun-CosyVoice3-0.5B（Apache-2.0），中文 + 18 种方言 + 9 种语言，FP16，由同一个 `tts/server.py` 提供。
运行时是 [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) 上游 commit 4c9ec6a（未修改）。它和主环境的
mlx-audio 都安装成模块 `mlx_audio`，而且要求 transformers < 5，所以用单独的 `tts/cosyvoice/.venv`（Python 3.12）：

```sh
tools/cosyvoice_env.sh      # 按 tts/cosyvoice/requirements.txt 和 mlx-audio-plus@4c9ec6a 安装，都用 --no-deps
```

`--no-deps` 加上完整固定的包列表，复现的是线上验证过的那一套；PyPI 上的 mlx-audio-plus 0.1.8 太旧（缺提示标记，16 句错读 4 句），不要用。

- `voice`：内置提示音 `zh` / `en`；OpenAI 声音名都映射到 `zh`。
- `ref_audio`（≤ 30 s）+ `ref_text` = 零样本克隆；只给 `ref_audio` = 跨语言克隆。
- `instructions` = 指令模式（方言、情感、语速，例如“请用四川话表达。”），此时不用 `ref_text`。
- 服务先按 ≤ 220 字分段，模型再按官方规则分句，每句生成完就输出。MLX 缓存上限 1 GiB（不设时长文本会涨到 85 GB）。

### 权重与提示音

```sh
M="$HOME/models/cosyvoice3"
hf download mlx-community/Fun-CosyVoice3-0.5B-2512-fp16 --local-dir "$M/mlx-fp16"     # 1.6 GB
hf download mlx-community/S3TokenizerV3 --local-dir "$M/s3tokenizer-v3"               # 0.9 GB，语音分词器（否则运行时从 Hub 下载）

# 内置提示音不随仓库分发，按来源下载并核对（文字稿在 tts/cosyvoice/voices/voices.json）
V=tts/cosyvoice/voices
curl -fL -o "$V/zh.wav" https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/074ca6d/asset/zero_shot_prompt.wav
curl -fL -o "$V/en.wav" https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav
shasum "$V/zh.wav" "$V/en.wav"
# 71e0e66821a378bbaba607a85bbc378ba1f225f1  zh.wav  （CosyVoice 仓库自带示例，仓库许可 Apache-2.0）
# b805bee361be81c216af270a634991ba04c14e9f  en.wav  （Seed-TTS 评测集里的 Common Voice 片段，取自第三方镜像）
```

两段都是真人录音。`zh.wav` 的许可随 CosyVoice 仓库；`en.wav` 原始来源是 Common Voice（CC0），但我们拿到的是第三方镜像，
来源链不完整，所以两段都不放进本仓库。也可以换成自己有权使用的录音：改 `voices.json` 里的文件名和准确的文字稿。
