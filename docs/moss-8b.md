# MOSS-Audio 与 MOSS-Music 8B

MOSS-Audio 用于语音、环境音和一般音频问答。MOSS-Music 面向音乐描述、歌词转写、曲式与乐理分析。每个系列都提供 Instruct 和 Thinking。四套权重均来自 OpenMOSS，许可为 Apache-2.0。

| 服务模型 ID | Hugging Face 检查点 |
| --- | --- |
| `moss-audio-8b-instruct` | [MOSS-Audio-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) |
| `moss-audio-8b-thinking` | [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) |
| `moss-music-8b-instruct` | [MOSS-Music-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) |
| `moss-music-8b-thinking` | [MOSS-Music-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) |

## 调用

向网关的 `POST /v1/chat/completions` 发送 OpenAI 聊天请求。音频用 `input_audio` 的 Base64 内容，或用 `audio_url` 的 `data:audio/…;base64,…` URL。支持文本消息、多条用户音频和 SSE 流式返回。Thinking 变体将模型输出的推理段放入 `reasoning_content`，回答放入 `content`。

```python
import base64
import os
from pathlib import Path

import requests

sample = base64.b64encode(Path("clip.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
    json={
        "model": "moss-audio-8b-instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"format": "wav", "data": sample}},
                {"type": "text", "text": "请用中文描述这段音频。"},
            ],
        }],
        "max_completion_tokens": 256,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

单模型同一时刻执行一个请求。每请求合计最多 600 秒音频、16 段音频与 64 MiB JSON。输入和预留输出合计最多 16,384 token，输出最多 8,192 token。设置 `stream: true` 启用 SSE，`stream_options.include_usage: true` 在结尾返回 token 用量。默认 `temperature: 0`。

## 权重准备

用 `hf` 下载下面的固定版本。目标是 `$LLM_MODELS` 中与服务 ID 同名的目录，默认位于 `~/models`。

```sh
models="${LLM_MODELS:-$HOME/models}"
hf download OpenMOSS-Team/MOSS-Audio-8B-Instruct --revision d4dc5a6d8cd79b43dcd82884c75e303b1ecd016d --local-dir "$models/moss-audio-8b-instruct" --exclude 'assets/*' --exclude 'examples/*'
hf download OpenMOSS-Team/MOSS-Audio-8B-Thinking --revision aacbb8036eb06dd968c105f91d2663af8fbee720 --local-dir "$models/moss-audio-8b-thinking" --exclude 'assets/*' --exclude 'examples/*'
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --revision fce7f8304e96cc2d3398b8106456cbb2ecec3139 --local-dir "$models/moss-music-8b-instruct" --exclude 'assets/*' --exclude 'examples/*'
hf download OpenMOSS-Team/MOSS-Music-8B-Thinking --revision 2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3 --local-dir "$models/moss-music-8b-thinking" --exclude 'assets/*' --exclude 'examples/*'
```

由生产 `.venv` 中已有的 mlx-audio 执行。每个变体按需加载，独立预约 GPU 内存，空闲 1800 秒后卸载。每次加载 MOSS 时，容量路由会在需要空间时卸载 Flash-Next。见[实测数据](performance.md#moss-audio-与-moss-music-8b)。
