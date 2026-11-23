# 客户端接入

所有模型都经网关 `http://<主机>:8080/v1`（OpenAI 兼容，Bearer key 来自状态目录的 `api_key`）。下面是我们给几个常用客户端填的设置；
具体的配置脚本属于各自的部署，不在本仓库。

| 客户端 | 功能 | 设置 |
| --- | --- | --- |
| Open WebUI | 对话 | OpenAI 连接 → 网关；模型 `qwen3.8-flash-next` |
| Open WebUI | 朗读（TTS） | 引擎 OpenAI，地址网关，模型 `qwen3-tts`，音色 `vivian` |
| Open WebUI | 语音转文字 | 引擎 OpenAI，模型 `qwen3-asr-1.7b`（或更快的 `qwen3-asr-1.7b-int8`） |
| Open WebUI | 出图／编辑 | 图片连接 → 网关；默认 `qwen-image-2.1-turbo`，可选 `-turbo-pe`、`qwen-image-2.1`、`minimax-h3-image` |
| Open WebUI | 检索 | 嵌入 `embeddinggemma-2`；混合检索 + 重排 `qwen3-reranker-4b`（TOP_K 20 → 6） |
| LobeChat | 对话、出图、TTS、ASR | 自定义 OpenAI 兼容提供方指向网关，按类型添加模型 ID（图片、`qwen3-tts`、`qwen3-tts-design`） |
| OpenClaw | TTS | `tts.provider = openai`，`baseUrl` 网关，`model` `qwen3-tts`，`speakerVoice` `vivian`，`responseFormat` mp3 |
| 任何 MCP 客户端 | 工具 | `clients/mcp/server.py`（见下） |

## MCP 服务（`clients/mcp/server.py`）

把本机模型包装成 MCP 工具（streamable HTTP，`/mcp`，Bearer 令牌）。工具：`list_models`、`chat`、`web_search`、`fetch_url`、
`generate_image`／`edit_image`、`generate_video`／`extend_video`／`get_video`、`transcribe`、`speak`、`embed`、`rerank`、`decide`、
`list_attachments`、`inspect_file`、`server_status`。

| 工具 | 后端 | 默认 |
| --- | --- | --- |
| `speak` | `/v1/audio/speech` | `qwen3-tts`、`vivian`；`-design` 用 `instructions` 描述音色（粤语用它）；`-clone` 用参考音频 + 文字稿 |
| `rerank` | `/v1/rerank` | `qwen3-reranker-4b`，返回按分数排序的 `[{index, score, text}]` |
| `decide` | `/v1/decisions`（或 SystemOne） | `elephant-1`（能与聊天模型共存）；`mammoth-1` 会卸下聊天模型 |
| `generate_image`／`edit_image` | `/v1/images/*` | `prompt_enhance`、`background=transparent` 只对 `qwen-image-*` 生效，返回 `revised_prompt` |

环境变量：`LLAMA_SWAP_URL`（网关，默认 `http://127.0.0.1:8080`）、`LLAMA_SWAP_KEY`、`MCP_TOKENS`（逗号分隔，每个客户端一个）、
`PUBLIC_BASE`（媒体链接前缀）、`ALLOWED_HOSTS`、`INPUT_HOSTS`（允许下载输入文件的内网主机）、`API_BASES`（`list_models`
里给出的 HTTP API 地址）、`SEARXNG_URL`；与 Open WebUI 的附件联动另需 `OWUI_URL`、`OWUI_API_KEY`、`CONTEXT_TOKENS`。
代码里的默认值都是本机 127.0.0.1；服务不和网关在同一台机器上时，env 文件至少要设下面 5 项（示例值，换成自己的）：

```sh
LLAMA_SWAP_URL=http://<gateway-lan-ip>:8080
PUBLIC_BASE=https://<public-domain>
INPUT_HOSTS=<chat-ui-s3-host>
API_BASES=LAN http://<gateway-lan-ip>:8080/v1, public https://<public-domain>/v1
ALLOWED_HOSTS=<public-domain>,llm-mcp:8000,localhost:8000,127.0.0.1:8000
```

```sh
docker build -t llm-mcp clients/mcp && docker run --env-file <your env file> -p 8000:8000 llm-mcp
python clients/mcp-tests/mcp_call.py http://127.0.0.1:8000/mcp <token> '[["list_models", {}]]'     # 冒烟测试
```

MCP 服务在容器里运行（Python 3.12，`clients/mcp/requirements.txt`），不用部署环境 `.venv`。
