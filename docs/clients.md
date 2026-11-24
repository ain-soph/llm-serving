# 客户端接入

所有模型都经网关 `http://<主机>:8080/v1`（OpenAI 兼容，Bearer key 来自状态目录的 `api_key`）。下面是我们给几个常用客户端填的设置；
具体的配置脚本属于各自的部署，不在本仓库。

| 客户端 | 功能 | 设置 |
| --- | --- | --- |
| Open WebUI | 对话 | OpenAI 连接 → 网关；模型 `qwen3.8-flash-next`；两个 Flash-Next 模型的高级参数设 `top_k = 20`（Open WebUI 原样转发；服务端看到采样字段后按思考状态补齐其余项，不设则是贪心解码） |
| Open WebUI | 朗读（TTS） | 引擎 OpenAI，地址网关，模型 `qwen3-tts`，音色 `vivian` |
| Open WebUI | 语音转文字 | 引擎 OpenAI，模型 `qwen3-asr-1.7b`（默认；中文、中英混合、长音频）；另有 `whisper-large-v3`（多语种，词级时间戳） |
| Open WebUI | 出图／编辑 | 图片连接 → 网关；默认 `qwen-image-2.1-turbo`，可选 `-turbo-pe`、`qwen-image-2.1`、`minimax-h3-image`、`minimax-h3-image-pe`（官方提示词扩写，多 15–30 s；只是加进图片连接的模型列表，默认不变；`-pe-expander` 是内部模型，不列出） |
| Open WebUI | 检索 | 嵌入 `embeddinggemma-2`；混合检索 + 重排 `qwen3-reranker-4b`（TOP_K 20 → 6） |
| Open WebUI | 图片型 PDF | `PDF_EXTRACT_IMAGES=true`：上传扫描件／图片型 PDF 时对页内图片做 OCR（Open WebUI 自带 RapidOCR，不经网关）。2026-11-23 端到端：4 行中英文发票 2.0 s 识别正确 |
| Open WebUI | 任务模型（标题、标签、追问） | `TASK_MODEL_EXTERNAL = qwen3.8-flash-next`，`TASK_MODEL_PARAMS = {"chat_template_kwargs": {"enable_thinking": false}, "max_tokens": 1000}`（关思考，避免标题生成等待长推理）。2026-11-23 端到端：标题生成 200，0.4 s；返回的是 JSON 串 `{"title": …}`，Open WebUI 自己解析 |
| LobeChat | 对话、出图、TTS、ASR | 自定义 OpenAI 兼容提供方指向网关，按类型添加模型 ID（图片、`qwen3-tts`、`qwen3-tts-design`）；`minimax-h3-image-pe` 是一行 image 类型，参数同 `minimax-h3-image`，显示名注明“官方提示词扩写，较慢” |
| LobeChat | 浏览器工具 | 在跑 `lh connect` 的设备上装 [agent-browser](https://www.npmjs.com/package/agent-browser)（`npm i -g agent-browser`，再 `agent-browser install` 装测试用 Chrome），`lh connect restart` 后设备日志显示 “Browser tool enabled”。2026-11-23 端到端：`lh agent run --tool lobe-browser` 用 `qwen3.8-flash-next` 调用 navigate／readPage／snapshot／screenshot，答案正确 |
| LobeChat | 决策 | 不支持：LobeChat 只对内置模型目录里的决策模型 ID 走 `/decisions`，自定义 ID 退化为 `/chat/completions` 模拟，而网关上的决策模型只提供 `/v1/decisions` 和 SystemOne，没有对话接口；用 MCP `decide` |
| OpenClaw | 对话采样 | `agents.defaults.models["<provider>/qwen3.8-flash-next"].params.extra_body.top_k = 20`（同 Open WebUI） |
| OpenClaw | TTS | `tts.provider = openai`，`baseUrl` 网关，`model` `qwen3-tts`，`speakerVoice` `vivian`，`responseFormat` mp3 |
| OpenClaw | 出图（**目前不可用**） | `plugins.entries.litellm.enabled = true`；`models.providers.litellm`（`baseUrl` = 网关 `/v1`，`api: openai-completions`）；`agents.defaults.mediaModels.image` = `litellm/minimax-h3-image`，fallback `litellm/qwen-image-2.1-turbo`。OpenClaw 的 SSRF 防护拦下内网地址：`Blocked hostname or private/internal/special-use IP address`，请求没有到网关。见下“已知限制” |
| OpenClaw | 语音转文字（**目前不可用**） | `tools.media.models` 一项：提供方 `openai`、模型 `qwen3-asr-1.7b`、`baseUrl` 网关、`request.auth` bearer 网关 key、`capabilities: [audio]`。`request.auth` 没有生效，报 `ProviderAuthError: No API key found for provider "openai"`，请求没有到网关 |
| OpenClaw | 视频 | 不能原生接：OpenClaw（2026.11.6）的 `video_generate` 只有内置的云厂商提供方，不能指向自定义的 OpenAI `/v1/videos` 端点；用 MCP `generate_video` |
| 任何 MCP 客户端 | 工具 | `clients/mcp/server.py`（见下） |

## MCP 服务（`clients/mcp/server.py`）

把本机模型包装成 MCP 工具（streamable HTTP，`/mcp`，Bearer 令牌）。工具：`list_models`、`chat`、`web_search`、`fetch_url`、
`generate_image`／`edit_image`、`generate_video`／`extend_video`／`get_video`、`transcribe`、`speak`、`embed`、`rerank`、`decide`、
`list_attachments`、`inspect_file`、`server_status`。

| 工具 | 后端 | 默认 |
| --- | --- | --- |
| `speak` | `/v1/audio/speech` | `qwen3-tts`、`vivian`；`-design` 用 `instructions` 描述音色（粤语用它）；`-clone` 用参考音频 + 文字稿 |
| `rerank` | `/v1/rerank` | `qwen3-reranker-4b`，返回按分数排序的 `[{index, score, text}]` |
| `decide` | `/v1/decisions`（或 SystemOne） | `elephant-1`（能与聊天模型共存）；`joey-1`（2B，最快，也能共存，16K token 约 0.95 s）；`mammoth-1` 会卸下聊天模型 |
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

## 已知限制（2026-11-23）

- **OpenClaw 出图**：网关在内网地址（`http://<gateway-lan-ip>:8080`）上，被 OpenClaw 的 URL 抓取防护拦下，两个候选模型都没有发出请求。可选修法（待定）：找 OpenClaw 允许私网地址的开关（提供方级或 URL 抓取白名单），或 `baseUrl` 改用公网 https 域名。注意：两个本地候选都失败后，OpenClaw 会继续尝试其他已配置的出图提供方（例如用户自己的 OpenAI 账户）；在修好之前，默认出图可能走到云端。
- **OpenClaw 语音转文字**：`tools.media` 里给 `openai` 提供方写的 `request.auth` 没有被用上，OpenClaw 去找 OpenAI 的 API key，找不到就报错，请求没有发出。可选修法（待定）：`tools.media` 改为 `type: cli` 调 curl（带网关 key），或另建一个 apiKey 为网关 key 的独立提供方 ID，不碰用户的 `openai` 提供方。
- **llama-swap playground 的出图尺寸**：尺寸下拉框的选择存在浏览器 localStorage（`playground-image-size`），每次请求都带上；“总是 1792x1024” 是这个浏览器记住了上次的选择，在下拉框里改选即可。没有“跟随模型默认”的选项或配置项，不改。
