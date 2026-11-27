# 客户端接入

所有模型都经网关 `http://<主机>:8080/v1`（OpenAI 兼容，Bearer key 来自状态目录的 `api_key`）。下面是我们给几个常用客户端填的设置；
具体的配置脚本属于各自的部署，不在本仓库。

| 客户端 | 功能 | 设置 |
| --- | --- | --- |
| Open WebUI | 对话 | OpenAI 连接 → 网关；模型 `qwen3.8-flash-next`，也是管理员设置的默认模型 `DEFAULT_MODELS = qwen3.8-flash-next`（不设时新对话会选模型列表里第一个可见的模型，而视频管道函数 `minimax_h3_video.*` 排在 OpenAI 连接的模型前面）；两个 Flash-Next 模型的高级参数设 `top_k = 20`（Open WebUI 原样转发；服务端看到采样字段后按思考状态补齐其余项，不设则是贪心解码） |
| Open WebUI | 按需对话模型 | OpenAI 连接的模型列表在两个 Flash-Next 之后加 `hy4-flash`，模型设置名称 “Tencent Hy4 Flash”、`temperature 0.9`、`top_p 1.0`（官方建议）、原生函数调用，能力和工具同 Flash-Next；默认模型仍是 Flash-Next，任务模型跟随当前对话模型（见“任务模型”一行）。说明写“按需加载，与 Flash-Next 互相切换，切换后第一次请求多等约 10–15 秒” |
| Open WebUI | 无审查对话模型 | `qwen3.8-flash-next-uncensored`：名称 “Qwen3.8 Flash Next Uncensored”，说明写“按需加载，与 Flash-Next 互相切换，切换后第一次请求多等约 20–30 秒”。11-24 至 11-26 权重只在 Mac mini 归档里，名称和说明曾标“已归档”；2026-11-26 首次请求时由 `bin/archive-restore` 拷回本机（按清单校验），标注已去掉。名称和说明的唯一来源是 `clients/open-webui/set_default_model.py` 的 `LABELS`（`labels` 模式重写）；以后再归档、本机缺副本时，在那里改标注再跑 |
| Open WebUI | 朗读（TTS） | 引擎 OpenAI，地址网关，模型 `qwen3-tts`，音色 `vivian` |
| Open WebUI | 语音转文字 | 引擎 OpenAI，模型 `qwen3-asr-1.7b`（默认；中文、中英混合、长音频）；另有 `whisper-large-v3`（多语种，词级时间戳） |
| Open WebUI | 出图／编辑 | 图片连接 → 网关；默认 `qwen-image-2.1-turbo`，可选 `-turbo-pe`、`qwen-image-2.1`、`minimax-h3-image`、`minimax-h3-image-pe`（官方提示词扩写，多 15–30 s；只是加进图片连接的模型列表，默认不变；`-pe-expander` 是内部模型，不列出） |
| Open WebUI | 检索 | 嵌入 `embeddinggemma-2`；混合检索 + 重排 `qwen3-reranker-4b`（TOP_K 20 → 6） |
| Open WebUI | 图片型 PDF | `PDF_EXTRACT_IMAGES=true`：上传扫描件／图片型 PDF 时对页内图片做 OCR（Open WebUI 自带 RapidOCR，不经网关）。2026-11-23 端到端：4 行中英文发票 2.0 s 识别正确 |
| Open WebUI | 任务模型（标题、标签、追问） | `TASK_MODEL_EXTERNAL` 为空 = 用当前对话模型（2026-11-26 起；11-23 至 11-26 固定为 `qwen3.8-flash-next`，两个 Flash-Next 不能同驻，uncensored / hy4-flash 对话每轮回复后都被换成 censored 版做追问/标题/标签，下一轮再换回；`clients/open-webui/set_task_model.py`，视频管道 v1.2.1 对任务请求直接返回空，不渲染）；`TASK_MODEL_PARAMS = {"chat_template_kwargs": {"enable_thinking": false}, "max_tokens": 1000}`（关思考，避免标题生成等待长推理）。2026-11-23 端到端：标题生成 200，0.4 s；返回的是 JSON 串 `{"title": …}`，Open WebUI 自己解析 |
| LobeChat | 对话、出图、TTS、ASR | 自定义 OpenAI 兼容提供方指向网关，按类型添加模型 ID（图片、`qwen3-tts`、`qwen3-tts-design`）；`minimax-h3-image-pe` 是一行 image 类型，参数同 `minimax-h3-image`，显示名注明“官方提示词扩写，较慢” |
| LobeChat | 按需对话模型 | 同一提供方加 chat 类型 `hy4-flash`（显示名 “Tencent Hy4 Flash”），能力：视觉、函数调用、推理，上下文 131072；说明同 Open WebUI；默认模型仍是 `qwen3.8-flash-next` |
| LobeChat | 无审查对话模型 | `qwen3.8-flash-next-uncensored` 的 `ai_models` 行：显示名 “Qwen3.8 Flash Next Uncensored”，说明同 Open WebUI（行里的显示名优先于 `VLLM_MODEL_LIST` 里的名字）。11-24 至 11-26 这一行曾标“已归档”；2026-11-26 权重拷回后按用户要求改回（容器内 psql 直接改这一行，没有生成脚本；备份 Mac mini `~/docker/config/lobechat/ai_providers_models.sql.20261126-uncens`） |
| LobeChat | 浏览器工具 | 在跑 `lh connect` 的设备上装 [agent-browser](https://www.npmjs.com/package/agent-browser)（`npm i -g agent-browser`，再 `agent-browser install` 装测试用 Chrome），`lh connect restart` 后设备日志显示 “Browser tool enabled”。2026-11-23 端到端：`lh agent run --tool lobe-browser` 用 `qwen3.8-flash-next` 调用 navigate／readPage／snapshot／screenshot，答案正确 |
| LobeChat | 决策 | 不支持：LobeChat 只对内置模型目录里的决策模型 ID 走 `/decisions`，自定义 ID 退化为 `/chat/completions` 模拟，而网关上的决策模型只提供 `/v1/decisions` 和 SystemOne，没有对话接口；用 MCP `decide` |
| OpenClaw | 对话采样 | `agents.defaults.models["<provider>/qwen3.8-flash-next"].params.extra_body.top_k = 20`（同 Open WebUI） |
| OpenClaw | TTS | `tts.provider = openai`，`baseUrl` 网关，`model` `qwen3-tts`，`speakerVoice` `vivian`，`responseFormat` mp3 |
| OpenClaw | 出图 | `plugins.entries.litellm.enabled = true`；`models.providers.litellm`（`baseUrl` = 网关 `/v1`，`api: openai-completions`，`request.allowPrivateNetwork: true`：只对这一个提供方放行内网地址，不是全局 SSRF 开关）；`agents.defaults.mediaModels.image` = `litellm/minimax-h3-image`，fallback 只有 `litellm/qwen-image-2.1-turbo`。写了显式的 primary/fallbacks，对话里的 `image_generate` 只试这两个，不会自动追加其他已登录的提供方。CLI `openclaw infer image generate` 一律带 `--model litellm/<模型>`（原因见“已知限制”）；`--output` 要指向真实目录（`/tmp` 是符号链接，会报 `directory component must be a directory`），或者不写 `--output`，图就存进 `~/.openclaw/media/generated/` |
| OpenClaw | 语音转文字 | `tools.media.models` 只有一项 `type: cli`：`~/.openclaw/local-asr/transcribe {{AttachmentPath}} --model qwen3-asr-1.7b --language {{Language}}`，`capabilities: [audio]`，`timeoutSeconds 600`。这个脚本（Python 标准库）运行时从 `openclaw.json` 的 `models.providers.litellm` 读网关地址和本机 key，不另存 key，key 也不出现在进程参数里；它 POST `/v1/audio/transcriptions`，只输出文本。没有用 `openai` 提供方：OpenClaw 2026.11.6 里自定义的提供方 ID 只有图片理解能力，做不了音频；借用 `openai` 又会和用户自己的 OpenAI 登录冲突。列表是显式的，所以失败也不会自动转去别的提供方 |
| OpenClaw | 视频 | 不能原生接：OpenClaw（2026.11.6）的 `video_generate` 只有内置的云厂商提供方，不能指向自定义的 OpenAI `/v1/videos` 端点；用 MCP `generate_video` |
| 任何 MCP 客户端 | 工具 | `clients/mcp/server.py`（见下） |

## Open WebUI 终端（Machines，2026-11-26）

Open WebUI 0.13.5 的 Machines（原 Terminal Server）接了两个 [Open Terminal](https://github.com/open-webui/open-terminal)（0.18.7）。
对话输入框的终端选择器里每个对话选一个；两个都只有管理员可用（连接的 `access_grants` 为空）。

| 终端 | 用途 | 位置 | 确认 |
| --- | --- | --- | --- |
| **容器终端**（id `container`） | **默认、优先用**：写代码、跑脚本、pip / git、处理文件 | Mac mini 容器 `open-terminal`（官方镜像，按 digest 固定），工作目录 = 宿主 `~/docker/data/ai/workspace`（容器内 `~/workspace`） | 不用确认，模型可连续多步调用 |
| **Mac mini 本机**（id `mac-mini`） | 只在需要改 Mac mini 本机（Homebrew、launchd、`~/docker` 配置等）时手动选 | 宿主原生进程，以登录用户身份运行（没有加 sudo 权限），只监听 `127.0.0.1:8722` | **每次调用都要点“批准”** |

- 三个对话模型（`qwen3.8-flash-next`、`-uncensored`、`hy4-flash`）的默认终端 `meta.machineId = container`：新对话自动选容器终端；
  选了终端的新对话里 Code Interpreter 默认关闭（Open WebUI 的设计，终端取代它），需要时在对话里再打开。
- **确认方式**：Open WebUI 只有“每个对话一个”审批模式（full / ask / auto），没有按工具或按终端的设置。
  `chat.tool_permissions.enable = true`，auto 关闭，所以界面默认仍是 full（生图等其他工具和以前一样不用确认）；
  全局过滤函数 `machine_approval_guard`（active + global）在请求选的是 `mac-mini` 时把该轮改成 ask，用户在对话里选了 full 也一样。
  批准、续跑、子代理、自动化都经同一个对话入口，都受它管。
- **容器终端的隔离**：uid 1000、无任何 capability、`no-new-privileges`（镜像里 user 的免密 sudo 因此失效）、根文件系统只读、
  不挂 docker.sock、不映射宿主端口；可写的只有 `~/workspace`（唯一的宿主目录）、`/home/user`（命名卷 `open-terminal-home`，
  `pip install --user`、git 配置、缓存都在这里，重建容器不丢）和 `/tmp`（tmpfs 1 GB）；限 2 CPU、3 GB 内存、512 进程。
  网络：容器加入 `netguard` 小容器的网络命名空间；`netguard` 只在 docker 网络 `owui-terminal` 上（另一个成员是 open-webui），
  只有它有 NET_ADMIN，用 iptables 拒绝终端发起的、去往私有 / 宿主 / 局域网 / 保留网段的新连接，公网放行，IPv6 关闭。
  这样做的原因：Open WebUI 信任自己 8080 端口上的 `X-authentik-email` 请求头（认证由 SWAG + Authentik 做），
  能直连 open-webui:8080 的进程就能冒充管理员；终端和 open-webui 在同一网络，所以必须在网络层挡住。
  2026-11-26 实测：open-webui、`ipv6` 网络上的 lobechat / authentik-postgresql / swag / llm-mcp、宿主（含本机终端端口）、
  局域网全部连不上，iptables 改不了；pypi、github 可以访问。
- **本机终端**：uv 建的独立 venv（不碰系统 Python）；desktop / browser 关闭（macOS 上 desktop 默认就是控制真实桌面）；
  `claude`、`codex`、`hermes` 等编码 agent 的发现被关掉（`OPEN_TERMINAL_*_COMMAND=/usr/bin/false`），否则它们会作为模型出现，
  在宿主上直接运行而不经过逐次确认。局域网其他机器连不上（只绑 loopback），容器终端也连不上（防火墙）；open-webui 经 OrbStack
  的 `host.docker.internal` 访问。launchd 下的进程没有“完全磁盘访问”，碰不到桌面、文稿、下载等受保护目录。

文件（Mac mini；本仓库外的部署副本在 `clients/open-terminal/`，不公开）：

| 项 | 位置 |
| --- | --- |
| 容器终端 compose / 防火墙 | `~/docker/compose/ai/open-terminal/{compose.yaml,netguard.sh}` |
| 容器终端 key | `~/docker/config/open-terminal/open-terminal.env`（600；Open WebUI 连接里存同一个） |
| open-webui compose | `~/docker/compose/ai/open-webui/compose.yaml`：多加网络 `owui-terminal`（镜像仍是 `:main`，watchtower 照常更新） |
| 本机终端 | `~/.local/share/open-terminal-host/{venv,run.sh}`，key `~/.config/open-terminal-host/api-key`（600） |
| 本机终端 LaunchAgent | `~/Library/LaunchAgents/com.local.open-terminal-host.plist`，日志 `~/Library/Logs/open-terminal-host/` |
| Open WebUI 设置 | `clients/open-webui/set_terminals.py`（show / backup / apply / rollback）、过滤函数 `machine_approval_guard.py`、验收 `test_terminals.py` |

启停：

```sh
# 容器终端
cd ~/docker/compose/ai/open-terminal && docker compose up -d      # 停：docker compose down（工作目录、home 卷保留）
# 本机终端
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.open-terminal-host.plist   # 启动
launchctl bootout  gui/$(id -u)/com.local.open-terminal-host                                  # 停止
```

只想暂时禁用某个终端：管理员设置 → Machines 里关掉该连接。改 netguard 规则后要 `docker compose up -d --force-recreate`
（两个容器一起重建：终端共享 netguard 的网络命名空间）。

回滚（全部可逆）：

1. Open WebUI：`docker exec -i open-webui python - rollback < set_terminals.py`（删两个连接、`tool_permissions.enable` 恢复原值、
   删过滤函数、模型去掉 `machineId`）；备份 `/app/backend/data/{config-backup,model-backup}-20261126-terminals.json`、
   `webui.db.20261126-terminals`。
2. 本机终端：`launchctl bootout …` 后删 plist 和 `~/.local/share/open-terminal-host`、`~/.config/open-terminal-host`。
3. 容器终端：`docker compose down -v`（`-v` 连 home 卷一起删；工作目录 `~/docker/data/ai/workspace` 不会被删）。
4. open-webui 网络：`docker network disconnect owui-terminal open-webui`，compose 换回同目录的 `compose.yaml.bak-20261126-terminals`，
   最后 `docker network rm owui-terminal`。

## Open WebUI：ChatGPT 套餐（Sign in with ChatGPT，2026-11-26）

用 OpenAI 官方的 [Sign in with ChatGPT 套餐额度](https://developers.openai.com/siwc/token-sharing-open-source)
（OAuth scope `chatgpt.tokens.use.direct`）在 Open WebUI 里用 ChatGPT 订阅里的模型。这是 OpenAI 给开源、个人自托管工具的公开流程：
首次登录时动态注册 client（`client_id=dynamic_agent_client`），不需要 API key 和审批；推理走公开的 `POST /v1/responses`。

```
Open WebUI（第 2 个 OpenAI 连接，API 类型 Responses，前缀 chatgpt）
  → http://openai-plan:8000/v1（Mac mini 容器 openai-plan，docker 网络 owui-openai）
  → https://api.openai.com/v1/responses（Bearer = 用户的 SIWC access token）
```

- **为什么中间要有代理**：Open WebUI 有 Responses 连接类型，但直连过不去：access token 一小时过期，而连接里的 key 是静态的；
  这条套餐通道要求每个请求都是 `store:false` + `stream:true`，Open WebUI 的标题／标签等后台请求却是非流式的，还会带上不支持的字段
  （`max_output_tokens`、`temperature`、`metadata`、任务参数里的 `chat_template_kwargs` 等）；`/v1/models` 返回的是 `{"models":[{slug,…}]}`
  而不是 `{"data":[{id}]}`；函数工具要放进 namespace。代理只做这几件事：换上新 token（刷新串行，轮换后的 refresh token 原子写入 600 文件），
  强制 `store:false` / `stream:true`（非流式调用方拿到 `response.completed` 里的完整响应），只保留支持的顶层字段，
  `reasoning_effort` → `reasoning.effort`，`reasoning.summary=auto`（界面显示思考摘要），请求加密的 reasoning 以便 `store:false` 下回放，
  去掉回放条目的 id、`system` → `developer`，函数工具收进一个 namespace（`SIWC_TOOL_MODE=plain` 可关掉），改写模型列表，把套餐错误码转成中文提示。
  只接受 `/v1/responses`（连接必须是 Responses 类型）。代码 `clients/openai-plan/siwc.py`（只用标准库，不公开）。
- **可见性**：不建模型记录，所以 `chatgpt.*` 模型只有管理员看得到；不改默认模型（仍是 Flash-Next）、不改任务模型、不改其他连接。
  条款要求套餐只用于本人的请求，所以不要把这些模型共享给其他用户或群组。
- **能力**：流式输出；文字、图片（data URL）、文件输入；原生函数调用（终端、MCP 等 Open WebUI 工具）；推理强度用高级参数里的 `reasoning_effort`。
  不支持：OpenAI 托管工具（生图、文件检索、Code Interpreter、托管 MCP、computer use）、音视频输入、`temperature` / `top_p` / `max_tokens`（代理会丢掉）。
- **额度**：Plus 的 5 小时额度与其他用套餐的应用共用，Pro 只受每周额度限制；可在 [ChatGPT 设置 → Usage](https://chatgpt.com/settings/usage)
  给这个应用设每周上限。超限时对话里显示“ChatGPT 套餐额度已到上限…”（`subscription_sharing_usage_limit_exceeded`，429），不会自动转成按量计费。
- **后台任务**：任务模型为空，等于用当前对话模型（见上表“任务模型”一行），所以 ChatGPT 对话的标题、标签、追问也发给同一个 ChatGPT 模型，
  会占少量套餐额度。如果不想这样，可以在管理员设置 → 界面里关掉追问；或者把外部任务模型设成某个本地模型，但这会影响所有对话
  （11-26 撤掉的就是这种设置：会把 uncensored 换出去）。
- **隔离**：容器官方 `python:slim`（不固定，watchtower 跟随最新 Python）、uid 1000、无 capability、`no-new-privileges`、只读根、不挂 docker.sock、
  不映射端口；`owui-openai` 网络里只有 open-webui 和它。容器终端连不到它（2026-11-26 实测：域名解析不到，IP 也被拒）。
  凭据在 Mac mini `~/docker/config/openai-plan/siwc/credentials.json`（600），用户账号下的进程可读，本机终端（`mac-mini`，每次调用都要确认）也包括在内。

文件（Mac mini；部署副本在 `clients/openai-plan/`，不公开）：

| 项 | 位置 |
| --- | --- |
| compose + 代码 | `~/docker/compose/ai/openai-plan/{compose.yaml,siwc.py}` |
| 代理 key | `~/docker/config/openai-plan/openai-plan.env`（600；Open WebUI 连接里存同一个） |
| SIWC 凭据 | `~/docker/config/openai-plan/siwc/`（700）：`host.json`（本机 `ext_agent_host_id`）、`credentials.json`（600）、`pending.json`（进行中的登录） |
| open-webui compose | 多加网络 `owui-openai`（备份 `compose.yaml.bak-20261126-openai-plan`） |
| Open WebUI 设置 / 验收 | `clients/open-webui/set_openai_plan.py`（show / backup / apply / rollback）、`test_openai_plan.py`（models / chat / image / tool / title） |

登录（一次；refresh token 30 天，每次刷新都会续期，代理空闲时每周刷新一次保持会话）：

```sh
# 以下都在 Mac mini 上运行（OrbStack 的 docker 在 ~/.orbstack/bin）
# 交互式（需要终端）：
docker exec -it openai-plan python /app/siwc.py login
# 两步式（不需要交互终端，可以由 agent 代跑；链接 24 小时有效）：
docker exec openai-plan python /app/siwc.py login-start          # 打印链接
#   浏览器打开链接 → 登录 ChatGPT → 同意“使用 ChatGPT 套餐” → 页面跳到 http://127.0.0.1:1455/auth/callback?…（打不开，正常）
docker exec -i openai-plan python /app/siwc.py login-finish <<< '<复制的整串地址>'
docker exec openai-plan python /app/siwc.py status               # 不打印任何 token
```

回调地址固定是 `127.0.0.1`（OpenAI 的要求），在哪台设备的浏览器上登录都一样：页面打不开，复制地址栏里的地址即可。
地址里的授权码只能用一次、几分钟内过期，而且离开 Mac mini 上保存的 PKCE verifier 就用不了。

登录后验收（open-webui 容器里，模拟界面对话）：`for p in models chat image tool title; do docker exec -i open-webui python - $p < test_openai_plan.py; done`。
如果 `tool` 报 `subscription_sharing_unsupported_capability`（tools），把 compose 里的 `SIWC_TOOL_MODE` 改成 `plain` 再 `docker compose up -d` 重测。
多轮回放：`docker exec -i -e EFFORT1=medium open-webui python - multi < test_openai_plan_multi.py`（两轮界面对话，第二轮由 Open WebUI
从数据库历史回放第一轮的 reasoning／函数调用）；`test_openai_plan_replay.py` 直接对代理发两次请求，检查同样的回放。

2026-11-26 验收结果（gpt-6.2-sol，effort low，namespace 模式，代理未改动）：

| 项 | 结果 |
| --- | --- |
| models | 10 个 `chatgpt.*`，显示名 `chatgpt.GPT-6.2-Sol` 等（前缀 + 套餐的 display_name） |
| chat | 流式 44 个事件，长度各不相同，约 4 s |
| image | 回答“红色”，约 3 s |
| tool | 容器终端 `run_command` 真实执行，namespace 模式可用，约 5 s |
| title | 非流式后台任务由代理聚合，约 2 s |
| 多轮回放 | 第一轮（medium）产生带 `encrypted_content` 的 reasoning 和函数调用；第二轮回放后上游接受并再次调用工具 |
| 可见性 | 没有模型记录，`BYPASS_MODEL_ACCESS_CONTROL` 未开，所以只有管理员能看到 |

流式说明：Responses 类型的连接在 socket.io 上逐条转发原生 `response:completion` 事件（`response.output_text.delta`），
`chat:completion` 只带开始和结束两次快照，所以检查流式要数前者（`test_openai_plan.py` 已改）。
effort low 时简单问题的 reasoning tokens 是 0，不产生 reasoning 条目。

登出：`docker exec openai-plan python /app/siwc.py logout`（吊销 refresh token，保留 client_id 和 host id）；也可以在 ChatGPT 设置里断开这个应用。

回滚：

1. Open WebUI：`docker exec -i open-webui python - rollback < set_openai_plan.py`（删掉第 2 个连接；全量配置备份
   `/app/backend/data/config-backup-20261126-openai-plan.json`）。
2. 先登出，再 `cd ~/docker/compose/ai/openai-plan && docker compose down`，删除 `~/docker/config/openai-plan`。
3. 网络：`docker network disconnect owui-openai open-webui`，open-webui compose 换回 `compose.yaml.bak-20261126-openai-plan`，
   最后 `docker network rm owui-openai`。

## Qwen-Image ControlNet（2026-11-24）

三个 qwen-image 入口（`qwen-image-2.1-turbo`、`-turbo-pe`、`qwen-image-2.1`）接受两个可选字段，不带时输出与以前逐字节相同，客户端的默认设置都不变：

| 字段 | 说明 |
| --- | --- |
| `control_image` | **预处理好的**控制图（服务端不做预处理）：canny / depth / pose / HED / lineart / MLSD / scribble / grayscale。`/v1/images/generations` 里给 base64 或 data URL；`/v1/images/edits` 里是 multipart 文件，这时对 `image` 在 `mask` 透明处按控制图局部重绘（恰好 1 张图 + mask） |
| `control_strength` | 0–2，默认 1.0 |

不给 `size` 时画布按控制图比例，所以控制图最好就按想要的输出比例给（最大 2400×1792 面积）。不能和 `auto_mask` 同用。
Turbo 1 MP 约 10 s（无控制约 7 s），2048² 约 55 s。细节和数字：`reports/qwen-image.md` 8d 节。

**预处理在客户端做**（例子用 Python；mflux 在部署环境 `.venv` 里，客户端机器上要另装 `pip install mflux opencv-python`）：

| 控制类型 | 做法 |
| --- | --- |
| canny | `cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)`（OpenCV） |
| depth | `mflux-save-depth --image-path photo.jpg`（Apple DepthPro，写出 `photo_depth.png`；第一次运行下载权重） |
| HED（软边缘） | `from mflux.models.hed.hed import HED; HED().edge_map(pil_image)`（MLX 推理，但加载 `lllyasviel/Annotators` 的 `.pth` 需要 **torch**） |
| OpenPose（人体姿态） | `from mflux.models.openpose.openpose import OpenPoseBody; OpenPoseBody().pose_map(pil_image)`（同上，需要 torch） |
| grayscale / scribble / lineart | 灰度图、手绘线稿直接当控制图 |

```sh
img=$(base64 -i edges.png)
curl -s http://127.0.0.1:8080/v1/images/generations -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen-image-2.1-turbo\",\"prompt\":\"a red brick house, photo\",\"control_image\":\"data:image/png;base64,$img\",\"control_strength\":1.0}"
curl -s http://127.0.0.1:8080/v1/images/edits -H "Authorization: Bearer $KEY" -F model=qwen-image-2.1-turbo \
  -F prompt='a person waving' -F image=@photo.png -F mask=@mask.png -F control_image=@pose.png
```

**哪些客户端能用**：Open WebUI、LobeChat、OpenClaw 的出图界面都只发 prompt / size（编辑时发图），**传不了 `control_image`**，在它们里面照常出图，不受影响。
要用 ControlNet，用上面的脚本 / curl 或自己的 OpenAI SDK 调用（`extra_body={"control_image": ...}`）。MCP `generate_image` / `edit_image` 目前也不转发这个字段（要用需在 `clients/mcp/server.py` 加参数，未做）。

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

**在 Open WebUI 里显示媒体**（0.13.5 实测/核对源码，2026-11-27；工具返回文案和 MCP `instructions` 都按此写）：
`HTMLToken.svelte` 只把**标签之间的文字**当作地址（`src=` 属性被忽略），而且只有整段 HTML 作为一个块传给它时才出播放器；
`audio`／`video` 不在 CommonMark 的块级 HTML 列表里，单独一行的 `<video>URL</video>` 会被拆成三段、原样显示（上游
issue open-webui#34074）。所以：

- 图片：markdown `![image](URL)`。
- `speak` 的语音：MP3 存进对话文件库（`/api/v1/files/<id>/content`），模型调用 Open WebUI 内置工具 `display_file`
  显示可播放的文件卡片（模型开着内置工具时才有这个工具）；没有时写一行 `<div><audio>URL</audio></div>`。
  MCP 返回里的音频内容 Open WebUI 只接受 WAV，MP3 会被丢掉，不能靠它出播放器。
- 视频（`/media/<id>.mp4`，不在对话文件库里，`display_file` 打不开）：写一行 `<div><video>URL</video></div>`。
- 这类行要单独成段（前后空一行，不放在列表里）；两行紧挨着时只有第一行生效。不要写 `<audio controls src=…>`。

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

- **OpenClaw 出图的自动回退（2026-11-23 已修）**：OpenClaw 2026.11.6 的出图在候选都失败后，默认会自动追加所有已登录的出图提供方（代码里 `autoProviderFallback` 默认为 true；文档说显式配置就不追加，与代码不符）。对话里的 `image_generate` 只要 `mediaModels.image` 写了 primary/fallbacks，就会关掉追加；但 CLI `openclaw infer image generate` 不带 `--model` 时一直会追加，也没有配置项能关掉（除非停用 `openai` 插件或删掉它的登录）。所以 CLI 一律带 `--model`：按次指定的模型只试这一个。另外，对话模型如果是用户自己的 OpenAI（Codex 运行时），这个模型可以自己调用 Codex 内置的出图。那是用户选定的模型在用它自己的工具，不属于回退，不在这里管。
- **OpenClaw 访问内网**：OpenClaw 默认拦截内网地址（`Blocked hostname or private/internal/special-use IP address`）。出图走 `litellm` 提供方，需要它的 `request.allowPrivateNetwork: true`；对话（`vllm` 提供方）和记忆检索会自动信任自己配置的 `baseUrl`，不用另设。
- **llama-swap playground 的出图尺寸**：尺寸下拉框的选择存在浏览器 localStorage（`playground-image-size`），每次请求都带上；“总是 1792x1024” 是这个浏览器记住了上次的选择，在下拉框里改选即可。没有“跟随模型默认”的选项或配置项，不改。
