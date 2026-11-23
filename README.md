# llm-serving

面向 Apple Silicon 的本地模型接口与部署工具：文本、嵌入、重排、语音识别、图片和视频，全部在一台 Mac 上运行（参数来自 **M5 Ultra、96 GiB 统一内存**）。主要文本后端是 **Strata Metal**。

> **速度、内存、磁盘和每项修改的收益：见 [性能与资源报告](docs/performance.md)。**

```text
客户端 → Caddy :8080 → llama-swap 127.0.0.1:8081 → 按需加载模型
                 └→ 只读媒体服务 127.0.0.1:8090
```

模型服务独立运行；不依赖另一台电脑、外部推理 API、聊天前端、MCP 或公网反向代理。

## 模型

| 类别 | 模型 ID | 后端 | 说明 |
| --- | --- | --- | --- |
| 文本／视觉 | `qwen3.8-flash-next`、`-uncensored` | [Strata fork](https://github.com/ain-soph/Strata/tree/serving) | 262k 上下文 × 1 路，MTP 投机解码 |
| 嵌入 | `embeddinggemma-2`、`-multimodal` | 文本 MLX，媒体 llama.cpp | 文本常驻；多模态支持图片、音频、视频 |
| 嵌入 | `qwen3-embedding-0.6b`／`4b`／`8b`、`voyage-4-nano` | MLX，自写服务 | 32K 输入；零拷贝；8B 为按层类 W8A8 混合副本，其余 16 位 |
| 重排 | `qwen3-reranker-0.6b`／`4b`／`8b` | MLX，自写服务 | `/v1/rerank`，按长度排序批量计算；按层类 W8A8 混合副本，零拷贝；RAG 用 `4b` |
| 决策 | `pplx-decider-flash` | MLX 移植 | SystemOne `/v1/systemone`：判断／选择／打分概率；文本与图片同一入口，按层类 W8A8，预量化存盘 |
| 决策 | `elephant-1` | MLX 移植 + 官方 OneJev | 9B，文本，最长 16K token；按层类 W8A8 / 8-bit，预量化存盘，与聊天模型共存 |
| 决策 | `mammoth-1` | 同上 | 27B，同样按层类量化；加载时卸下聊天模型 |
| 语音转写 | `qwen3-asr-1.7b`、`-int8` | [mlx-qwen3-asr fork](https://github.com/ain-soph/mlx-qwen3-asr/tree/serving) | 默认；中文、中英混合；强制对齐器给词级时间戳；`-int8` 解码器 W8A16，快 1.27–1.31× |
| 语音转写 | `whisper-large-v3` | [mlx-whisper fork](https://github.com/ain-soph/mlx-examples/tree/serving/whisper) | 多语种 |
| 语音合成 | `qwen3-tts`、`-clone`、`-design` | mlx-audio fork | `/v1/audio/speech`：预设音色（含四川话、北京话）、声音克隆、声音设计；边生成边输出 |
| 图片生成／编辑 | `minimax-h3-image` | [h3-image-mlx fork](https://github.com/ain-soph/h3-image-mlx/tree/serving) | W8A8，零拷贝，融合内核；各尺寸与聊天模型共存 |
| 图片生成／编辑 | `qwen-image-2.1-turbo`、`qwen-image-2.1-turbo-pe`、`qwen-image-2.1` | [mflux fork](https://github.com/ain-soph/mflux/tree/serving) | W8A8 DiT + W8A16 文本编码器；8／40 步；最多 10 张参考图、遮罩、透明 PNG、提示词增强；非商用许可 |
| 图片生成 | `z-image-turbo` | [mflux fork](https://github.com/ain-soph/mflux/tree/serving) | W8A8 DiT + BF16 文本编码器；9 步文生图；与聊天模型共存；Apache-2.0 |
| 视频＋音频 | `minimax-h3-video` | [h3.c fork](https://github.com/ain-soph/h3.c/tree/serving) | `best`／`standard`／`turbo`／`fast` 四档（`turbo` 需另备 22 GB 权重，限 8 s），最长 15 s；INT8 DiT 零拷贝、按 block 释放；文本编码器与 H3 图片共用 INT8 文件 |

MLX 本身用 [ain-soph/mlx fork](https://github.com/ain-soph/mlx/tree/serving) 的 wheel（零拷贝加载、页对齐保存、`MLX_CACHE_LIMIT_MB`）。完整 commit 固定在 `native.lock.json`（原生引擎）和 `uv.lock`（Python 包）；[原生构建脚本](tools/native.py)、[模型路由模板](platform/llama-swap/conf.d/)、[Strata 配置模板](services/text/strata/configs/) 和自写接口都在本仓库。权重准备与服务说明：[Strata](docs/strata.md)、[重排](docs/rerank.md)、[零拷贝与 W8A8 副本](docs/w8a8.md)、[决策模型](docs/decisions.md)、[SOV 语音转写](docs/sov.md)、[语音合成](docs/tts.md)、[H3 图片](docs/h3-image.md)、[Qwen-Image](docs/qwen-image.md)、[Z-Image](docs/z-image.md)、[H3 视频](docs/h3-video.md)。

## 上游贡献

我们在 fork 里做的改动，能独立成立的已向上游提交（2026-11-17 均为开放状态，所有 PR 的 bot review 意见都已处理）：

| 上游 | PR | 内容 |
| --- | --- | --- |
| antirez/h3.c | [#98](https://github.com/antirez/h3.c/pull/98) | 修两处 threadgroup 归约竞争（因果 GQA softmax、VAE 编码器 group norm），条件渲染可复现 |
| antirez/h3.c | [#99](https://github.com/antirez/h3.c/pull/99) | 可选的离线 INT8 DiT 打包（`h3_pack_int8`）与页对齐零拷贝加载 |
| Anionex/h3-image-mlx | [#20](https://github.com/Anionex/h3-image-mlx/pull/20) | 零拷贝权重（`align`、`H3_ZERO_COPY`）与 DiT 缓存预算 |
| Anionex/h3-image-mlx | [#21](https://github.com/Anionex/h3-image-mlx/pull/21) | W8A8 GEMM 64×128 分块与 DiT 逐元素融合（输出相同） |
| mflux-community/mflux | [#998](https://github.com/mflux-community/mflux/pull/998) | Qwen-Image-2.1 文生图也支持 `--enhance-prompt` |
| mflux-community/mflux | [#999](https://github.com/mflux-community/mflux/pull/999) | Qwen-Image-2.1 W8A8 DiT（ComfyUI int8_convrot，M5 TensorOps） |
| mflux-community/mflux | [#1000](https://github.com/mflux-community/mflux/pull/1000) | Qwen-Image-2.1 VAE 分条解码（1536² VAE 峰值 35.6 → 8.8 GB） |
| ml-explore/mlx-examples | [#1464](https://github.com/ml-explore/mlx-examples/pull/1464) | Whisper 融合注意力，词级对齐按需导入 |
| moona3k/mlx-qwen3-asr | [#32](https://github.com/moona3k/mlx-qwen3-asr/pull/32) | 贪心解码先排下一步再读当前 token |
| Neywa-Labs/mlx-audio | [#1206](https://github.com/Neywa-Labs/mlx-audio/pull/1206) | SOV-Transcribe-1 解码同样的流水线 |
| Niko1221/Strata | issue [#6182](https://github.com/Niko1221/Strata/issues/6182) + draft PR [#6183](https://github.com/Niko1221/Strata/pull/6183) | Metal `--prefill auto` 让主机行与 GPU 重叠 |

## 目录结构

仓库就是部署：clone 到 `~/llm-serving`，线上运行的脚本、服务代码和 llama-swap／网关配置就是仓库里的文件。
同一棵树里还有本机私有的东西（权重链接、venv、日志、密钥、fork 子仓库、实验），它们被 `.gitignore` 默认忽略：
只有 [`public-paths.txt`](public-paths.txt) 列出的路径公开（`tools/gen_gitignore.py` 由它生成 `.gitignore`）。

| 目录 | 内容 |
| --- | --- |
| `bin/` | gpu-lock（显存预约、排队、托管运行）、gpu-top、gpu-pool、memguard、lsw-apply、archive-restore／model-archive、regress、health-check |
| `platform/llama-swap/` | `config.yaml`、`conf.d/*.yaml`（本机路径写成 `${env.LLM_SERVING}` 等）、`env.sh`、`serve.sh` |
| `platform/gateway/` | Caddyfile（`{$LLM_SERVING}`、`{$LLM_TRUSTED_PROXY}`） |
| `platform/launchd/` | 各 LaunchAgent 模板（`*.plist.in`，渲染到 `~/Library/LaunchAgents`） |
| `platform/agent-bridge/` | llm-collab MCP 启动器与配置、多 agent 规则、本地 worker |
| `services/<类别>/` | 各模型的接口服务：text/strata（引擎配置模板）、embedding、rerank、decisions、asr、tts、image、video |
| `shared/common/` | MLX W8A8／8-bit 公共代码 |
| `standby/` | 已下线但保留的代码（`standby/asr/sov/`） |
| `clients/` | MCP 服务（`clients/mcp/`）与测试脚本 |
| `docs/`、`tools/`、`tests/` | 文档、安装与检查工具、单元测试 |

本机私有、不进库：`.venv/`、`.local/`（本机取值、`serving.env`、原生构建）、`.backup/`、`.locks/`、`logs/`、`data/`、`evals/`、
`reports/`、`upstream/`、`archive/`、`platform/llama-swap/.api_key` 与活动数据库、`platform/agent-bridge/{state,runtime,workspace,release}/`、
各服务目录里的 fork 子仓库、实验、结果和输出。

**在 `~/llm-serving` 里不要运行** `git clean`、`git reset --hard`、`git checkout .`／`git restore .`、`git stash -u/-a`：
被忽略的线上数据没有别的保护。改动只提交自己的路径（`git commit -o -- <paths>`），未上线的开发放在 `git worktree` 里。

## 安装与迁移

需要 Apple Silicon、macOS 及适用的 Apple Command Line Tools。先安装 [Homebrew](https://brew.sh)，再执行：

```sh
git clone https://github.com/ain-soph/llm-serving.git ~/llm-serving
cd ~/llm-serving
./tools/bootstrap.sh
.venv/bin/python tools/native.py all
```

`bootstrap.sh` 安装 Homebrew 依赖和 `.venv`，启用隐私检查的 git 钩子，把多 agent 分工规则装进 `~/.claude/CLAUDE.md`
（标记块合并，不覆盖原有内容），最后运行 `tools/install_serving.py`，见[多 agent 分工与恢复](docs/multi-agent.md)。

`Brewfile` 统一安装 **Homebrew Python 3.15**、uv、Caddy、llama-swap、llama.cpp、FFmpeg、CMake、Ninja。uv 使用 Homebrew 的 Python 3.15 创建项目唯一的 `.venv`：**线上只有这一个环境**，所有 Python 服务使用同一个解释器；开发新模型时用它自己的 `.venv-dev`，上线时并进来。`pyproject.toml` 只写版本下限，`uv.lock` 精确锁定全部依赖，包括各 fork 的完整 commit。
在 `~/llm-serving` 下没有自己 `pyproject.toml` 的子目录里运行 `uv run`／`uv sync` 会同步这个线上环境；实验目录用自己的项目或 `uv run --no-project`。

原生引擎从 `native.lock.json` 的完整 commit 构建到 `.local/src/`。编译后的可执行文件、动态库、Metal shader 和构建信息留在同一个本地构建树，不能只搬一个可执行文件。

本机取值写在 `.local/serving-values.json`（不进库；缺省时：权重在 `~/models`，Strata 用 `native.py` 的构建，网关前没有受信代理）：

| 键 | 含义 |
| --- | --- |
| `MODELS` | 模型权重目录 |
| `STRATA_RELEASE`、`STRATA_EXE`、`STRATA_VISION_EXE` | 部署的 Strata 构建树与两个可执行文件 |
| `TRUSTED_PROXY` | 网关前唯一受信的反向代理 IPv4（它的 `X-Real-IP` 成为客户端地址） |
| `SHARE_BASE`、`FETCH_HOSTS` | `/media` 链接的公开地址；视频渲染可下载输入文件的主机（逗号分隔） |

```sh
.venv/bin/python tools/install_serving.py          # .local/serving.env、Strata 引擎 JSON、LaunchAgents、API key（只建缺失的）
.venv/bin/python tools/doctor.py                   # 不加载模型：llama-swap -validate、caddy validate、可执行文件、权重
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.llama-server.plist   # 其余 LaunchAgent 同理
```

`install_serving.py` 不启动、不重启任何东西，只打印加载命令；`--check` 只比较，`--update` 先备份到 `.backup/` 再替换。
llama-swap 的配置不渲染：`platform/llama-swap/env.sh` 读取 `.local/serving.env`，缺任一变量时 llama-swap 拒绝启动并给出变量名。
网关监听 `0.0.0.0:8080`，llama-swap 只监听 `127.0.0.1:8081`。

GPU 池由 `bin/gpu-pool` 按当前 `iogpu.wired_limit_mb` 计算（减去系统与常驻嵌入的预留）；当前完整聊天配置要求池至少 **65200 MiB**。
工具不会因为识别出一台 Mac 就提高内存上限，也不会把这台机器的上下文／并发参数缩放成未经验证的其他机型参数。

## 模型与状态

- 模型目录完全独立，必要布局见 `models.required.json`。转换过的 Strata pack、INT8 视频权重（[准备步骤](docs/h3-video.md)）和页对齐图片权重（Qwen-Image 见[准备步骤](docs/qwen-image.md)）应一并迁移；原始 Hub 权重不一定能直接替代这些文件。
  归档到另一台机器的模型由 `bin/archive-restore` 在第一次请求时按清单拷回并校验（`<MODELS>/<名>.archive`）。
- 运行数据在仓库目录里，全部被忽略：API key `platform/llama-swap/.api_key`（0600）、活动数据库、`logs/`、图片 `data/media/images`、视频 `services/video/h3-video/outputs`。模型权重和运行数据都不提交到 Git。
- 迁移时可以保留旧 API key，放到 `platform/llama-swap/.api_key`，权限设为 `0600`；否则 `install_serving.py` 会生成一个新凭证，供客户端使用。不要将密钥放到命令行参数或公开配置中。
- 有权重文件后，推理不需要外部 API key 或下载 token。个别模型首次下载可能需要模型托管站点的授权；这属于准备阶段。

## 验证范围

已在开发机器使用 Python **3.15.2** 完成统一依赖安装与导入，以及 Numba JIT、PyTorch/SciPy CPU 和日语对齐 tokenizer 基础调用检查；同时完成 llama-swap/Caddy 配置校验、FFmpeg 编码器检查，以及安装路径和隐私扫描测试。重排的 Python 3.15 真实模型/API 检查见[专项说明](docs/rerank.md)。参照部署 2026-11-18 起所有 Python 入口都运行在这个统一环境里（同一份 `uv.lock`），切换前后 `bin/regress` 全部用例一致；入库的 llama-swap 配置展开后与线上文件逐字节相同。

尚未在干净系统完成全套原生重编译和端到端推理恢复演练。本仓库提供可审查的迁移工具和当前配置，首次切换前仍需运行真实 API 冒烟检查。`doctor.py` 不加载模型，不替代正确性/性能验证。

更多说明：[多 agent 分工与恢复](docs/multi-agent.md)、[设计与硬件优化](docs/design.md)、[运维：回归测试与健康检查](docs/operations.md)、[客户端接入与 MCP](docs/clients.md)、[发布与隐私](docs/privacy.md)。
