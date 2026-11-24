# llm-serving

面向 Apple Silicon 的本地模型接口与部署工具：文本、嵌入、重排、决策、语音识别与合成、图片和视频，全部在一台 Mac 上运行（参数来自 **M5 Ultra、96 GiB 统一内存**）。主要文本后端是 **Strata Metal**。

每个模型都按同一流程部署：先对照官方实现确认正确，再测各阶段的速度、内存和磁盘，找瓶颈，然后优化。下面的[通用优化](#通用优化)和[各模型的优化](#各模型的优化)列出做了什么、带来多少收益；完整数字、测试条件和方法见 **[性能与资源报告](docs/performance.md)**。

```text
客户端 → Caddy :8080 → llama-swap 127.0.0.1:8081 → 按需加载模型
                 └→ 只读媒体服务 127.0.0.1:8090
```

模型服务独立运行；不依赖另一台电脑、外部推理 API、聊天前端、MCP 或公网反向代理。

## 模型

| 类别 | 模型 ID | 后端 | 说明 |
| --- | --- | --- | --- |
| 文本／视觉 | `qwen3.8-flash-next`、`-uncensored`¹ | [Strata fork](https://github.com/ain-soph/Strata/tree/serving) | 262k 上下文 × 2 路，MTP 投机解码，图片输入 |
| 嵌入 | `embeddinggemma-2`、`-multimodal` | 文本 MLX，媒体 llama.cpp | 文本常驻；多模态支持图片、音频、视频 |
| 嵌入 | `qwen3-embedding-0.6b`／`4b`／`8b`、`voyage-4-nano` | MLX，自写服务 | 32K 输入 |
| 重排 | `qwen3-reranker-0.6b`／`4b`／`8b` | MLX，自写服务 | `/v1/rerank`；RAG 用 `4b` |
| 决策 | `pplx-decider-flash`¹ | MLX 移植 | 4B；文本与图片，最长 32K token |
| 决策 | `elephant-1`¹、`mammoth-1`¹ | MLX 移植 + 官方 OneJev | 9B／27B，文本，最长 16K token；`mammoth-1` 加载时卸下聊天模型 |
| 决策 | `joey-1` | MLX 移植 + 官方 OneJev | 2B，文本，最长 16K token（约 0.95 s）；与聊天模型共存 |
| 决策 | `drex-v1.5` | nace-ai llama.cpp 分支（未改） | 9B，文本，最长 16K token |
| 语音转写 | `qwen3-asr-1.7b` | [mlx-audio fork](https://github.com/ain-soph/mlx-audio/tree/serving) + 自写引擎 | 默认；中文、中英混合、长音频；词级时间戳 |
| 语音转写 | `whisper-large-v3` | [mlx-whisper fork](https://github.com/ain-soph/mlx-examples/tree/serving/whisper) | 多语种 |
| 语音合成 | `qwen3-tts`、`-clone`、`-design` | [mlx-audio fork](https://github.com/ain-soph/mlx-audio/tree/serving) | 预设音色（含四川话、北京话）、声音克隆、声音设计；边生成边输出 |
| 语音合成 | `fish-drama-3` | [mlx-audio fork](https://github.com/ain-soph/mlx-audio/tree/serving) | Fish Audio Drama 3（35B MoE，8 位）：文本内语气提示、多说话人、克隆；约 2.6 倍实时；独占入口；非商用许可 |
| 图片生成／编辑 | `minimax-h3-image` | [h3-image-mlx fork](https://github.com/ain-soph/h3-image-mlx/tree/serving) | 单边最大 4096，超出返回 400；与聊天模型共存 |
| 图片生成／编辑 | `minimax-h3-image-pe` | 同上 + mlx-vlm 跑官方 [MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) | 可选：先用官方提示词扩写，返回 `revised_prompt`；扩写时卸下聊天模型² |
| 图片生成／编辑 | `qwen-image-2.1-turbo`、`-turbo-pe`、`qwen-image-2.1` | [mflux fork](https://github.com/ain-soph/mflux/tree/serving) | 8／40 步；最多 10 张参考图、遮罩、透明 PNG、提示词增强；非商用许可 |
| 图片生成 | `z-image-turbo` | [mflux fork](https://github.com/ain-soph/mflux/tree/serving) | 9 步文生图；与聊天模型共存；Apache-2.0 |
| 视频＋音频 | `minimax-h3-video` | [h3.c fork](https://github.com/ain-soph/h3.c/tree/serving) | `best`／`standard`／`turbo`／`fast` 四档，最长 15 s（`turbo` 限 8 s）；首尾帧、参考视频续写；渲染时独占 GPU |

¹ 权重归档在另一台机器，第一次请求时由 `bin/archive-restore` 拷回并校验，之后与其他模型相同。
² `/v1/models` 里的 `minimax-h3-image-pe-expander` 是它内部用的扩写模型，客户端不直接调用。

## 通用优化

跨模型共用的做法；各模型条目只写本模型特有的部分。

- **零拷贝、页对齐加载**：权重存成页对齐的标准 safetensors，加载时直接映射页缓存，不再复制一份。已并入 [MLX fork](https://github.com/ain-soph/mlx/tree/serving)（`mx.load(..., zero_copy=True)`）。例：重排 4B 加载后 footprint 8.1 → 0.4 GB；H3 视频 DiT 加载 2.29 → 0.33 s。
- **按层类选精度**：每个模型按投影类别（q/k/v、o、gate/up、down……）分别测速度和误差，误差以官方 FP32 为参照。W8A8（INT8 权重和激活，M5 TensorOps）只用在测得提速的类，其余用 W8A16、8 位权重或 BF16／FP16；选好后预量化存盘。例：Decider 的 down_proj 回 FP16，误差 0.0245 → 0.0009，速度只少 2–5%。[原则与逐类结果](docs/w8a8.md)
- **一份 W8A8 内核**：ConvRot 旋转量化和 INT8 GEMM 并入 MLX fork，mflux、h3-image-mlx 和本仓库原来的三份副本删除，输出逐字节相同。
- **按层／按 block 流式**：大模型只在计算某层时占 GPU 内存。例：Qwen-Image 生成期 wired 44–52 → 5.6–8.9 GB；H3 视频 768p 21.9 → 14.3 GB。
- **容量路由**：每个入口在 llama-swap 声明实测 wired 峰值加余量（`capacities.gpu`），放不下才换出。聊天模型预约 69,500 MiB，多数入口能与它同驻。agent 的测试用 `bin/gpu-lock` 预约内存，测速时独占。
- **归档与按需拷回**：少用的权重归档到另一台机器，`bin/archive-restore` 在第一次请求时按清单拷回并校验。
- **回归检查**：`bin/regress` 对每个入口比对金标准输出，每次升级或改配置后运行。

## 各模型的优化

收益是同一台机器上的前后对比；“逐字节相同”指输出不变。

**文本：Qwen3.8 Flash-Next（Strata）** · [说明](docs/strata.md)
- 从 llama.cpp 换到 Strata，同一份 AD-4.27 权重：单路 decode 59.9 → 123.2 tok/s（2.06×），长提示 prefill 1.72×。
- 修 Metal `--prefill auto` 的分块：prefill +8.6%／+11.3%（26K／116K token），不再需要手动参数。
- 省内存双槽：262k × 2 路的 wired 峰值 70,962 → 67,695 MiB。
- 对话缓存用上游默认（按空闲内存定，现网 8192 MiB）并溢出到 SSD：108K token 会话续问 40.4 → 0.85 s。
- 现网：单路 decode 约 152 tok/s，prefill 约 2,700 tok/s（26K–116K token）。

**嵌入**
- EmbeddingGemma 2 文本入口换 MLX（FP32 激活）：4 × 8000 token 35,300 tok/s（原 llama.cpp 19,000）；对 FP32 参照 SciFact top-10 重合 2997/3000。
- Qwen3-Embedding、voyage：自写 MLX 服务替代 llama-server，1.5–2.2×；8B 按层类 W8A8，1300 篇 62.6 → 45.6 s；4B 零拷贝，加载后 footprint 7.9 → 0.2 GB。
- voyage 改回 BF16：SciFact 上 NaN 向量 327 篇 → 0。

**重排：Qwen3-Reranker** · [说明](docs/rerank.md)
- MLX 批量服务，只算 yes／no 输出头：比原 llama.cpp 快 1.42–2.77×。
- 按长度排序成批：4B 快 11.5%；零拷贝：加载后 footprint 8.2 → 0.3 GB。
- 按层类 W8A8：4B 60 × 20 篇 63.5 → 47.1 s，8B 132.8 → 81.7 s。

**决策** · [说明](docs/decisions.md)
- `pplx-decider-flash`：按层类 W8A8 预量化存盘、MLX 缓存 4096 MB：32.6K token 12.70 → 5.64 s，加载 1.2 → 0.5 s，对官方 FP32 最大概率差 0.0009。
- `elephant-1`、`mammoth-1`：只换模型前向，其余用官方代码，比官方在 Mac 上（MPS BF16）快 5–10 倍；按层类混合比全 8 位快，16K token elephant 4.03 → 2.82 s、mammoth 13.5 → 8.5 s；预量化存盘后加载 2.2／6.2 → 0.4 s。
- `joey-1`（2B）：按层类 W8A8（`in_proj_z`、`out_proj` 8 位只量化权重），对官方 FP32 最大概率差 0.0065（全 W8A8 0.0113），16K token 1.112 → 0.947 s。
- `drex-v1.5`：`-ub 512`，wired 峰值 12.5 → 11.0 GiB，放得进聊天模型旁边（长请求 4.0 → 4.3 s）。

**语音转写**
- Qwen3-ASR：8 位存盘（W8A16）快 1.27–1.31×；换 mlx-audio 加自写快速路径和长音频批处理，短集再快 4.5–7.6%，12–16 分钟音频 2.3–2.9×；错误率不变。
- Whisper：融合注意力 1.26–1.31×；8 位存盘，wired 4545 → 3198 MiB（容量 6000 → 4500）；按需导入词级对齐，导入 0.53 → 0.13 s（[说明](docs/whisper-startup.md)）。

**语音合成：Qwen3-TTS、Fish Audio Drama 3** · [说明](docs/tts.md)
- 8 位存盘（W8A16）：实时率低 19–23%，MLX 峰值少 1.5–1.8 GB，ASR 回写错误仍为 0%。首段音频 0.08–0.14 s。
- Drama 3：零拷贝加载 2 s；实时率 0.37，首段 0.17 s；官方只能在 CUDA 上跑，用 ASR 回写验证（16 句 0%）。

**图片：MiniMax H3 Image** · [说明](docs/h3-image.md)
- 零拷贝：1024² 13.1 → 4.2 s。
- DiT 内核（快速 Hadamard 旋转量化、GEMM 分块、逐元素融合）：4096² DiT 108.7 → 56.4 s。
- 提示词与参考图缓存：重复提示词 1024² 3.17 → 2.62 s，同图编辑 8.9 → 6.3 s，逐字节相同。
- 超长序列 DiT 切片（> 35,000 行）：DiT footprint 10.6 → 8.3 GB，放进 10 GB 容量。
- `-pe` 官方提示词扩写：Bench 每图 +7.5～+8.8 分，每请求多 14–34 s。

**图片：Qwen-Image-2.1** · [说明](docs/qwen-image.md)
- 零拷贝：进程 RSS 33.8 → 2.0 GB；逐层流式 + 条带 VAE：wired 44–52 → 5.6–8.9 GB。
- W8A8 DiT：去噪 Turbo 1.49×、base 1.69×；逐元素融合每 block 再快 11–12%。
- W8A16 文本编码器：文件 16.4 → 10.9 GB。首步速度场误差 3.64%（官方自己 BF16 → FP16 为 10.9%）。
- `-pe` 聊天模型改写：Qwen-Image-Bench 总分 54.79 → 58.91。

**图片：Z-Image-Turbo** · [说明](docs/z-image.md)
- W8A8 DiT（从官方 FP32 量化）：1024² 6.97 → 5.31 s，DiT 内存 12.3 → 6.3 GB。
- 文本编码器与 DiT 分别流式：wired 峰值 15.9 → 8.8 GiB，逐字节相同。

**视频：MiniMax H3 Video** · [说明](docs/h3-video.md)
- INT8 DiT 预打包：每份 66.3 → 23.9 GB；零拷贝常驻，按 block 释放：768p wired 21.9 → 14.3 GB。
- Hadamard 旋转 W8A8 DiT（含 `turbo`）：对官方 BF16 的速度场误差 4.05e-2 → 2.96e-2。
- VAE：BF16 计算 2.35×；线性层分块，解码 33.6 → 25.0 s；融合与权重预读再快 13.7%；首尾帧编码每张 5.01 → 1.25 s。后三项逐字节相同。
- 文本编码器 W8A16，与 H3 图片共用 INT8 文件：每次渲染读权重 49 → 24 GB。
- 修两处 threadgroup 竞争，条件渲染可复现。
- 768p 5 s 片段：`best` 约 690 s，`standard` 约 290 s，`turbo` 约 200 s，`fast` 约 160 s。

## 上游贡献

fork 里能独立成立的改动已向上游提交。状态为 2026-11-23 查询结果。

| 上游 | PR／issue | 内容 | 状态 |
| --- | --- | --- | --- |
| antirez/h3.c | [#98](https://github.com/antirez/h3.c/pull/98) | threadgroup 归约竞争 | 开放（搁置） |
| antirez/h3.c | [#99](https://github.com/antirez/h3.c/pull/99) | INT8 DiT 打包、零拷贝 | 开放（搁置） |
| Anionex/h3-image-mlx | [#20](https://github.com/Anionex/h3-image-mlx/pull/20) | 零拷贝、DiT 缓存预算 | 已合并 |
| Anionex/h3-image-mlx | [#21](https://github.com/Anionex/h3-image-mlx/pull/21) | W8A8 GEMM 分块、逐元素融合 | 已合并 |
| Anionex/h3-image-mlx | [#25](https://github.com/Anionex/h3-image-mlx/pull/25) | GEMM 尾块越界读 | 开放 |
| mflux-community/mflux | [#998](https://github.com/mflux-community/mflux/pull/998) | 文生图提示词增强 | 已合并 |
| mflux-community/mflux | [#999](https://github.com/mflux-community/mflux/pull/999) | Qwen-Image W8A8 DiT | 已合并 |
| mflux-community/mflux | [#1000](https://github.com/mflux-community/mflux/pull/1000) | VAE 分条解码 | 已合并 |
| mflux-community/mflux | [#1018](https://github.com/mflux-community/mflux/pull/1018) | W8A8 输入检查 | 开放 |
| mflux-community/mflux | [#1020](https://github.com/mflux-community/mflux/pull/1020) | 去噪 block 逐元素融合 | 开放 |
| mflux-community/mflux | [#1022](https://github.com/mflux-community/mflux/pull/1022)、issue [#1019](https://github.com/mflux-community/mflux/issues/1019) | 零拷贝、逐层流式 | 开放 |
| ml-explore/mlx-examples | [#1464](https://github.com/ml-explore/mlx-examples/pull/1464) | Whisper 融合注意力 | 开放（不再跟进） |
| moona3k/mlx-qwen3-asr | [#32](https://github.com/moona3k/mlx-qwen3-asr/pull/32) | 贪心解码流水线 | 开放（搁置） |
| Neywa-Labs/mlx-audio | [#1206](https://github.com/Neywa-Labs/mlx-audio/pull/1206) | SOV 解码流水线 | 已合并 |
| Neywa-Labs/mlx-audio | [#1234](https://github.com/Neywa-Labs/mlx-audio/pull/1234) | Qwen3-TTS 量化 embedding 加载 | 开放 |
| Neywa-Labs/mlx-audio | [#1235](https://github.com/Neywa-Labs/mlx-audio/pull/1235) | Qwen3-ASR 量化音频塔加载 | 开放 |
| Neywa-Labs/mlx-audio | [#1236](https://github.com/Neywa-Labs/mlx-audio/pull/1236) | Qwen3-ASR prompt 模板 | 已合并 |
| Neywa-Labs/mlx-audio | [#1237](https://github.com/Neywa-Labs/mlx-audio/pull/1237) | Qwen3-ASR 语言码、中日文拼接 | 已合并 |
| Neywa-Labs/mlx-audio | [#1238](https://github.com/Neywa-Labs/mlx-audio/pull/1238) | Qwen3-ASR 重复停止 | 关闭（未采纳） |
| Neywa-Labs/mlx-vlm | [#2789](https://github.com/Neywa-Labs/mlx-vlm/pull/2789) | EmbeddingGemma 2 长输入注意力 | 开放 |
| Niko1221/Strata | [#6183](https://github.com/Niko1221/Strata/pull/6183)、issue [#6182](https://github.com/Niko1221/Strata/issues/6182) | `--prefill auto` 主机与 GPU 重叠 | 开放（草稿） |
| Niko1221/Strata | [#6415](https://github.com/Niko1221/Strata/pull/6415) | resident prefill 跳过旧草稿行 | 开放（草稿） |
| Niko1221/Strata | issue [#6410](https://github.com/Niko1221/Strata/issues/6410)–[#6412](https://github.com/Niko1221/Strata/issues/6412) | 省内存双槽、主机行预取、M5 Ultra prompt 内核 | 开放（方案讨论） |

## 目录结构

仓库就是部署：clone 到 `~/llm-serving`，线上运行的脚本、服务代码和 llama-swap／网关配置就是仓库里的文件。
同一棵树里还有本机私有的东西（权重链接、venv、日志、密钥、fork 子仓库、实验），它们被 `.gitignore` 默认忽略：
只有 [`public-paths.txt`](public-paths.txt) 列出的路径公开（`tools/gen_gitignore.py` 由它生成 `.gitignore`）。

| 目录 | 内容 |
| --- | --- |
| `bin/` | gpu-lock（显存预约、排队、托管运行）、gpu-top、gpu-pool、memguard、lsw-apply、archive-restore／model-archive、regress、health-check |
| `platform/llama-swap/` | `config.yaml`、[`conf.d/*.yaml`](platform/llama-swap/conf.d/)（本机路径写成 `${env.LLM_SERVING}` 等）、`env.sh`、`serve.sh` |
| `platform/gateway/` | Caddyfile（`{$LLM_SERVING}`、`{$LLM_TRUSTED_PROXY}`） |
| `platform/launchd/` | 各 LaunchAgent 模板（`*.plist.in`，渲染到 `~/Library/LaunchAgents`） |
| `platform/agent-bridge/` | llm-collab MCP 启动器与配置、多 agent 规则、本地 worker |
| `services/<类别>/` | 各模型的接口服务：text/strata（[引擎配置模板](services/text/strata/configs/)）、embedding、rerank、decisions、asr、tts、image、video |
| `shared/common/` | MLX W8A8／8 位公共代码 |
| `standby/` | 已下线但保留的代码（`standby/asr/sov/`） |
| `clients/` | MCP 服务（`clients/mcp/`）与测试脚本 |
| `docs/`、`tools/`、`tests/` | 文档、安装与检查工具（[原生构建脚本](tools/native.py)）、单元测试 |

各引擎和库的完整 commit 固定在 `native.lock.json`（原生引擎）和 `uv.lock`（Python 包，含 MLX fork 的 wheel）。

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

GPU 池由 `bin/gpu-pool` 按当前 `iogpu.wired_limit_mb` 计算（减去系统与常驻嵌入的预留）；聊天模型预约 **69,500 MiB**，池至少要这么大。
工具不会因为识别出一台 Mac 就提高内存上限，也不会把这台机器的上下文／并发参数缩放成未经验证的其他机型参数。

## 模型与状态

- 模型目录完全独立，必要布局见 `models.required.json`。转换过的文件要一并迁移：Strata pack、INT8 视频权重（[准备步骤](docs/h3-video.md)）、页对齐图片权重（Qwen-Image 见[准备步骤](docs/qwen-image.md)）、各模型预量化的 8 位或按层类混合副本（`save_q8.py`、`save_mix.py` 生成）。原始 Hub 权重不一定能直接替代这些文件。
  归档到另一台机器的模型由 `bin/archive-restore` 在第一次请求时按清单拷回并校验（`<MODELS>/<名>.archive`）。
- 运行数据在仓库目录里，全部被忽略：API key `platform/llama-swap/.api_key`（0600）、活动数据库、`logs/`、图片 `data/media/images`、视频 `services/video/h3-video/outputs`、图片阶段缓存 `data/cache/`。模型权重和运行数据都不提交到 Git。
- 迁移时可以保留旧 API key，放到 `platform/llama-swap/.api_key`，权限设为 `0600`；否则 `install_serving.py` 会生成一个新凭证，供客户端使用。不要将密钥放到命令行参数或公开配置中。
- 有权重文件后，推理不需要外部 API key 或下载 token。个别模型首次下载可能需要模型托管站点的授权；这属于准备阶段。

## 验证范围

已在开发机器使用 Python **3.15.2** 完成统一依赖安装与导入，以及 Numba JIT、PyTorch/SciPy CPU 和日语对齐 tokenizer 基础调用检查；同时完成 llama-swap/Caddy 配置校验、FFmpeg 编码器检查，以及安装路径和隐私扫描测试。重排的 Python 3.15 真实模型/API 检查见[专项说明](docs/rerank.md)。参照部署 2026-11-18 起所有 Python 入口都运行在这个统一环境里（同一份 `uv.lock`），切换前后 `bin/regress` 全部用例一致。

2026-11-23 仓库与部署合并时，从 GitHub 新 clone 到临时目录做过恢复演练：`uv sync`、渲染出的 LaunchAgent／Strata JSON／`serving.env` 与线上逐字节相同，llama-swap 与 Caddy 配置校验通过，单元测试通过；合并前后完整 `bin/regress` 逐例相同。尚未在干净系统完成全套原生重编译和端到端推理恢复演练，首次切换前仍需运行真实 API 冒烟检查。`doctor.py` 不加载模型，不替代正确性/性能验证。

更多说明：[多 agent 分工与恢复](docs/multi-agent.md)、[设计与硬件优化](docs/design.md)、[GPU 预约](docs/gpu-lock.md)、[运维：回归测试与健康检查](docs/operations.md)、[部署流程](docs/deploy-process.md)、[客户端接入与 MCP](docs/clients.md)、[发布与隐私](docs/privacy.md)、[零拷贝与 W8A8](docs/w8a8.md)、[SOV 语音转写（已下线）](docs/sov.md)。
