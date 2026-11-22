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
| 视频＋音频 | `minimax-h3-video` | [h3.c fork](https://github.com/ain-soph/h3.c/tree/serving) | `best`／`standard`／`turbo`／`fast` 四档（`turbo` 需另备 22 GB 权重，限 8 s），最长 15 s；INT8 DiT 零拷贝、按 block 释放；文本编码器与 H3 图片共用 INT8 文件 |

MLX 本身用 [ain-soph/mlx fork](https://github.com/ain-soph/mlx/tree/serving) 的 wheel（零拷贝加载、页对齐保存、`MLX_CACHE_LIMIT_MB`）。完整 commit 固定在 `native.lock.json`（原生引擎）和 `uv.lock`（Python 包）；[原生构建脚本](tools/native.py)、[模型路由模板](templates/conf.d/)、[Strata 配置](templates/strata/) 和自写接口都在本仓库。权重准备与服务说明：[Strata](docs/strata.md)、[重排](docs/rerank.md)、[零拷贝与 W8A8 副本](docs/w8a8.md)、[决策模型](docs/decisions.md)、[SOV 语音转写](docs/sov.md)、[语音合成](docs/tts.md)、[H3 图片](docs/h3-image.md)、[Qwen-Image](docs/qwen-image.md)、[H3 视频](docs/h3-video.md)。

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

## 安装与迁移

需要 Apple Silicon、macOS 及适用的 Apple Command Line Tools。先安装 [Homebrew](https://brew.sh)，再执行：

```sh
git clone https://github.com/ain-soph/llm-serving.git
cd llm-serving
./tools/bootstrap.sh
.venv/bin/python tools/native.py all
```

`bootstrap.sh` 最后还会把多 agent 分工规则装进 `~/.claude/CLAUDE.md`（标记块合并，不覆盖原有内容）并链接本地 worker，见[多 agent 分工与恢复](docs/multi-agent.md)。

`Brewfile` 统一安装 **Homebrew Python 3.15**、uv、Caddy、llama-swap、llama.cpp、FFmpeg、CMake、Ninja。uv 使用 Homebrew 的 Python 3.15 创建项目唯一的 `.venv`：**线上只有这一个环境**，所有 Python 服务使用同一个解释器；开发新模型时用它自己的 `.venv-dev`，上线时并进来。`pyproject.toml` 只写版本下限，`uv.lock` 精确锁定全部依赖，包括各 fork 的完整 commit。

原生引擎从 `native.lock.json` 的完整 commit 构建到 `.local/src/`。编译后的可执行文件、动态库、Metal shader 和构建信息留在同一个本地构建树，不能只搬一个可执行文件。

将已验证的模型目录复制到新机器上的任意位置，再生成本机配置：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models"
.venv/bin/python tools/doctor.py
.venv/bin/python tools/install.py --activate
```

`configure.py` 只生成本机文件，不启动服务。默认监听 `127.0.0.1:8080`；提供局域网接口时添加 `--listen 0.0.0.0`。可用 `--state`、`--output` 自定义运行数据和配置目录。代码可以放在任意目录，路径含空格也支持。

当前完整聊天配置要求 GPU 池至少 **65200 MiB**。默认从当前 `iogpu.wired_limit_mb` 减去系统与常驻嵌入预留的 6032 MiB；系统上限为 0 时要求显式给出已验证的 `--pool-mb`。工具不会因为识别出一台 Mac 就自动提高内存上限，也不会把这台机器的上下文/并发参数缩放成未经验证的其他机型参数。

若迁移机器需要持久化 GPU 内存上限，可以显式指定，例如针对已经验证过的 96 GiB 机器：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models" --pool-mb 82032
.venv/bin/python tools/install.py --wired-limit-mb 88064 --activate
```

此选项通过 `sudo` 安装 root 拥有的 LaunchDaemon；普通模型服务由当前用户的 LaunchAgent 运行，在用户登录后启动。若旧部署仍占用 8080/8081/8090，应先停掉旧部署，再激活新服务。

## 模型与状态

- 模型目录完全独立，必要布局见 `models.required.json`。转换过的 Strata pack、INT8 视频权重（[准备步骤](docs/h3-video.md)）和页对齐图片权重（Qwen-Image 见[准备步骤](docs/qwen-image.md)）应一并迁移；原始 Hub 权重不一定能直接替代这些文件。
- 运行数据默认在 `~/Library/Application Support/llm-serving`：API key、活动数据库、日志、图片与视频。模型权重和运行数据都不提交到 Git。
- 迁移时可以保留旧 API key，放到新状态目录的 `api_key` 文件，权限设为 `0600`；否则配置工具会生成一个新凭证，供客户端使用。不要将密钥放到命令行参数或公开配置中。
- 有权重文件后，推理不需要外部 API key 或下载 token。个别模型首次下载可能需要模型托管站点的授权；这属于准备阶段。

## 验证范围

已在开发机器使用 Python **3.15.2** 完成统一依赖安装与导入，以及 Numba JIT、PyTorch/SciPy CPU 和日语对齐 tokenizer 基础调用检查；同时完成配置生成、llama-swap/Caddy 配置校验、FFmpeg 编码器检查，以及安装路径和隐私扫描测试。重排的 Python 3.15 真实模型/API 检查见[专项说明](docs/rerank.md)。参照部署 2026-11-18 起所有 Python 入口都运行在这个统一环境里（同一份 `uv.lock`），切换前后 `tools/regress` 全部用例一致；公开仓库本身的生成配置仍只做了静态检查。

尚未在干净系统完成全套原生重编译和端到端推理恢复演练。本仓库提供可审查的迁移工具和当前配置，首次切换前仍需运行真实 API 冒烟检查。`doctor.py` 不加载模型，不替代正确性/性能验证。

更多说明：[多 agent 分工与恢复](docs/multi-agent.md)、[设计与硬件优化](docs/design.md)、[运维：回归测试与健康检查](docs/operations.md)、[客户端接入与 MCP](docs/clients.md)、[发布与隐私](docs/privacy.md)。
