# llm-serving

面向 Apple Silicon 的本地模型接口与部署工具。当前参数来自 **M5 Ultra、96 GiB 统一内存**的单机部署，包含文本、嵌入、重排、语音识别、图片和视频。**主要文本后端是 Strata Metal**，提供 Qwen3.8 Flash-Next 及其 Uncensored 版本。

模型服务独立运行；不依赖另一台电脑、外部推理 API、聊天前端、MCP 或公网反向代理。

```text
客户端 → Caddy :8080 → llama-swap 127.0.0.1:8081 → 按需加载模型
                 └→ 只读媒体服务 127.0.0.1:8090
```

## 部署模型、后端与实测资源

测试平台：**Apple M5 Ultra / 96 GiB / macOS 27**，性能数据来自 2026-11-15～16 的独占 GPU 会话；图片 4096² 共存测试另行标注。本轮只整理已有测试、重新统计磁盘，没有重新跑推理。不同输入、上下文、缓存冷热和并发下的数字不可直接比较。

下表的磁盘为**运行所需模型权重与配置文件**的逻辑大小（GB = 10⁹ bytes），不含源码、venv、媒体或评测备选权重。内存明确区分 **wired 增量、RSS、物理 footprint、GPU 分配峰值**，不是路由预留容量。具体条件、精度及数据来源见 [实测说明](docs/benchmarks.md)，磁盘字节数见 [文件统计](docs/model-storage.json)。

| 类别 | 模型 ID | 后端 | 我们的修改 | 实测速度／耗时 | 实测内存 | 权重磁盘 |
| --- | --- | --- | --- | --- | --- | --- |
| 文本／视觉输入 | `qwen3.8-flash-next` | **Strata Metal fork** | **S：引擎修改**，修复 auto prefill 分块与主机/GPU 重叠 | prefill **2014 / 2225 tok/s**（26k / 116k 输入）；decode **117.2 tok/s**（c1）、74.4/路（c2） | wired 峰值 **+69868 MiB**（262k ctx × 2 槽） | **96.94 GB**，另共享 MTP 0.89 GB |
| 文本／视觉输入 | `qwen3.8-flash-next-uncensored` | Strata Metal fork | 同 S | 历史同规格 decode **118–139 tok/s**；fork 最终版的独立 prefill 数据待补 | 历史 wired 峰值 **+68771 MiB**（262k × 2） | **96.95 GB**，共享同一份 MTP |
| 文本嵌入 | `embeddinggemma-2` | Homebrew llama.cpp + 自写适配层 | **A：只改适配层**，维度、输入前缀、base64；不改引擎 | 4 篇共 22036 token：**0.93–1.04 s**（约 2.2 万 tok/s） | wired **+1664 MiB**，常驻 | **0.558 GB** |
| 多模态嵌入 | `embeddinggemma-2-multimodal` | 同上 | 同 A，增加多模态请求适配 | 图片 **0.25 s**；5 s 音频 **0.17 s**；4 s 视频 **0.49 s** | 用后 wired **+2996 MiB** | **1.540 GB**，已含共享的文本权重 |
| 重排 | `qwen3-reranker-0.6b` | Homebrew llama.cpp | **未改引擎**，标准 Q8_0 + 部署调参 | **0.36 s**，prefill 约 **22000 tok/s** | wired **+2.2 GB**；RSS **1.6 GB** | **0.639 GB** |
| 重排 | `qwen3-reranker-4b` | Homebrew llama.cpp | 未改引擎；自行转换为标准 Q8_0 | **1.64 s**，prefill 约 **4900 tok/s** | wired **+3.1–6.2 GB**；RSS **5.4 GB** | **4.280 GB** |
| 重排 | `qwen3-reranker-8b` | Homebrew llama.cpp | 未改引擎；自行转换为标准 Q8_0 | **2.56 s**，prefill 约 **3100 tok/s** | wired **+5.0–6.6 GB**；RSS **9.0 GB** | **8.047 GB** |
| 语音转写 | `nemotron-3.5-transcribe` | NeMo-Speech.cpp Metal | **未改引擎**，AD-IQ4_NL 部署配置 | 英文 100 条、c4：**22.66 s**（批次总耗时） | wired **+22284 MiB**；RSS **19318 MiB** | **21.35 GB**，含 F16 编码器 |
| 语音转写 | `whisper-large-v3` | mlx-whisper | **W：包与服务层**，按需导入词级对齐；单 worker、512 MiB 缓存上限 | 英文 100 条、c4：**26.20 s**（批次总耗时） | wired **+4545 MiB**；RSS **3509 MiB** | **3.084 GB** |
| 语音转写 | `qwen3-asr-1.7b` | mlx-qwen3-asr | **A：自写服务层**，本地对齐器、单 worker、512 MiB 缓存上限 | 英文 100 条、c4：**16.97 s**；主模型 phase 测试 prefill **7776** / decode **179.8 tok/s** | wired **+6776 MiB**；加载/预热 profiler 峰值 **8.16 GB**（另一轮测试） | **6.543 GB**，含 0.6B 对齐器 |
| 图片生成／编辑 | `minimax-h3-image` | **h3-image-mlx fork** | **I：引擎修改**，DiT 缓存预算、DiT/文本编码器页对齐 mmap 零拷贝 | 6 步：1024² **4.19 s**、2048² **12.18 s**；4096² 与空闲 Flash-Next 共存约 **120 s** | DiT footprint **1.95 / 3.29 GB**（1024² / 2048²）；不含可回收权重页缓存 | **49.32 GB** |
| 视频／音频生成 | `minimax-h3-video` | **h3.c fork** | **V：引擎+服务层修改**，INT8 磁盘打包、BF16 VAE；fast 档稀疏注意力与 token 削减 | fast、1344×768：约 5 s 片段 **170 s**；约 15 s **758 s**（优化会话） | 同轮 DiT GPU 分配峰值 **22.20 / 33.00 GiB**（5 s / 15 s），不是 wired | **125.62 GB**，两种服务模式及 FL2VA/Ref2VA 共享，勿重复相加 |
| 视频／音频生成 | `minimax-h3-video-stream` | h3.c fork，SSD streaming | 同 V，按层流式读取 INT8 DiT | 打包验证：512×288、25 帧 **19.82 s**（VAE FP32、稀疏关闭）；最终 fast 档待补测 | 同一打包验证 footprint **10.80 GB**；不能套用到 768p/15 s | **与上一行完全共享** |

重排负载为 1 个查询 × 20 篇约 400 token 文档；ASR 的 c4 是客户端并发，MLX 后端由单 worker 顺序推理。视频三档是同一模型的请求预设，不是三份权重。尚未在最终打包版本重测的项目保留原测试条件，不将历史数字冒充当前全部场景的验收结果。

## 修改前后的收益与代价

“原版”在这里指各行写明的**具体本机基线**；没有官方实现的同口径实测就不标“比官方快”。S/I/V 是引擎修改，W 是 Whisper 包的导入修改，A 是服务层修改，换后端和量化选择另作说明。

| 修改 | 对照负载与基线 | 修改前 → 修改后 | 速度变化 | 内存变化 | 磁盘变化／正确性 |
| --- | --- | --- | --- | --- | --- |
| **S：Strata auto prefill 修复** | 同规格 262k × 2；上游 v0.2.13 默认 auto 与 fork 热缓存，基线来自较早会话 | 26k：1854 → **2014 tok/s**；116k：1999 → **2225 tok/s** | 吞吐 **+8.6% / +11.3%**；不是同次随机交错 A/B | 没有证明节省内存；最终 wired **+69868 MiB** | 模型权重不变；与上游手动调优配置的 38k prompt 回答逐字节一致 |
| **S：与已手动调优的上游对照** | 同一次 A/B；上游 `prefill 4096 + chunk 1024` 对 fork auto | 116k：2221.8 → **2225 tok/s**；decode 116.7 → **117.2** | **+0.1% / +0.4%**，视为基本持平 | 两臂 wired 峰值均 **69868 MiB** | 收益是修复默认路径、去掉手动覆盖，不宣称额外大幅提速 |
| **I：零拷贝（固定缓存预算）** | 同一 fork 关闭/启用零拷贝的配对测试；1024²、6 步、预算 8000 | **13.109 → 4.190 s** | **3.13×**；耗时 **−68.0%** | DiT footprint **2.88 → 1.95 GB（−32.3%）**；wired **1581 → 2681 MiB（+69.6%）** | 最终模型 **49.32 GB**；页对齐前后精确磁盘差未留档；PNG 逐字节一致 |
| **I：同上，2048²** | 同一配对测试与预算 | **16.038 → 12.181 s** | **1.32×**；耗时 **−24.0%** | DiT footprint **4.70 → 3.29 GB（−30.0%）**；wired **3457 → 2892 MiB（−16.3%）** | 同上；不是新权重量化，未额外保留一套未对齐权重 |
| **V：INT8 磁盘打包** | 512×288、25 帧；新二进制读 BF16 并即时量化，对预打包 INT8；VAE FP32、稀疏关闭 | **28.92 → 17.95 s** | **1.61×**；耗时 **−37.9%**，主要省加载 | 该小样本进程 footprint **18.36 → 22.69 GB（+23.6%）**，不宣称全程省内存 | 每套 DiT 约 **66.28 → 23.91 GB（−63.9%）**；无条件文生视频逐字节一致，条件生成落在原版自身波动范围 |
| **V：BF16 VAE** | 1344×768、124 帧；基线 FP32 VAE，对 BF16 线性层+注意力 | VAE 阶段 **79.01 → 33.59 s** | **2.35×**；阶段耗时 **−57.5%** | VAE GPU 分配峰值 **9.554 → 5.139 GiB（−46.2%）**；不是全进程峰值 | 权重磁盘不变；有舍入差异；仅 BF16 线性层方案 PSNR 51.7 dB，完整方案未测 PSNR |
| **V：fast 视频预设** | INT8 h3 本机基线，45 层/reuse 2、密集注意力/FP32 VAE；加稀疏、token 削减及 BF16 VAE | 768p、约 5 s：**412 → 170 s** | **2.42×**；耗时 **−58.7%** | 同轮 DiT GPU 峰值 **22.191 → 22.196 GiB**，基本不变；VAE 阶段另见上行 | 不额外存权重；**近似算法，构图可能改变，不是无损加速** |
| **W：Whisper 按需导入对齐模块** | Python 3.15.2；7 组交替新进程，仅测包导入 | **0.530 → 0.134 s**（中位数） | **3.96×**；导入耗时 **−74.7%**，不是转写吞吐 | 导入进程 RSS 峰值中位数 **167.3 → 71.4 MiB**，约少 **96 MiB**；模型 MLX 峰值两臂均 **4.069 GB** | 权重不变；普通/词级输出及 JSON/verbose JSON/SRT/VTT 对照一致，见[专项测试](docs/whisper-startup.md) |
| **A：Whisper 缓存上限** | 同一 MLX 后端，无缓存上限 → 512 MiB；长音频 verbose+c4 复测 | c4 **25.65 → 26.20 s** | 耗时 **+2.1%**，基本持平 | wired **5804 → 4545 MiB（−21.7%）** | 权重不变；复测错误率相同 |
| **A：Qwen ASR 缓存上限** | 同一 MLX 后端与长音频/c4 复测 | c4 **16.95 → 16.97 s** | 耗时 **+0.1%**，基本持平 | wired **10332 → 6776 MiB（−34.4%）** | 权重不变；复测错误率相同 |

速度比是基线耗时 ÷ 新耗时；吞吐提升按新值 ÷ 旧值 − 1 计算。大部分结果是单次样本，没有置信区间。图片零拷贝节省拷贝与部分进程 footprint，但权重仍占用文件页缓存；不能把“1.95 GB footprint”理解为整台机器只需要 1.95 GB 内存。

Strata 替代 llama.cpp 的历史测试约为单流 **59.9 → 123.2 tok/s（2.06×）**、长 prompt **1197 → 2058 tok/s（1.72×）**；两边槽位、上下文和 KV 精度不同，**不计入 Strata fork 的修改收益**。嵌入/重排没有本机官方实现的同配置速度基线；Whisper 切换后端的测试解码设置尚未完全对齐，均不据此声称修改后的加速倍数。

源码入口：[Whisper fork](https://github.com/ain-soph/mlx-examples/tree/serving/whisper)、[Strata fork](https://github.com/ain-soph/Strata/tree/serving)、[图片 fork](https://github.com/ain-soph/h3-image-mlx/tree/serving)、[视频 fork](https://github.com/ain-soph/h3.c/tree/serving)。完整 commit 固定在 `native.lock.json` 或 `uv.lock`；[原生构建脚本](tools/native.py)、[模型路由](templates/conf.d/)、[Strata 配置](templates/strata/) 和自写接口都在本仓库。详见 [Strata 部署说明](docs/strata.md)。

## 安装与迁移

需要 Apple Silicon、macOS 及适用的 Apple Command Line Tools。先安装 [Homebrew](https://brew.sh)，再执行：

```sh
git clone https://github.com/ain-soph/llm-serving.git
cd llm-serving
./tools/bootstrap.sh
.venv/bin/python tools/native.py all
```

`Brewfile` 统一安装 **Homebrew Python 3.15**、uv、Caddy、llama-swap、llama.cpp、FFmpeg、CMake、Ninja。uv 使用 Homebrew 的 Python 3.15 创建项目唯一的 `.venv`；所有 Python 服务使用同一个解释器。`uv.lock` 固定 Python 依赖，包括图片引擎 fork 的完整 commit。

原生引擎从 `native.lock.json` 的完整 commit 构建到 `.local/src/`。编译后的可执行文件、动态库、Metal shader 和构建信息留在同一个本地构建树，不能只搬一个可执行文件。

将已验证的模型目录复制到新机器上的任意位置，再生成本机配置：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models"
.venv/bin/python tools/doctor.py
.venv/bin/python tools/install.py --activate
```

`configure.py` 只生成本机文件，不启动服务。默认监听 `127.0.0.1:8080`；提供局域网接口时添加 `--listen 0.0.0.0`。可用 `--state`、`--output` 自定义运行数据和配置目录。代码可以放在任意目录，路径含空格也支持。

当前完整聊天配置要求 GPU 池至少 **71000 MiB**。默认从当前 `iogpu.wired_limit_mb` 减去系统与常驻嵌入预留的 4872 MiB；系统上限为 0 时要求显式给出已验证的 `--pool-mb`。工具不会因为识别出一台 Mac 就自动提高内存上限，也不会把这台机器的上下文/并发参数缩放成未经验证的其他机型参数。

若迁移机器需要持久化 GPU 内存上限，可以显式指定，例如针对已经验证过的 96 GiB 机器：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models" --pool-mb 83192
.venv/bin/python tools/install.py --wired-limit-mb 88064 --activate
```

此选项通过 `sudo` 安装 root 拥有的 LaunchDaemon；普通模型服务由当前用户的 LaunchAgent 运行，在用户登录后启动。若旧部署仍占用 8080/8081/8090，应先停掉旧部署，再激活新服务。

## 模型与状态

- 模型目录完全独立，必要布局见 `models.required.json`。转换过的 Strata pack、INT8 视频权重和页对齐图片权重应一并迁移；原始 Hub 权重不一定能直接替代这些文件。
- 运行数据默认在 `~/Library/Application Support/llm-serving`：API key、活动数据库、日志、图片与视频。模型权重和运行数据都不提交到 Git。
- 迁移时可以保留旧 API key，放到新状态目录的 `api_key` 文件，权限设为 `0600`；否则配置工具会生成一个新凭证，供客户端使用。不要将密钥放到命令行参数或公开配置中。
- 有权重文件后，推理不需要外部 API key 或下载 token。个别模型首次下载可能需要模型托管站点的授权；这属于准备阶段。

## 验证范围

已在开发机器使用 Python **3.15.2** 完成统一依赖安装与导入，以及 Numba JIT、PyTorch/SciPy CPU 和日语对齐 tokenizer 基础调用检查；同时完成配置生成、llama-swap/Caddy 配置校验、FFmpeg 编码器检查，以及安装路径和隐私扫描测试。当前生产服务仍使用原来的目录和环境。

尚未在干净系统完成全套原生重编译和端到端推理恢复演练。本仓库提供可审查的迁移工具和当前配置，首次切换前仍需运行真实 API 冒烟检查。`doctor.py` 不加载模型，不替代正确性/性能验证。

更多说明：[设计与硬件优化](docs/design.md)、[发布与隐私](docs/privacy.md)。
