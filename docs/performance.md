# 性能与资源报告

测试平台：Apple **M5 Ultra / 96 GiB 统一内存 / macOS 27**；测试日期 2026-11-15 ～ 17；速度测量都独占 GPU（共存测试另行注明）。
各项的测试条件见每节第一行；指标的含义（wired、RSS、footprint 等）和计算口径见文末[测量方法与口径](#测量方法与口径)。

## 总览

| 类别 | 模型 ID | 后端 | 权重精度 | 速度（关键数） | 内存（关键数） | 权重磁盘 |
| --- | --- | --- | --- | --- | --- | ---: |
| 文本／视觉 | `qwen3.8-flash-next` | Strata Metal fork | AD-4.27 混合（3.57 bpw 主体） | prefill 2014／2225 tok/s（26k／116k）；decode 117.2 tok/s | wired +69868 MiB（262k × 2 槽） | 96.94 GB |
| 文本／视觉 | `qwen3.8-flash-next-uncensored` | Strata Metal fork | 同上 | decode 118–139 tok/s（历史） | wired +68771 MiB（历史） | 96.95 GB |
| 文本嵌入 | `embeddinggemma-2` | llama.cpp + 适配层 | BF16 | 22036 token 0.93–1.04 s | wired +1664 MiB（常驻） | 0.558 GB |
| 多模态嵌入 | `embeddinggemma-2-multimodal` | 同上 | BF16 | 图片 0.25 s；音频 0.17 s；视频 0.49 s | wired +2996 MiB | 1.540 GB |
| 重排 | `qwen3-reranker-0.6b` | MLX，自写服务 | BF16 | 20 篇 0.137 s | RSS 1.66 GB；wired +2772 MiB | 1.203 GB |
| 重排 | `qwen3-reranker-4b` | 同上 | BF16 | 20 篇 0.847 s | RSS 8.45 GB；wired +8694 MiB | 8.055 GB |
| 重排 | `qwen3-reranker-8b` | 同上 | Q8_0 等价 MLX 8-bit | 20 篇 1.793 s | RSS 9.68 GB；wired +9869 MiB | 9.228 GB |
| 语音转写 | `nemotron-3.5-transcribe` | NeMo-Speech.cpp Metal | AD-IQ4_NL + F16 编码器 | 100 条 c4 22.66 s | wired +22284 MiB | 21.35 GB |
| 语音转写 | `whisper-large-v3` | mlx-whisper fork | FP16 | 100 条 c4 26.20 s | wired +4545 MiB | 3.084 GB |
| 语音转写 | `qwen3-asr-1.7b` | mlx-qwen3-asr | BF16 文件／FP16 运行 | 100 条 c4 16.97 s | wired +6776 MiB | 6.543 GB |
| 图片 | `minimax-h3-image` | h3-image-mlx fork | INT8 ConvRot（W8A8） | 1024² 4.19 s；2048² 12.18 s（6 步） | DiT footprint 1.95／3.29 GB | 49.32 GB |
| 图片 | `qwen-image-2.1-turbo` | mflux fork | W8A8 DiT + BF16 文本编码器 | 1024² 约 9 s；2048² 约 43 s（8 步） | 生成期 wired 6.7–9.0 GB | 42.78 GB¹ |
| 图片 | `qwen-image-2.1` | mflux fork | 同上 | 1024² 约 28 s（40 步） | 同上 | 同上¹ |
| 图片 | `qwen-image-2.1-turbo-int8`／`qwen-image-2.1-int8` | mflux fork | W8A8 DiT + INT8 文本编码器 | 与对应 BF16 文本编码器入口相同 | 页缓存少 8.8 GB | 同上¹ |
| 视频＋音频 | `minimax-h3-video` | h3.c fork | INT8 DiT、BF16 VAE 计算 | fast 768p：5 s 片段 170 s；15 s 片段 758 s | DiT GPU 峰值 22.20／33.00 GiB | 125.62 GB² |
| 视频＋音频 | `minimax-h3-video-stream` | h3.c fork，SSD 流式 | 同上 | 512×288 25 帧 19.82 s（验证样本） | footprint 10.80 GB（验证样本） | 同上² |

¹ 四个 Qwen-Image 入口共用一套文件。² 两个视频入口共用一套文件，不要重复相加。

## 文本：Qwen3.8 Flash-Next（Strata）

**条件**：AD-4.27、262144 上下文 × 2 槽、MTP Q2_0（spec=3）、KV INT8；输入 26,106／115,855 token；热 shader。

| 指标 | `qwen3.8-flash-next` | `-uncensored` |
| --- | --- | --- |
| prefill | 2014 tok/s（26k）；2225 tok/s（116k） | fork 版未单独测 |
| decode | 117.2 tok/s（c1）；74.4 tok/s／路（c2） | 118–139 tok/s（fork 切换前，同规格） |
| wired 峰值 | +69868 MiB | +68771 MiB（历史） |
| 权重 | 专家 IQ4_NL／IQ2_S／IQ3_S，dense Q8_0；MTP Q2_0 | 同左 |
| 磁盘 | 96.94 GB（Metal 主体 56.133、PLE 38.411、pack 约 1.496、视觉投影约 0.904）+ 共享 MTP 0.89 GB | 96.95 GB（独立权重） |

**我们的修改：Strata `--prefill auto` 修复**（fork 分支 `serving`）

| 对照 | 修改前 | 修改后 | 变化 |
| --- | --- | --- | --- |
| 上游 v0.2.13 默认 auto → fork auto（S1→S2，跨会话） | 1854／1999 tok/s | 2014／2225 tok/s | +8.6%／+11.3% |
| 上游手动调优（prefill 4096 + chunk 1024）→ fork auto（S2，同次 A/B） | 116k 2221.8；decode 116.7 | 2225；117.2 | +0.1%／+0.4%，持平 |
| wired 峰值（S2 两臂） | 69868 MiB | 69868 MiB | 不变 |

- 收益是修好默认路径、去掉手动参数；与手动调优的上游相比不额外提速。
- 正确性：38k 提示词、temperature=0，与上游手动调优版回答逐字节一致；API 对话、流式、工具、视觉通过。
- 第一遍 26k 输入测得 1230.5 tok/s，含约 7.3 s Metal 内核编译，表中用热 shader 的 2013.8。28123 token 的 2269.5 tok/s 输入不同，不替换同输入对照。
- 换后端的历史对比（S4，不计入 fork 收益，两边槽位、上下文、KV 精度不同）：llama.cpp → Strata 单流 59.9 → 123.2 tok/s（2.06×），长提示词 1197 → 2058 tok/s（1.72×）。

## 嵌入：EmbeddingGemma 2

**条件**：Homebrew llama.cpp，文本 4 槽 × 8192；4 篇共 22036 token；多模态分别测图片、5 s 音频、4 s 视频。

| 入口 | 速度 | 内存 | 磁盘 |
| --- | --- | --- | --- |
| `embeddinggemma-2` | 0.93–1.04 s（约 2.2 万 tok/s） | wired +1664 MiB，常驻 | 0.558 GB |
| `embeddinggemma-2-multimodal` | 图片 0.25 s；音频 0.17 s；视频 0.49 s | 用后 wired +2996 MiB | 1.540 GB（含文本 0.558 + 投影 0.982） |

- 权重 BF16 GGUF，投影器 BF16；无生成 KV，llama.cpp 上下文缓冲默认 F16；无 decode。
- 我们的修改只在适配层（维度、输入前缀、base64、多模态请求），不改引擎。
- 正确性：文本 BF16 GGUF 与官方 float32 CPU 输出余弦 1.00000，6/6 检索排序一致；多模态因预处理不同有差异。没有本机官方实现的同配置速度基线。

## 重排：Qwen3-Reranker

**条件**：1 个查询 × 20 篇文档，共 9950 个实际 prompt token，c1；MLX 批量预算 8192 token、缓存 256 MiB；Python 3.15 统一环境复测（[验证记录](rerank-validation.json)、[说明](rerank.md)）。

| 入口 | 精度 | 耗时 | 端到端输入吞吐 | RSS | wired | 磁盘 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-reranker-0.6b` | BF16 | 0.137 s | 72466 tok/s | 1.657 GB | +2772 MiB | 1.203 GB |
| `qwen3-reranker-4b` | BF16 | 0.847 s | 11753 tok/s | 8.453 GB | +8694 MiB | 8.055 GB |
| `qwen3-reranker-8b` | Q8_0 等价 MLX 8-bit（group 32，FP16 scale/bias，FP16 激活） | 1.793 s | 5548 tok/s | 9.680 GB | +9869 MiB | 9.228 GB |

**我们的修改：改用 MLX 批量服务，只算 yes/no 输出头**（相对已退休的本机 llama.cpp Q8_0，1 槽、8192 ctx/batch/ubatch）

| 尺寸 | 耗时 前 → 后 | 倍数 | 历史输入吞吐 前 → 后 | RSS 前 → 后 | 磁盘 前 → 后 |
| --- | --- | ---: | --- | --- | --- |
| 0.6B | 0.36 → 0.13 s | 2.77× | 约 22000 → 61000 tok/s | 1.6 → 1.66 GB | 0.639 → 1.203 GB |
| 4B | 1.64 → 0.84 s | 1.95× | 约 4900 → 9500 tok/s | 5.4 → 8.45 GB | 4.280 → 8.055 GB |
| 8B | 2.56 → 1.80 s | 1.42× | 约 3100 → 4400 tok/s | 9.0 → 9.68 GB | 8.047 → 9.228 GB |

- 历史吞吐按每篇约 400 token 估算；旧 wired 2.2 GB、3.1–6.2 GB、5.0–6.6 GB（后两者为测试记录范围）。
- 提速是后端、批处理、精度（0.6B/4B 改 BF16）一起换的结果，不能归给单一改动；内存是跨会话对比（新值含加载峰值）。
- 正确性：相对官方 BF16 最大分数差约 0.001／0.010／0.004，三个公开查询排序一致（旧 4B/8B GGUF 为 0.025／0.007）。

## 语音转写

**条件**：英文 100 条，c1／c4（c4 是客户端并发、批次墙钟，MLX 后端单 worker 顺序推理）；不含首次加载，含 HTTP 和音频预处理。长音频复测：英文 946.9 s、中文 722.6 s。

| 入口 | 权重 | 100 条 c1 | 100 条 c4 | wired | 其他内存 | 磁盘 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `nemotron-3.5-transcribe` | decoder AD-IQ4_NL + encoder F16 | 30.03 s | 22.66 s | +22284 MiB | RSS 19318 MiB；4 槽 × 18432 KV cells（共 73728） | 21.35 GB（decoder 19.679 + encoder 1.670） |
| `whisper-large-v3` | FP16 | 28.80 s | 26.20 s | +4545 MiB | RSS 3509 MiB；top MEM 4807 MiB | 3.084 GB |
| `qwen3-asr-1.7b` | 文件 BF16，运行 FP16；对齐器 FP16 | — | 16.97 s | +6776 MiB | 加载／预热 MLX 峰值 8.16 GB（另一轮） | 6.543 GB（主模型 4.703 + 对齐器 1.840） |

Qwen3-ASR 分阶段测量（英文前 20 条、122.2 s 音频、c1、1892 个 prompt token、441 步 decode，不含对齐器）：prefill 7776 tok/s，decode 179.8 tok/s，KV FP16。Nemotron、Whisper 的 prefill／decode tok/s 与 KV 精度尚未测（Nemotron 按默认实现推断 KV F16／recurrent F32）。

**我们的修改**

| 修改 | 对照 | 前 → 后 | 速度 | 内存 |
| --- | --- | --- | --- | --- |
| Whisper：按需导入词级对齐（fork） | Python 3.15.2，7 组交替新进程，只测导入 | 0.530 → 0.134 s（中位数） | 3.96×（导入耗时 −74.7%） | 导入 RSS 167.3 → 71.4 MiB（约 −96 MiB）；模型 MLX 峰值两臂均 4.069 GB |
| Whisper：MLX 缓存上限 512 MiB | 长音频 verbose + c4 | 25.65 → 26.20 s | +2.1%，持平 | wired 5804 → 4545 MiB（−21.7%） |
| Qwen3-ASR：MLX 缓存上限 512 MiB | 长音频 + c4 | 16.95 → 16.97 s | +0.1%，持平 | wired 10332 → 6776 MiB（−34.4%） |

- 正确性：缓存上限前后错误率相同；Whisper 导入修改后普通／词级输出及 JSON、verbose JSON、SRT、VTT 一致（[专项测试](whisper-startup.md)）。
- Nemotron 长中文自动模式曾输出英文翻译，低英文 WER 不代表中文可靠。
- Whisper.cpp 与 MLX 的历史解码设置未完全对齐，换后端的差异不算作修改收益；Qwen3-ASR 请求期 RSS 1053 MiB 漏掉了加载峰值，不代表模型大小。

## 图片：MiniMax H3 Image

**条件**：同提示词、seed=42、6 步、GPU 预算 8000、`H3_ASYNC=1`；零拷贝开／关配对（I）。

| 尺寸 | 耗时 | DiT footprint | DiT wired |
| --- | ---: | ---: | ---: |
| 1024² | 4.19 s | 1.95 GB | 2681 MiB |
| 2048² | 12.18 s | 3.29 GB | 2892 MiB |
| 4096²，与空闲 Flash-Next 共存（I4） | 约 120 s | — | — |

权重：DiT 与文本编码器 INT8 ConvRot，VAE FP16；磁盘 49.32 GB（文本编码器 27.157、DiT 20.988、VAE 1.169），只保留页对齐版本。

**我们的修改：页对齐 + mmap 零拷贝**（fork 分支 `serving`）

| 尺寸 | 耗时 前 → 后 | 倍数 | footprint 前 → 后 | wired 前 → 后 |
| --- | --- | ---: | --- | --- |
| 1024² | 13.109 → 4.190 s | 3.13×（−68.0%） | 2.88 → 1.95 GB（−32.3%） | 1581 → 2681 MiB（+69.6%） |
| 2048² | 16.038 → 12.181 s | 1.32×（−24.0%） | 4.70 → 3.29 GB（−30.0%） | 3457 → 2892 MiB（−16.3%） |
| 文本编码阶段 | 2.619 → 0.526 s（1024²）；2.621 → 0.522 s（2048²） | 约 5× | — | — |

- PNG 逐字节一致。文本阶段的提速已包含在整体耗时里，不要再乘一次；更早的“缓存从 96 MB 放宽”是另一项改动，两臂都固定预算 8000，不计入这里。
- 零拷贝不消除权重的文件页缓存，1.95 GB footprint 不等于整机只需要 1.95 GB。
- 4096² 共存时，下一次聊天多约 2–3 s（页缓存替换）；独占出图约 71–74 s 但要重载聊天模型约 26 s，因此保留共存路径。

## 图片：Qwen-Image-2.1

**条件**：mflux fork（零拷贝 + 逐层流式 + 两段条带 VAE + W8A8）；Turbo 8 步、base 40 步；单张图耗时（扩散模型没有 tok/s）。

| 入口 | 步数 | DiT | 文本编码器 | 1024² 耗时 | 其他尺寸 |
| --- | ---: | --- | --- | ---: | --- |
| `qwen-image-2.1-turbo` | 8 | W8A8 | BF16 | 约 9 s；经 API 含进程启动 14.0 s | 2048² 约 43 s；透明 8.2 s；编辑 10.5 s；遮罩 12.3 s |
| `qwen-image-2.1-turbo-int8` | 8 | W8A8 | INT8 | 同上 | 同上 |
| `qwen-image-2.1` | 40 | W8A8 | BF16 | 约 28 s；经 API 32 s | — |
| `qwen-image-2.1-int8` | 40 | W8A8 | INT8 | 同上 | — |

| 内存与磁盘 | 数值 |
| --- | --- |
| 生成期 wired 峰值：文生图 | 6.7–9.0 GB（1024² 约 7 GB；2048²、2752×1536、2400×1792 为 8.2–9.0 GB） |
| 生成期 wired 峰值：3／6／10 张参考图 | 5.6–8.4 GB |
| llama-swap 容量 | 11000 MiB（与 Flash-Next 共存：71000 + 11000 < 83192） |
| MLX 缓存 | 去噪 6 GB（带参考图 2 GB）；VAE 2 GB |
| 磁盘 | 42.78 GB（W8A8 DiT 7.26 × 2、INT8 文本编码器 9.35、BF16 文本编码器 17.53、FP32 VAE 1.35） |
| KV | 无生成 KV；文本／参考图前缀的 K/V 在步间复用，BF16 |

**我们的修改及收益**

| 修改 | 前 → 后 | 说明 |
| --- | --- | --- |
| 零拷贝（页对齐 safetensors + mmap） | 进程 RSS 33.8 → 2.0 GB | 输出逐字节一致 |
| 逐层流式 + MLX 缓存上限 + 两段条带 VAE | 生成期 wired 44–52 → 5.6–8.9 GB（BF16） | 条带与整图解码差最大 3/255（PSNR 74 dB） |
| W8A8 DiT（Comfy-Org `int8_convrot` + M5 TensorOps INT8 GEMM） | 去噪 Turbo 1024² 7.48 → 5.03 s（1.49×）；base 40.1 → 23.7 s（1.69×）；2048² 每 block 148 → 119 ms（1.25×） | 2048² 注意力占 41%，不加速 |
| 激活旋转量化改为 Metal 内核 | 16384×12288：7.2 → 2.2 ms | 快速 Hadamard |
| 去噪缓存 2 → 6 GB | 2048² 每 block 160 → 119 ms | 带参考图时仍用 2 GB |
| INT8 文本编码器（可选入口） | 页缓存 −8.8 GB | embedding 误差 2.6% → 6.2% |

**正确性**（同噪声与官方 diffusers 对比，第一步速度场相对 L2；官方自身 BF16 → FP16 为 10.9%）

| 配置 | Turbo | base |
| --- | ---: | ---: |
| mflux BF16 | 3.0% | 4.0% |
| + W8A8 DiT | 3.5% | 4.6% |
| + INT8 文本编码器 | 4.0% | 6.3% |

- 终图 PSNR（BF16，官方 vs mflux）：Turbo 28.5 dB、base 26.5 dB；编辑 43.9–44.0 dB。mflux 自带 Q8 在 Turbo 上改变构图（21.5 dB），不用。
- 共存压测（2752×1536 与 Flash-Next 11.6 万 token 提示词同时计算）：出图 90 s，Flash-Next 首 token 74.4 s，压缩区 10.9 GB，交换 +1.05 GB；日常交替使用不会同时计算。
- 许可：Qwen Research License（非商用）。

## 视频＋音频：MiniMax H3 Video

**条件**：1344×768；5 s 片段 = 124 帧（约 5.17 s），15 s 片段 = 362 帧（约 15.08 s）；45 层、reuse 2；fast 档（V1、V2）。

| 场景 | 总耗时 | DiT | VAE | DiT GPU 分配峰值 |
| --- | ---: | ---: | ---: | ---: |
| fast，5 s 片段 | 170 s | 128.630 s | 33.588 s | 22.196 GiB |
| fast，15 s 片段 | 758 s | 651.682 s | 97.801 s | 33.004 GiB |
| stream，512×288 25 帧（VP 验证样本，VAE FP32、稀疏关闭） | 19.82 s | — | — | footprint 10.80 GB |

- 预设三档：`best`、`standard`（默认，均精确计算）、`fast`（稀疏注意力 + token 削减，近似）；5 s 片段约 12／6／3 分钟。三档是同一模型的请求参数，不是三份权重。
- 权重：磁盘 DiT INT8 + F32 scale／AdaLN 表；文本编码器 BF16；VAE 文件 F32，服务默认 BF16 计算。无生成 KV。
- 磁盘 125.62 GB（FL2VA + Ref2VA，共享文本编码器和 VAE，第二个检查点只多约 23.937 GB）。
- **零拷贝 + 合并为单一入口：进行中**（2026-11-17）。768p 常驻 367 s，零拷贝 365 s，输出 MD5 相同；完成后补本节并去掉 `-stream` 入口。

**我们的修改及收益**

| 修改 | 对照 | 前 → 后 | 倍数 | 内存 | 磁盘／正确性 |
| --- | --- | --- | ---: | --- | --- |
| INT8 磁盘打包 | 512×288、25 帧、seed=7、20 步；BF16 读入即时量化 vs 预打包 | 28.92 → 17.95 s | 1.61×（−37.9%，主要省加载） | footprint 18.36 → 22.69 GB（+23.6%） | DiT 每份 66.28 → 23.91 GB（−63.9%）；无条件生成逐字节一致 |
| BF16 VAE | 1344×768、124 帧 | VAE 79.013 → 33.588 s | 2.35×（−57.5%） | VAE GPU 峰值 9.554 → 5.139 GiB（−46.2%） | 仅 BF16 线性层方案 PSNR 51.7 dB；完整方案未测 |
| fast 预设（稀疏 + token 削减 + BF16 VAE） | INT8 基线，密集注意力、FP32 VAE | 412 → 170 s（DiT 325.112 → 128.630 s） | 2.42×（−58.7%） | DiT 峰值 22.191 → 22.196 GiB | 近似，构图可能改变 |

- VP 三种加载方式的进程 RSS：BF16 12.787、预打包 resident 30.352、stream 15.223 GB；footprint 18.357／22.688／10.801 GB。
- 5 s 的 2.42× 由多项改动叠加，不能再乘 VAE 的 2.35×。15 s 片段历史记录 2727 → 758 s（约 3.60×），但不是同一会话的配对测试，只列新值。
- 首帧／参考条件生成的差异在原实现自身两次运行的波动范围内；条件生成不能保证逐字节复现。

## 磁盘明细

2026-11-16 统计（Qwen-Image 为 2026-11-17），排除 `.cache`，按 `(device, inode)` 去重硬链接；字节数见 [model-storage.json](model-storage.json)。

| 文件组 | 运行需要 | 相关目录总量 | 说明 |
| --- | ---: | ---: | --- |
| 默认 Flash-Next | 96.944 GB | 96.944 GB | 见文本一节 |
| Uncensored Flash-Next | 96.947 GB | 96.947 GB | 独立权重，不假设与默认版共享 extent |
| 共用 MTP | 0.889 GB | 10.556 GB | 只需 Q2_0 GGUF 与草稿词表；目录里的 Q4／Q8 和转换张量不是部署所需 |
| EmbeddingGemma 全部模态 | 1.540 GB | 1.540 GB | 文本 0.558 + 投影 0.982 |
| 三个 MLX 重排器 | 18.487 GB | 18.487 GB | 旧目录 116.349 GB 含评测副本，已清理 |
| Nemotron | 21.349 GB | 21.349 GB | decoder 19.679 + encoder 1.670 |
| Whisper MLX | 3.084 GB | 3.084 GB | FP16 |
| Qwen ASR + 对齐器 | 6.543 GB | 6.543 GB | 4.703 + 1.840 |
| H3 Image | 49.320 GB | 49.320 GB | 只保留页对齐版本 |
| Qwen-Image-2.1（4 个入口） | 42.777 GB | 71.283 GB | 运行时不读 BF16 DiT（2 × 14.23 GB，只读其 `config.json`） |
| H3 Video（FL2VA + Ref2VA） | 125.622 GB | 125.622 GB | 两种服务模式共用 |

视频 DiT 原始约 66.280 GB／检查点（当时逐张量报告），现 INT8 每份 23.914 GB（22.27 GiB），约省 42.366 GB（−63.9%）；只针对 DiT，不是整个视频目录。旧报告的“62 → 22 GB”混用了 GiB，以这里为准。

## 测量方法与口径

- **单位**：GB = 10⁹ bytes；MiB = 2²⁰ bytes；GiB = 2³⁰ bytes。EmbeddingGemma 原记录写 MB、实际按 2²⁰ 计算，这里统一为 MiB。
- **内存指标互不等同，不能相加**：
  - wired：系统锁定页相对空闲基线的增量（`vm_stat`）；
  - RSS：进程驻留页（不含 Metal 缓冲区，MLX 模型常常偏低）；
  - footprint：进程物理 footprint；
  - GPU 分配峰值：引擎记录的 Metal 分配峰值，不是 wired 或 RSS。
- **文件页缓存另计**：mmap 的权重（Strata 专家／PLE、图片模型的 DiT 与文本编码器）不算进某个 GPU 峰值，但仍占统一内存。
- **路由容量**（llama-swap `capacities`）是调度预留 = 实测峰值 + 余量，不是实测内存。
- **耗时**：不含首次下载和权重准备；图片阶段计时含子进程启动；Strata 热 shader 数据不含首次编译。
- **样本**：大多是单次测量，没有方差或置信区间。
- **收益计算**：倍数 = 旧耗时 ÷ 新耗时；吞吐增幅 = 新 ÷ 旧 − 1；耗时变化 = 新 ÷ 旧 − 1。没有同口径基线的项不计算收益；“原版”指各行写明的本机基线，没有同口径实测就不写“比官方快”。
- **环境**：数据来自各组件当时的环境（Strata／ASR 为 Python 3.12，图片、视频包装器为 3.15）；公开仓库的统一 3.15 环境只在重排上复测过，依赖验证不等于性能验收。
- **磁盘**：`stat().st_size` 逻辑大小；`model-storage.json` 另记 `st_blocks × 512`，APFS 共享 extent 未去重，不能用来精确计算可回收空间。不含源码、venv、Homebrew、uv 缓存和生成的媒体。

### 测试记录索引

原始结果目录不上传；以下是各节引用的测试。

| ID | 日期／记录 | 条件 |
| --- | --- | --- |
| S1 | 11-16，Strata prefill 分块 A/B 报告 | AD-4.27、262144 × 2、MTP Q2_0 spec=3、KV INT8；旧 auto 26,106／115,855 token：1854／1999 tok/s |
| S2 | 11-16，`strata-fork-auto-ab-1116-1748` | 同规格；上游手动 4096/1024 vs fork auto；26,106／115,856 输入；2013.8／2225.0 tok/s，decode 117.2／74.4；两臂 wired 69868 MiB |
| S3 | 11-15，`flashnext-final-1115-1804` | Uncensored，262144 × 2；decode 118–139 tok/s，wired 68771 MiB（fork 切换前） |
| S4 | 11-15，`compare-flashnext-1115-1651` | Strata 128k × 4、MTP、KV INT8 vs llama.cpp 262k × 8、无 MTP、KV F16；只说明换后端的整体差异 |
| E | 11-15，EmbeddingGemma Metal 部署记录 | 文本 4 槽 × 8192，22036 token；多模态单独测试 |
| R | 11-16，重排基准；复测见 [rerank-validation.json](rerank-validation.json) | 1 查询 × 20 篇，每篇约 400 token |
| A | 11-15，ASR 完整评测与 cache512 诊断 | 英文 100 条 c1／c4；长英文 946.9 s、长中文 722.6 s |
| I | 11-16，`textab0.jsonl`／`textab1.jsonl` 第二组 | seed=42、6 步、预算 8000、`H3_ASYNC=1`，零拷贝开／关 |
| I4 | 11-16，`image-alternate-test-1116-1726` | 4096²，Flash-Next 已加载但空闲 |
| Q | 11-16～17，Qwen-Image `bench/results/s4`、`s13`、`s24`、`s25`，`qwen-image-w8a8-deploy-1117-0005` | 见 Qwen-Image 一节 |
| V1 | 11-16，`s768/base.log`／`s768/trC.log` | 1344×768、124 帧，45 层／reuse 2 |
| V2 | 11-16，`p2-15s/trC.log` | 1344×768、362 帧 |
| VP | 11-16，`int8-pack-1116-0910` | 512×288、25 帧、seed=7、20 步、VAE FP32、稀疏关闭 |
