# 模型、内存与速度一览

每个 llama-swap 入口的用途、精度、内存预约、加载方式、能否同驻，以及大致速度。数字来自已有实测，出处见每节；完整数据与测量条件见[性能与资源报告](performance.md)。

**测量条件**：Apple M5 Ultra、96 GiB 统一内存、macOS 27；速度测量独占 GPU（没有其他模型同时计算），单路（并发 1）除非注明；
内存是 `vm_stat` 的 wired 增量峰值（相对空载基线）。多数是单次或两次测量，“约”表示取整或跨版本的近似值。状态截至 2026-11-24。

## 1. 内存池与调度规则

| 项 | 值 | 说明 |
| --- | ---: | --- |
| GPU wired 上限 `iogpu.wired_limit_mb` | 88,064 MiB | 开机时由 LaunchDaemon 设置 |
| 系统预留 | 2,872 MiB | 空载时 macOS 自己 wired 的部分 |
| 常驻嵌入 `embeddinggemma-2` | 3,160 MiB | 永远加载，不参与调度，直接从池里扣掉 |
| **调度池**（`bin/gpu-pool`） | **82,032 MiB** | 88,064 − 2,872 − 3,160 |
| 聊天模型 `qwen3.8-flash-next` 预约 | 69,500 MiB | 默认模型，开机预加载，空闲不卸载 |
| **Flash-Next 旁边剩余** | **12,532 MiB** | 其他入口的预约之和不超过这个数才能与它同驻 |

规则（llama-swap 容量路由，`platform/llama-swap/config.yaml`）：

- 每个入口声明 `capacities.gpu`（实测 wired 峰值 + 余量）。新请求的模型放不下时，llama-swap 卸下“损失最小”的一组模型
  （`evict_cost` 之和最小）再加载；卸载前等在途请求结束。
- `evict_cost`：Flash-Next 100、Flash-Next Uncensored 50、`hy4-flash`／ASR／TTS／`fish-drama-3`／PE 扩写器 10、其余 0（图片 `-pe` 为 1）。
  所以位置不够时先卸图片、嵌入、重排、决策，聊天模型最后才卸。
- **独占入口**（`minimax-h3-video`、`fish-drama-3`）预约等于整个池：加载时卸下所有其他入口（常驻嵌入除外）。
- `ttl`：空闲多久后自动卸下。Flash-Next 与常驻嵌入不卸；嵌入／重排／决策 1800 s；多数其他入口 600 s；PE 扩写器 300 s；视频 120 s。
  会挤掉 Flash-Next 的入口（`hy4-flash`、`mammoth-1`、`fish-drama-3`、`minimax-h3-image-pe`）都用 600 s 或更短，空出来后 Flash-Next 在下一次请求时重载。
- 预约只是调度用的账，不等于整机内存：零拷贝映射的权重还占页缓存。实测同驻时出现过压缩内存（几 GB），没有失败，见第 3 节。

## 2. 模型总表

加载方式：**常驻** = 永不卸载；**按需** = 第一次请求时加载、空闲 ttl 后卸下；**独占** = 加载时卸下其他所有入口；
**归档** = 权重在另一台机器的归档盘上，第一次请求时由 `bin/archive-restore` 拷回并校验 SHA-256 后再启动。
“同驻 FN” = 能否与已加载的 Flash-Next 同时驻留（只看预约，≤ 12,532 MiB 即可）。

| 模型 ID | 用途 | 引擎 | 权重精度 | 预约 MiB | 实测 wired 峰值 | 加载方式 | 同驻 FN |
| --- | --- | --- | --- | ---: | --- | --- | :---: |
| `qwen3.8-flash-next` **（默认）** | 聊天、视觉、工具，262K × 2 路 | Strata Metal | AD-4.27 混合（专家 IQ4_NL／IQ3_S／IQ2_S，dense Q8_0，主体约 3.57 bpw），MTP Q2_0，KV INT8 | 69,500 | +68,468–68,489 MiB（11-24） | 常驻（预加载） | — |
| `qwen3.8-flash-next-uncensored` | 同上，去审查权重 | 同上 | 同上 | 69,500 | +65,684 MiB（11-22，chunk 2048 之前） | 按需，**归档**（90.3 GiB） | 否（互斥） |
| `hy4-flash` | 备用聊天、视觉，128K，一次一个请求 | Strata Hy4 Metal | SQ-64（路由专家约 3.0 bpw）+ Engram Q8（页缓存，不 wired），KV fp16 MLA | 71,000 | +68,037 MiB（服务峰值，11-23） | 按需 | **否**（分时切换） |
| `embeddinggemma-2` | 文本嵌入（RAG） | MLX | BF16 权重，FP32 激活 | 不占池（预留 3,160） | +2,680 MiB | **常驻** | 是 |
| `embeddinggemma-2-multimodal` | 图片／音频／视频嵌入 | llama.cpp | BF16 | 3,500 | +2,996 MiB | 按需 | 是 |
| `qwen3-embedding-0.6b`／`4b`／`8b` | 文本嵌入，32K 输入 | MLX | FP16／FP16／按层类 W8A8 | 3,500／12,000／10,500 | +2,729／+11,218／9,983 MiB（32K 输入） | 按需 | 是 |
| `voyage-4-nano` | 文本嵌入 | MLX | BF16 | 3,000 | +2,263 MiB | 按需 | 是 |
| `qwen3-reranker-0.6b`／`4b`／`8b` | 重排 | MLX | 按层类 W8A8（0.6b／4b 的 down_proj 用 BF16／FP16） | 3,000／8,000／9,500 | 2,388／7,309／8,775 MiB（32K 文档） | 按需 | 是 |
| `pplx-decider-flash` | 决策（文本＋图片） | MLX 移植 | 按层类 W8A8，down_proj／视觉塔 FP16 | 10,000 | +8,783 MiB | 按需 | 是 |
| `elephant-1` | 决策，9B | MLX 移植 | 按层类 W8A8 | 10,000 | +8,730 MiB | 按需 | 是 |
| `joey-1` | 决策，2B | MLX 移植 | 按层类 W8A8 | 4,500 | +3,101 MiB | 按需 | 是 |
| `mammoth-1` | 决策，27B | MLX 移植 | 按层类 W8A8 | 28,500 | +27,458 MiB | 按需 | **否** |
| `drex-v1.5` | 决策，9B | llama.cpp（nace-ai 分支） | Q8_0，KV F16 | 12,000 | +11,028 MiB | 按需 | 是 |
| `qwen3-asr-1.7b` **（默认 ASR）** | 语音转写，词级时间戳 | mlx-audio fork | W8A16（8 位权重） | 6,500 | +4,599 MiB（含对齐器） | 按需 | 是 |
| `whisper-large-v3` | 多语种转写 | mlx-whisper fork | 解码器 W8A16，编码器 FP16 | 4,500 | +3,198 MiB | 按需 | 是 |
| `qwen3-tts`／`-clone`／`-design` | 语音合成 | mlx-audio fork | W8A16（group 32），BF16 激活 | 6,500／7,000／6,500 | +3,554／+4,876／+3,853 MiB（8 位，16 句） | 按需 | 是 |
| `fish-drama-3` | 语音合成（多说话人、语气） | mlx-audio | 路由专家 8 位，其余 BF16 | 整个池 | +41,340 MiB | **独占** | **否** |
| `minimax-h3-image` | 图片生成／编辑，单边 ≤ 4096 | h3-image-mlx fork | DiT 与文本编码器 INT8 ConvRot（W8A8），VAE FP16 | 10,000 | 约 2–6 GB（1024²–4096²），超长序列切片后 ≤ 8.3 GB | 按需 | 是 |
| `minimax-h3-image-pe` | 同上 + 官方提示词扩写 | 同上 | 同上 | 10,000 | 同上 | 按需 | 出图时是；扩写时**否** |
| `minimax-h3-image-pe-expander` | 上一项内部用的扩写模型 | mlx-vlm | NVFP4（约 4.5 bit），BF16 KV | 30,000 | +23.2 GB（单路）／+28.5 GB（8 路） | 按需 | **否** |
| `qwen-image-2.1-turbo`／`-turbo-pe`／`qwen-image-2.1` | 图片生成／编辑，LoRA、ControlNet | mflux fork | DiT W8A8，文本编码器 W8A16，VAE FP32；ControlNet 分支 W8A8 | 11,000 | 6.7–9.0 GB；带 `control_image` 最高 +9,617 MiB | 按需 | 是 |
| `z-image-turbo` | 文生图 | mflux fork | DiT W8A8（注意力＋前馈），文本编码器／VAE BF16 | 11,000 | 8.0–8.8 GiB | 按需 | 是 |
| `minimax-h3-video` | 视频＋立体声 | h3.c fork | DiT INT8（Hadamard 旋转 W8A8），文本编码器 W8A16，VAE BF16 计算 | 整个池 | 768p standard +14.3 GB；fast DiT 分配峰值 22.2 GiB（5 s）／33.0 GiB（15 s） | **独占** | **否** |

- 只有 `qwen3.8-flash-next-uncensored` 的服务权重归档在另一台机器（90.3 GiB）。按归档报告的实测拷回速度（约 157–201 MiB/s）估计要 8–10 分钟，
  且 `archive-restore` 要求拷完后本机仍剩 > 100 GiB，空间不够时返回 503 并给出手动恢复命令。其余入口的 `archive-restore` 只是在本机副本完整时直接启动。
  决策模型、重排、嵌入的归档另有一份，但本机副本仍在；Qwen-Image Turbo／ControlNet、Qwen3-TTS／ASR、Whisper 归档的是原版（BF16／FP16），线上用本机的 8 位副本。
- `:worker` 后缀（`qwen3.8-flash-next:worker`）是同一个已加载模型的别名，不另占内存，不在 `/v1/models` 里。

## 3. 能否并存

**与 Flash-Next 同驻**（剩余 12,532 MiB）：

| 组合（加上已加载的 Flash-Next） | 预约合计 MiB | 结果 |
| --- | ---: | --- |
| + `minimax-h3-image`（任意尺寸） | 79,500 | 可以（实测 4096²、2160p 与聊天交替使用，不重载） |
| + `qwen-image-2.1-turbo`（含 ControlNet） | 80,500 | 可以（`-turbo-pe` 改写时回调的就是这个 Flash-Next） |
| + `z-image-turbo` | 80,500 | 可以（实测；对话缓存装满时第二张图最多慢 21%） |
| + `qwen3-asr-1.7b` 或 `whisper-large-v3` | 76,000／74,000 | 可以 |
| + `qwen3-tts` | 76,000 | 可以（实测 1449 字合成 + 11.6 万 token 提示同时计算） |
| + `qwen3-asr-1.7b` + `whisper-large-v3` | 80,500 | 可以 |
| + `whisper-large-v3` + `qwen3-tts` | 80,500 | 可以 |
| + `qwen3-asr-1.7b` + `qwen3-tts` | 82,500 | **不行**（超 468），后加载的会卸下前一个 |
| + `qwen3-reranker-4b` + `qwen3-embedding-0.6b` | 81,000 | 可以 |
| + `qwen3-embedding-4b` 或 `drex-v1.5` | 81,500 | 可以（单独） |
| + 任一图片入口 + ASR／TTS／重排／嵌入／决策 | > 82,032 | **不行**：图片入口（10,000–11,000）旁边只剩 1,500–2,500，放不下其他按需入口 |
| + 两个图片入口 | > 82,032 | **不行**，后一个会卸下前一个 |
| + `mammoth-1` | 98,000 | **不行**：卸下 Flash-Next |
| + `minimax-h3-image-pe` 扩写 | 99,500 | **不行**：扩写时卸下 Flash-Next（之后出图入口留在内存），下一次聊天重载 Flash-Next |
| + `hy4-flash` 或 `-uncensored` | ≥ 139,000 | **不行**：两个聊天模型分时切换 |
| + `fish-drama-3` 或 `minimax-h3-video` | 整个池 | **不行**：独占，卸下所有入口 |

**Flash-Next 不在时**（池 82,032 MiB 全空）：

- `hy4-flash`（71,000）旁边剩 11,032：可以带一个图片入口（H3 10,000，Qwen-Image／Z-Image 11,000）、ASR、TTS、≤ 10,000 的决策和重排；
  `qwen3-embedding-4b`、`drex-v1.5`（12,000）放不下。
- `mammoth-1`（28,500）与其他按需入口基本都能同驻（除了两个聊天模型、PE 扩写器和独占入口）。
- 所有非独占、非聊天的按需入口之和超过池子，同时最多驻留其中一部分，按需换出。

**已实测的同驻**（其余组合按预约推算，没有逐一实测）：Flash-Next + H3 图片 4096²／2160p、+ Qwen-Image（含 2752×1536 与 11.6 万 token 提示同时计算：出图 90 s、首 token 74.4 s、压缩 10.9 GB、swap +1.05 GB）、
+ Z-Image、+ Qwen3-ASR、+ Qwen3-TTS（同时计算时 TTS 实时率 0.30、Flash-Next prefill 1,968 tok/s，压缩 6.7 GB，无 swap）。
日常交替使用（不同时计算）时这些组合没有明显减速；同时满载计算会互相拖慢。

**切换代价**（把 Flash-Next 换出再换回）：

| 切换 | 时间 | 出处 |
| --- | ---: | --- |
| Flash-Next → `hy4-flash`（首请求多出的时间） | 13.2–14.2 s | 2026-11-23 上线验证，经 :8080 |
| `hy4-flash` → Flash-Next | 10.0–10.1 s | 同上 |
| Flash-Next 冷加载（页缓存里没有）经 :8080 首 token | 9.6 s | 2026-11-23，`reports/strata.md` 预读一节 |
| Flash-Next 预读后加载 | 4.6 s | 同上（`flashnext-prefetch` 在空闲时把 55 GiB 读回页缓存） |
| 视频／Drama 3／mammoth-1 用完后回到聊天 | 约 4.6–9.6 s + 等 ttl 或卸载 | 按上两行 |

## 4. 速度

### 文本

| 模型 | 设置 | prefill tok/s | decode tok/s | 出处 |
| --- | --- | --- | --- | --- |
| `qwen3.8-flash-next` | 单路 | 2K 约 2,150；8K 约 2,850；26–32K 约 3,105；104–128K 约 3,050 | 154.0–154.4 | 11-24 `a21ef5378`；11-23 chunk 2048 |
| 同上 | 2 路同时 | 两路各约 258K 合计 2,891 | 合计 172.6–180.5 | 11-24 |
| 同上 | 4 路（2 路运行、2 路排队） | — | 合计 175.6–180.3 | 11-24 |
| 同上 | 短提示首 token | 0.144 s | — | 11-23 |
| 同上 | 108K 会话被挤出后续问（SSD 读回） | 首 token 0.72–0.85 s | — | 11-23 |
| `qwen3.8-flash-next-uncensored` | 单路／2 路 | 26K 2,762；116K 2,701 | 148.9／合计 154.3 | 11-22（旧版，未在现行版本复测） |
| `hy4-flash` | 单路，DFly2 投机解码 | 4K 1,474；32K 2,010；128K 1,951 | 短提示 102–133；32K 81；128K 68（不开投机 46–59） | 11-23 |
| `minimax-h3-image-pe-expander` | 单路 | 前缀缓存命中 0.13–0.19 s | 41.8–46.8 | 11-18、11-23 |

### 图片（单张、热态、独占，除非注明）

| 模型 | 1024² | 2048² | 其他 | 出处 |
| --- | ---: | ---: | --- | --- |
| `minimax-h3-image`（6 步） | 3.2 s（同提示词缓存命中 2.6 s） | 约 10 s | 4096² 约 60 s；编辑 1 张图 8.8 s（命中 6.3 s）、4 张 33 s（命中 24 s）；冷启动首张 11 s | 11-23 |
| `minimax-h3-image-pe` | 25.6 s（扩写 22.2 + 出图 3.3） | — | `size=auto` 17.2 s；编辑 1 张图 41.7 s；加卸下／重载 Flash-Next | 11-23 |
| `qwen-image-2.1-turbo`（8 步） | 7.8–9 s | 约 43 s | 透明 8.2 s；编辑 10.5 s；遮罩 12.3 s；冷启动首张 14 s | 11-17～11-23 |
| 同上 + `control_image` | 9.4–10.0 s | 55.4 s | 2752×1536 60.1 s；2400×1792 61.1 s | 11-24 |
| `qwen-image-2.1-turbo-pe` | 文生图 36.3 s、编辑 27.6 s（中位） | — | 含 Flash-Next 改写 18–31 s | 11-17（旧 Flash-Next 版本） |
| `qwen-image-2.1`（40 步） | 约 28 s | — | 带 ControlNet + CFG 69.8 s | 11-17、11-24 |
| `z-image-turbo`（9 步） | 5.6 s | 36.1 s | 冷启动首张 7.5 s | 11-22 |

### 视频（`minimax-h3-video`，独占）

1344×768（768p）；5 s 片段 = 124 帧，15 s = 362 帧。四档：`best`（精确）、`standard`（默认）、`turbo`（≤ 8 s）、`fast`。

| 任务 | best | standard | turbo | fast | 出处 |
| --- | ---: | ---: | ---: | ---: | --- |
| 768p 5 s 文生视频 | 约 690 s（11.5 min） | 约 283 s（4.7 min） | 约 191 s（3.2 min） | 约 162 s（2.7 min） | best 11-18；其余 11-21 |
| 768p 5 s 首帧条件 | 约 761 s | 约 318 s | 约 218 s | — | 11-18 |
| 768p 5 s 首尾帧 | — | 363 s | — | — | 11-22 |
| 768p 15 s 文生视频 | — | — | 不支持 | 758 s（12.6 min） | 早期版本（V2） |
| 832×480 15 s 文生视频 | — | 447 s | — | — | 11-17 |
| 512² 5 s 文生视频 | — | 49 s | — | — | 11-22 |
| 512² 参考视频续写（3 s 上下文 → 8 s） | — | 172 s | — | — | 11-22 |

之后 best／standard／turbo 去噪又快 1.4–1.8%（旋转 W8A8，11-22／23），表中未重测整次。渲染时另一个渲染请求返回 429。

### 语音

| 模型 | 速度 | 出处 |
| --- | --- | --- |
| `qwen3-asr-1.7b` | 11.3 分钟英文（100 条）11.0 s（约 60 倍实时）；15.8 分钟长英文 4.26 s（约 220 倍）；12 分钟长中文 2.44 s（约 290 倍）；decode 311–322 tok/s | 11-22，进程内 |
| `whisper-large-v3` | 11.3 分钟英文 19.1 s（约 35 倍实时）；decode 230 tok/s | 11-18，进程内 |
| `qwen3-tts`／`-clone`／`-design` | 首段音频 0.08–0.14 s；实时率 0.17／0.18／0.17（约 5.5–6 倍实时） | 11-23，8 位，进程内 |
| `fish-drama-3` | 首段 0.17–0.20 s；实时率 0.37（约 2.6 倍实时）；加载约 2 s | 11-23 |

### 嵌入、重排、决策

| 模型 | 速度 | 出处 |
| --- | --- | --- |
| `embeddinggemma-2` | 16 token 查询 5.6 ms；批量约 58,000 tok/s | 性能报告总览 |
| `embeddinggemma-2-multimodal` | 图片 0.04–0.12 s；5 s 音频 0.035 s；4 s 视频 0.34 s | 性能报告总览 |
| `qwen3-embedding-0.6b`／`4b`／`8b`、`voyage-4-nano` | 64 × 450 token：0.39／2.50 s、voyage 0.24 s；8b 短查询 29 ms | 性能报告总览 |
| `qwen3-reranker-0.6b`／`4b`／`8b` | 20 篇约 0.1／0.6／1.1 s | 性能报告总览 |
| `pplx-decider-flash` | 短请求 23–47 ms；8K／32.6K token 0.75／4.47 s | 11-23 |
| `elephant-1`／`mammoth-1` | 16.1K token 2.52／7.81 s | 11-23 |
| `joey-1`／`drex-v1.5` | 16K token 0.95 s／12.8K token 4.3 s | 11-23／11-16 |

## 5. 缺少实测的项

- `qwen3.8-flash-next-uncensored` 在现行 Strata（并发 MTP、chunk 2048、QSA 修复）上的速度和 wired 峰值：表中是 11-22 旧版数字。
- `hy4-flash` 从冷 SSD 加载（权重不在页缓存时）的时间；它按设计一次只跑一个请求，没有并发数字。
- 视频：768p 15 s 的 `best`／`standard`、首尾帧的 `turbo`／`fast`；15 s `fast` 是早期版本的数字，之后的 VAE 和 DiT 优化后没有重测整次。
- 图片：H3 4096² 与 Flash-Next 同驻的耗时只有优化前的约 120 s（11-16）；`qwen-image-2.1-turbo-pe` 的改写时间是旧版 Flash-Next 测的；`qwen-image-2.1` 2048²、`z-image-turbo` 多张（`n` > 1）没有测。
- 语音：TTS 8 位只有进程内实时率，没有经 HTTP 的端到端和长文本数字；Whisper 8 位没有长音频数字。
- 并存：第 3 节多数组合是按预约推算的，只有列出的几组实测过；多入口同时满载计算的速度没有系统测量。
