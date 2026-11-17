# 模型实测、修改收益与资源口径

README 的数据摘自 2026-11-15～16 在 Apple M5 Ultra、96 GiB 统一内存、macOS 27 上进行的测试。性能测试按当时记录独占 GPU；4096² 图片与 Flash-Next 共存的测试单独注明。这里是经过筛选的数值摘要，不包含原始请求、音频、图片、视频、用户路径或服务日志。

## 如何解读

- **不是迁移环境的新一轮测速**：原测试分别使用各组件当时的环境，Strata/ASR 为 Python 3.12，图片与视频包装器为 Python 3.15。统一部署环境的依赖验证不会把这些数据变成新 Python 版本的性能验收。
- **GB = 10⁹ bytes；MiB = 2²⁰ bytes；GiB = 2³⁰ bytes**。路由配置的容量数值是调度预留，不作为实测内存。
- wired 是系统锁定页相对空闲基线的增量；RSS 是进程驻留页；footprint 是进程物理 footprint；GPU peak 是引擎记录的 Metal 分配峰值。这些指标互不等同，也不能相加作为一份“总内存”。
- mmap 权重可能占用大量可回收文件页缓存。尤其是 Strata 的专家/PLE 与图片的 DiT/文本编码器，不计入某个 GPU 分配峰值不代表它们不消耗统一内存。
- 时间通常不含首次下载和准备权重；图片阶段计时包含启动相应子进程，Strata 热 shader 数据排除首次编译。单次结果不提供方差或置信区间。
- 增速 = 新吞吐/旧吞吐 − 1；耗时变化 = 新耗时/旧耗时 − 1；加速倍数 = 旧耗时/新耗时。没有同口径原版基线的指标不计算收益。

## 测试条件与证据索引

下列 ID 对应本地历史报告及结果文件的公开数值摘录。原始目录不上传；记录日期、负载、数值和适用范围供阅读者判断。README 的原始字节数另见 [model-storage.json](model-storage.json)。

| ID | 日期／原始记录 | 条件与摘录范围 |
| --- | --- | --- |
| S1 | 11-16，Strata prefill 分块 A/B 报告 | AD-4.27、262144 ctx × 2、MTP Q2_0 spec=3、KV INT8；旧 auto prefill 26,106/115,855 token 为 1854/1999 tok/s |
| S2 | 11-16，`strata-fork-auto-ab-1116-1748` | 同规格，上游手动 4096/1024 与 fork auto；26,106/115,856 输入；最终 fork 热 shader 2013.8/2225.0 tok/s，decode c1/c2 每请求 117.2/74.4 tok/s；两臂 wired 69868 MiB |
| S3 | 11-15，`flashnext-final-1115-1804` | Uncensored、262144 × 2；decode 118–139 tok/s、wired 68771 MiB，属于 fork 切换前同规格历史数据 |
| S4 | 11-15，`compare-flashnext-1115-1651` | Strata AD-4.27 128k × 4、MTP、KV INT8；llama.cpp 262k × 8、无 MTP、KV F16。只用于说明更换后端的整套配置差异，不用于估计 fork 修改的净收益 |
| E | 11-15，EmbeddingGemma Metal 部署记录 | 文本 4 槽 × 8192；4 篇共 22036 token。多模态分别测试图片、5 s 音频、4 s 视频；文本与多模态不是两个独立模型权重集 |
| R | 11-16，rerank 基准报告；统一环境复测另见 [rerank-validation.json](rerank-validation.json) | 1 查询 × 20 文档，每篇约 400 token；旧 llama.cpp 为 1 槽、8192 ctx/batch/ubatch，新 MLX 为 8192 补齐后 batch token、256 MiB 缓存；无生成 decode |
| A | 11-15，ASR 完整评测及 cache512 诊断 | 英文 100 条 c1/c4；长英文 946.9 s、长中文 722.6 s；不含模型初次加载，包含 HTTP/音频预处理；c4 是批次墙钟而非单条延迟 |
| I | 11-16，`textab0.jsonl` / `textab1.jsonl` 第二组样本 | 相同提示、seed=42、6 步、预算 8000、H3_ASYNC=1；开关零拷贝。1024²：13.109/4.190 s；2048²：16.038/12.181 s |
| I4 | 11-16，`image-alternate-test-1116-1726` | 4096² 生图时 Flash-Next 已加载但空闲，约 120 s；它是共存场景，不能与独占高缓存预算约 71–74 s 直接计算优化比 |
| V1 | 11-16，`s768/base.log` / `s768/trC.log` 优化记录 | 1344×768、124 帧，约 5.17 s；45 层/reuse 2。412→170 s；DiT 325.112→128.630 s，VAE 79.013→33.588 s。fast 加稀疏和 token 削减，输出可改变 |
| V2 | 11-16，`p2-15s/trC.log` 优化记录 | 1344×768、362 帧，约 15.08 s；fast 总计约 758 s，DiT 651.682 s，VAE 97.801 s；DiT GPU peak 33.004 GiB。不是 5 s 对照样本的简单放大 |
| VP | 11-16，`int8-pack-1116-0910` | 512×288、25 帧、seed=7、20 步、45 层/reuse 2；VAE FP32、稀疏关闭。新二进制 BF16 读入即时量化为 28.92 s，预打包 resident 17.95 s、stream 19.82 s |

S1 与 S2 属于不同会话，所以 8.6%/11.3% 是历史基线对比，不是同时交错的因果估计。S2 中上游手动调优版与 fork 的 116k prefill、decode、内存基本持平；修复解决的是上游 auto 路径没有自动使用该调优行为。

S2 第一遍 26k 输入曾测得 1230.5 tok/s，包含约 7.3 s Metal kernel 编译；README 使用热 shader 的 2013.8 tok/s。后续 28123 token 的 2269.5 tok/s 因输入不同，不拿来替换同输入对照。

## 权重、KV 与内存明细

| 模型／配置 | 权重精度 | KV／运行状态 | 速度计量补充 | 内存适用范围 |
| --- | --- | --- | --- | --- |
| 两个 Flash-Next | AD-4.27 混合量化：专家 IQ4_NL/IQ2_S/IQ3_S，dense Q8_0；MTP Q2_0 | KV INT8；262144 × 2 槽 | decode 每请求 tok/s；prefill 为实际 prompt token/TTFT | S2 默认版 wired 69868 MiB；S3 Uncensored 68771 MiB。未给 Uncensored 套用默认版最终 fork 的速度或内存 |
| EmbeddingGemma 文本／多模态 | BF16 GGUF，投影器 BF16 | 无生成 KV；llama.cpp 上下文缓冲默认 F16 | 文本 22036 token / 0.93–1.04 s；无 decode | 原记录写 MB，采样代码实际除以 2²⁰，统一标为 MiB：文本1664、多模态用后2996 |
| Qwen3 Reranker 三档（当前） | 0.6B/4B BF16；8B 为 Q8_0 等价 MLX affine 8-bit，group32，FP16 scale/bias | BF16/BF16/FP16 激活；无生成 KV 复用、无 decode；8192 batch token 是批量预算 | 历史约61000/9500/4400 输入 tok/s（按400 token/篇估算）；复测实际9950 token，0.137/0.847/1.793 s，72466/11753/5548 tok/s | 本轮 RSS 1.657/8.453/9.680 GB、wired +2772/8694/9869 MiB；采样见 [验证记录](rerank-validation.json)，不把历史实验的 MLX 分配峰值当 RSS |
| Qwen3 Reranker 三档（已退休基线） | llama.cpp Q8_0 | 上下文缓冲 F16，8192 ctx；无生成 decode | 历史约22000/4900/3100 输入 tok/s；0.36/1.64/2.56 s | 历史 RSS 1.6/5.4/9.0 GB；wired 2.2、3.1–6.2、5.0–6.6 GB，后两者为测试记录范围 |
| Nemotron 3.5 Transcribe | decoder AD-IQ4_NL + encoder F16 | 默认实现推断 KV F16/recurrent F32，尚未运行探针确认 | 自回归 prefill/decode tok/s 待测；en100 c1 30.03 s、c4 22.66 s | RSS19318/wired22284 MiB；4 槽，每槽18432 KV cells，总73728 |
| Whisper large-v3 MLX | FP16 | KV／内部状态未完成 profiler 确认 | 自回归 prefill/decode tok/s 待测；原 en100 c1 28.80 s；cache512 的 c4 26.20 s | cache512 复测 RSS3509、top MEM4807、wired4545 MiB。不能把限制前的5804 MiB当作最终占用 |
| Qwen3-ASR + 对齐器 | 文件 BF16、主模型运行FP16；对齐器FP16 | 主模型实测KV FP16，对齐器无生成KV | phase：英文前20条、122.2 s音频，c1，prefill1892 tokens、decode441步；7776/179.8 tok/s，不含对齐器 | cache512 wired6776 MiB。请求期RSS1053 MiB漏掉加载峰值，不用它表示总模型大小；另一轮含加载/预热的MLX peak为8.16 GB |
| H3 Image | DiT与文本编码器INT8 ConvRot，VAE FP16 | 无生成KV；DiT6步 | 使用端到端秒数及阶段秒数，prefill/decode tok/s不适用于扩散主干 | I中DiT footprint：1024² 2.88→1.95 GB，2048² 4.70→3.29 GB；文件页缓存另计 |
| H3 Video | 磁盘DiT INT8+F32 scales/AdaLN表，文本编码器BF16，VAE文件F32 | 服务默认BF16 VAE计算；无生成KV | 文本编码器非自回归；不把DiT视频token或denoise吞吐当LLM decode tok/s | V1/V2为阶段GPU分配峰值，不是wired或RSS；最终完整打包版各分辨率的进程峰值尚待重测 |

### 图片零拷贝的代价

I 的同一配对数据中，1024² DiT wired 从1581增至2681 MiB，尽管进程 footprint从2.88降到1.95 GB。2048² wired从3457降到2892 MiB。两者不能合并成一个笼统的“内存减少百分比”。

文本编码阶段第二组样本为2.619→0.526 s（1024²）与2.621→0.522 s（2048²），约5倍；完整流程的收益已经体现在README的13.109→4.190与16.038→12.181中，不能再乘一次。原先分阶段96 MB的缓存限制放宽属于此前另一项改动，I对照的两臂都固定预算8000，不把它再次计入零拷贝收益。

I4中4096²与空闲Flash-Next共存后，后续第一次聊天多约2–3 s，主要是文件页替换；独占出图可约74 s，但重新加载聊天模型约26 s。当前部署选择保留共存路径，不用已退休的高缓存独占目标作为当前默认速度。

### 视频精度和内存

V1的DiT分配峰值22.191→22.196 GiB基本不变；VAE阶段9.554→5.139 GiB降低46.2%，不能据此声称全进程峰值也下降46.2%。V2的DiT峰值33.004 GiB表明视频长度也影响显存。

VP是较早的打包验证会话。其BF16/packed/stream进程RSS分别12.787/30.352/15.223 GB，footprint分别18.357/22.688/10.801 GB。预打包resident的小样本footprint反而增加约23.6%；打包主要明确改善磁盘体积与加载时间。它不是后来启用BF16 VAE及稀疏注意力的最终fast服务全流程复测。

视频5秒样本约412→170 s的2.42倍来自多个改动叠加，不能与VAE阶段2.35倍再相乘。15秒优化报告记录2727→758 s（约3.60倍），但旧基线与新配置不在同一个配对会话，README因此只列新值，不把该比率列为严格A/B结果。

## 正确性与尚缺基线

| 类别 | 已有检查 | 尚不能据此声称 |
| --- | --- | --- |
| Strata修复 | 38k prompt、temperature=0，上游手动调优与fork回答逐字节一致；API对话/流式/工具/视觉通过 | 所有prompt必然相同，或相对所有后端都有固定加速倍数 |
| 图片零拷贝 | 多尺寸同seed的PNG逐字节一致 | 页对齐/缓存调整降低了权重量化精度，或完全消除了权重页缓存 |
| 视频打包 | 无条件512×288样本与旧路径逐字节一致；首帧/参考条件差异在旧实现自身重复运行波动范围内 | 所有条件生成可逐字节复现 |
| 视频BF16 VAE | 仅BF16线性层方案对FP32的PSNR51.7 dB；完整BF16线性层+注意力方案未补同口径PSNR | 完整方案仍有51.7 dB，或属于无舍入变化的精确优化 |
| 视频fast | 抽帧未见伪影；存在构图变化 | 稀疏和token削减无质量代价 |
| 嵌入 | 文本BF16 GGUF对官方float32 CPU输出余弦1.00000，6/6检索排序一致；多模态因预处理存在差异 | 纯文本GGUF与含所有模态的官方包是同一磁盘口径，或已测官方Metal速度 |
| 重排 | 当前 MLX 相对官方 BF16 最大分数差约0.001/0.010/0.004，三个公开查询排序一致；旧4B/8B GGUF为0.025/0.007 | 后端切换、批处理和精度共同改变，不能把整体提速当作某个补丁的净收益；也不能说量化必然更快 |
| ASR缓存限制 | 限制前后复测错误率相同，c4墙钟基本不变 | 已证明所有音频和格式准确率完全相同 |

Nemotron长中文自动模式曾输出英文翻译，不能把低英文WER推断成可靠中文转写。Whisper.cpp与MLX的历史解码设置未完全对齐；更换后端的时间/准确率差异不作为我们的同后端修改收益。ASR各格式及路由的历史验收范围不完全一致，静态配置和模型列表不替代请求级验收。

## 磁盘：部署规模与现有目录

2026-11-16重新读取当前文件大小，排除`.cache`，按`(device,inode)`去重硬链接。`model-storage.json`还记录`st_blocks × 512`的分配字节；它不是APFS共享extent的独占物理大小，不能用于精确求整个卷可回收空间。

| 文件组 | 运行需要／逻辑大小 | 当前相关目录逻辑总量 | 差异说明 |
| --- | ---: | ---: | --- |
| 默认Flash-Next | 96.944 GB | 96.944 GB | Metal主体56.133、PLE38.411、pack约1.496、视觉投影约0.904 GB |
| Uncensored Flash-Next | 96.947 GB | 96.947 GB | 独立权重；不能假设与默认版本共享磁盘extent |
| 两者共用的MTP | 0.889 GB | 10.556 GB | 运行只需Q2_0 GGUF及draft词表；目录还有Q4/Q8及转换张量，不计入部署最低需求；本轮未删除 |
| EmbeddingGemma全部模态 | 1.540 GB | 1.540 GB | 文本0.558与投影0.982 GB；纯文本条目不重复相加 |
| 三个MLX重排器合计 | 18.487 GB | 18.487 GB | 0.6B/4B BF16、8B Q8_0 等价格式；旧116.349 GB含评测副本，已由模型维护任务清理；本轮仅重新统计 |
| Nemotron含编码器 | 21.349 GB | 21.349 GB | decoder19.679、encoder1.670 GB |
| Whisper MLX | 3.084 GB | 3.084 GB | FP16权重与配置 |
| Qwen ASR + 对齐器 | 6.543 GB | 6.543 GB | 主模型4.703与对齐器1.840 GB |
| H3 Image | 49.320 GB | 49.320 GB | 文本27.157、DiT20.988、VAE1.169 GB；只保留页对齐版本 |
| H3 Video，FL2VA + Ref2VA | 125.622 GB | 125.622 GB | 共享文本编码器/VAEs，只为第二个检查点增加约23.937 GB；两个服务模式不额外复制权重 |

视频历史原始DiT约66.280 GB/检查点，当前每个INT8目录23.914 GB（22.27 GiB），约省42.366 GB、63.9%。这只是DiT，不能把整个视频模型目录也标为“省63.9%”。旧报告中的“62→22 GB”混用了近似GiB，这里统一更正。历史文件已移除，基线66.280 GB来自当时逐张量报告，精确当前值来自本轮stat统计。

所有源码、可执行文件、Homebrew库、uv缓存、venv及生成媒体都未算入权重表，因此README数字是模型部署规模，而不是整台机器或整个项目的磁盘总占用。
