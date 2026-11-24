# 运维：回归测试、健康检查、内存守护

| 工具 | 作用 | 怎么用 |
| --- | --- | --- |
| `bin/regress` | 每个入口一条固定请求（温度 0／固定 seed），与黄金输出逐项比较 | `bin/regress --record`（首次录制；自动使用仓库的 `.venv`）；`--quick`（每次改动后）；不带参数为完整一套（升级 MLX／环境／引擎后） |
| `bin/health-check` | 一轮健康检查：网关 `/health` 与 `/v1/models`、常驻嵌入一次请求、llama-swap 与 Caddy 进程、磁盘、内存、重启／panic | LaunchAgent `com.local.health` 每 10 分钟跑一次（`tools/install_serving.py` 从 `platform/launchd/` 渲染） |
| `bin/flashnext-prefetch` | Flash-Next 没加载时把它启动要读的权重（56 GB 专家 GGUF、MTP、mmproj，共 55.3 GiB）预读进页缓存，下次加载走热路径（直连 /health 8.5 → 3.4 s，经 :8080 首 token 9.6 → 4.6 s）。只在没人持有/排队 gpu-lock、llama-swap 里没有 Flash-Next、没有归档传输、空闲内存 ≥ 缺的字节 + 12 GiB 时读；后台低优先级 IO，读的过程中条件一变就停 | LaunchAgent `com.local.flashnext-prefetch` 每 15 秒检查一次；`--status` 看缓存和判断；关掉：`touch .local/flashnext-prefetch.off`（或 `FLASHNEXT_PREFETCH=0`）；日志 `logs/prefetch/` |
| `bin/memguard` | 包住一个命令，进程树 RSS 超限或系统可用内存过低时杀掉整个进程组 | `bin/memguard --max-gb 70 -- <command>`（测速、转换这类大内存任务） |

- **比较规则**（`bin/regress`）：对话温度 0 文本逐字节；嵌入向量相同为 PASS、最小余弦 ≥ 0.9999 为 WARN；重排同序且最大差 ≤ 0.01 为 WARN；
  决策答案相同且最大 |Δp| ≤ 0.01 为 WARN；转写逐字；TTS 字节相同或时长 ±10% 且回写相同；图片 PSNR ≥ 40 dB、视频 ffmpeg PSNR ≥ 40 dB 为 WARN；
  其余 FAIL。`/v1/models` 里没有的入口跳过。有 FAIL 时退出码 1。
- **黄金输出不在仓库里**：每台机器第一次用 `--record` 在自己的部署上录制，存在 `evals/golden/`（或 `LLM_GOLDEN_DIR`；本机私有，不进库）。
  不同机器、不同 macOS／MLX 版本的输出可能有舍入级差别，所以不共用。ASR 用例需要两段音频放在 `<golden>/inputs/`：
  `en/1089-134691-0015.wav`（LibriSpeech test-clean）和 `zh/aishell_00050.wav`（AISHELL-1 test），16 kHz 单声道 WAV。
- **健康检查只写日志**：每轮一行 JSON 追加到 `logs/health/health.jsonl`（按天轮转，保留 30 天），`logs/health/STATUS` 是最近一轮摘要
  （第一行 `OK|WARN|FAIL <时间>`）。不加载会卸下聊天模型的模型，只用标准库。门限：磁盘 < 60 GiB WARN、< 20 GiB FAIL；
  `kern.memorystatus_level` < 15 WARN、< 8 FAIL。
- **多 agent GPU 锁**：`bin/regress` 先用 `bin/gpu-lock` 占满整个 GPU 池再跑（`--no-lock` 跳过，调用方已持锁时用）；健康检查报告占锁过久的份额（`.locks/holders`）。
  规则见 [gpu-lock.md](gpu-lock.md)。已归档到另一台机器、本机缺失的入口记为 `SKIP(archived)`，不发请求（不会触发 `bin/archive-restore` 拷回权重）。
