# 运维：回归测试、健康检查、内存守护

| 工具 | 作用 | 怎么用 |
| --- | --- | --- |
| `tools/regress` | 每个入口一条固定请求（温度 0／固定 seed），与黄金输出逐项比较 | `.venv/bin/python tools/regress --record`（首次录制）；`--quick`（每次改动后）；不带参数为完整一套（升级 MLX／环境／引擎后） |
| `tools/health-check` | 一轮健康检查：网关 `/health` 与 `/v1/models`、常驻嵌入一次请求、llama-swap 与 Caddy 进程、磁盘、内存、重启／panic | LaunchAgent `app.llm-serving.health` 每 10 分钟跑一次（`tools/install.py` 安装） |
| `tools/memguard` | 包住一个命令，进程树 RSS 超限或系统可用内存过低时杀掉整个进程组 | `tools/memguard --max-gb 70 -- <command>`（测速、转换这类大内存任务） |

- **比较规则**（`tools/regress`）：对话温度 0 文本逐字节；嵌入向量相同为 PASS、最小余弦 ≥ 0.9999 为 WARN；重排同序且最大差 ≤ 0.01 为 WARN；
  决策答案相同且最大 |Δp| ≤ 0.01 为 WARN；转写逐字；TTS 字节相同或时长 ±10% 且回写相同；图片 PSNR ≥ 40 dB、视频 ffmpeg PSNR ≥ 40 dB 为 WARN；
  其余 FAIL。`/v1/models` 里没有的入口跳过。有 FAIL 时退出码 1。
- **黄金输出不在仓库里**：每台机器第一次用 `--record` 在自己的部署上录制，存在 `<状态目录>/golden/`（或 `LLM_GOLDEN_DIR`）。
  不同机器、不同 macOS／MLX 版本的输出可能有舍入级差别，所以不共用。ASR 用例需要两段音频放在 `<golden>/inputs/`：
  `en/1089-134691-0015.wav`（LibriSpeech test-clean）和 `zh/aishell_00050.wav`（AISHELL-1 test），16 kHz 单声道 WAV。
- **健康检查只写日志**：每轮一行 JSON 追加到 `<状态目录>/logs/health/health.jsonl`（按天轮转，保留 30 天），`logs/health/STATUS` 是最近一轮摘要
  （第一行 `OK|WARN|FAIL <时间>`）。不加载会卸下聊天模型的模型，只用标准库。门限：磁盘 < 60 GiB WARN、< 20 GiB FAIL；
  `kern.memorystatus_level` < 15 WARN、< 8 FAIL。
- **可选的多 agent GPU 锁**：设置 `GPU_LOCK`（锁工具路径）后 `tools/regress` 会先占满整个 GPU 池再跑；设置 `GPU_LOCK_HOLDERS` 后健康检查会报告占锁过久。
  锁工具本身不在本仓库。
