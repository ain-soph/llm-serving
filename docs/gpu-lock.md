# GPU 锁（gpu-lock）使用规则

几个 agent 同时在一台 Mac 上工作时，用 `~/llm-serving/bin/gpu-lock` 按显存预约共享 GPU。脚本本身在
[`serving/bin/gpu-lock`](../serving/bin/gpu-lock)，由 `tools/install_serving.py` 装到 `~/llm-serving/bin/`（见
[serving-restore.md](serving-restore.md)）。本页是部署目录 `AGENTS.md` 中规则 1（gpu-lock v2）的公开版；规则正文以
生产 `AGENTS.md` 为准，两边改动要同步。

## 规则 1：GPU 锁 = 显存预约

**什么时候需要**：加载约 2 GB 以上的模型（直接加载或经 llama-swap）、跑 benchmark、暂停/恢复/重启 llama-swap、应用
llama-swap 配置改动。调研、下载、编译、写文件和只用 CPU 的转换不需要。多个 agent 可以**同时**持有份额，只要预约总和
不超过池大小（`bin/gpu-pool` 计算：`iogpu.wired_limit_mb` 减去系统基线和常驻模型）。

- **加入**：`gpu-lock acquire NAME MB`（每次调用最多等约 9 分钟）或 `gpu-lock try NAME MB`（exit 0 = 拿到份额，1 = 仍在排队）。
  **MB = 本次会话要加载的全部内容实测的 GPU wired 峰值 + 余量**（与 llama-swap `capacities.gpu` 同一规则；例如小 ASR
  模型 3000–6000，Flash-Next 67000–76000）。不知道就先用独占份额测一次。不得加载超过所申报的量；用新 MB 再次
  `acquire NAME MB` 可调整：缩小总能成功，扩大只能用空闲额度。
- **不写 MB（或 `all`）= 独占**整个池。**凡是要报告的速度测量都必须独占**（decode/prefill tok/s、吞吐、延迟、TTFT、ASR
  实时率、图像/视频渲染时间）：共享 GPU 和内存带宽的邻居会扭曲结果。持有较小份额时先扩到整个池
  （`gpu-lock acquire NAME all`）。`lsw-apply` 也要求独占。质量评测、功能检查可以共享；共享时测得的速度不得写进报告。
- **排队**：先到先得；后来的请求若放得下可以越过放不下的较早请求，直到后者等满 20 分钟。**轮到你（排第一且放得下）时，
  5 分钟内取走**，否则后面放得下的请求先走；你保留票据，之后再次轮询时重新轮到。至少每 15 分钟轮询一次，否则失去位置
  （由后台会话脚本里的 acquire 循环来做，见下面“等待”）。先做完全部准备。
- **只在真正使用 GPU 时持有份额**：GPU 工作开始前一刻再 acquire；分析结果、写报告、下载或等待时不得持有；GPU 工作一结束
  立即 release。会话脚本必须在退出 trap 里释放（正常结束、出错或中断都要释放）。
- **等待：一条后台命令，不轮询**（用户 2026-11-21）：agent 每次唤醒都要重读整个上下文（H3 subagent 曾每 3 分钟醒一次，
  每次 400–470k token）。每个 GPU 阶段写成一条后台命令（如 Claude Code `run_in_background` Bash）：exit 1 时重复
  `gpu-lock acquire NAME all|MB`（exit 4 停下）；在脚本**顶层**设 `trap '…/gpu-lock release NAME' EXIT INT TERM`（不要放在
  zsh 函数里：那里的 EXIT trap 提前触发，立刻放了锁）；跑 memguard 包住的任务（长任务在后台每 ≤20 分钟 heartbeat）；结果写进
  文件。然后结束回合，等退出通知唤醒。可选兜底心跳 ≥30 分钟（如后台 `sleep 1800`，查一次再重设），不用更短的 sleep 或轮询；
  过了预计结束时间每约 30 分钟查一次。有未完成的工作时，结束回合前必须有会唤醒你的东西；编排者也不主动查。长任务在每个
  阶段边界把 checkpoint（状态、数字、commit、下一步）写进运行目录；subagent 上下文约 300k token 以上且要进入长等待或新阶段
  时，写好 checkpoint 再汇报，由编排者另起新 agent 接着做。
- 把 GPU 工作集中在一个会话里（最长约 90 分钟；每 ≤20 分钟 `gpu-lock heartbeat NAME`；60 分钟没有心跳的份额会被丢弃），
  安排测试顺序以减少模型重复加载。
- 第一个持有者进入时脚本暂停 llama-swap（其模型不属于任何人的预约），最后一个持有者离开且无人排队时恢复（不预加载）。
  `gpu-lock release NAME` 归还份额；`gpu-lock cancel NAME` 撤销不再需要的排队票据（先停掉还在等待的 `acquire` 循环，
  否则它会再次排队；不要手工删除 `.locks/queue` 里的文件）。需要第二个会话就重新排队。暂停会一次卸载所有模型并切断
  仍在流式输出的回答，所以第一个持有者最多等待 120 s（`GPU_LOCK_DRAIN_S`），让来自本机之外的用户请求
  （由 `bin/lsw-inflight` 统计）结束；等待期间 acquire/try 报告 "waiting for N user request(s)"。本机发出的请求（agent、
  regress）不等待。`lsw-apply` 重启后 llama-swap 处于未暂停状态，会话期间用户请求会恢复，可能被你下一次重启或测试引起的
  模型换出打断。
- **每次 try/acquire 都做死锁检查**：永远无法满足的请求（持有者互相等待对方扩容，或脚本在持有一个份额的同时申请第二个、
  两者加起来放不下）以 **exit 4** 拒绝且不排队。exit 4 不要循环重试：先释放或缩小份额。只对 exit 1 重试。stderr 上的
  警告表示你的 agent 进程还持有别的份额且这次请求现在放不下：不要一边持有那个份额一边在前台等待。排队前自己确认你持有
  或等待的东西不依赖这个请求。
- `gpu-lock status` 显示持有者及其 MB、空闲 MB 和队列。共享运行时在 benchmark 日志里记下同时持有者。

## 相关规则

- 规则 2：只有持有份额时才能为直接测试（在 llama-swap 之外起服务）腾内存。脚本已替你暂停 llama-swap，再暂停一次无害；
  别人持有份额时不要自己 resume llama-swap。自测服务用 18100–18199 端口，释放份额前先停掉。
- 大内存任务用 `tools/memguard` 包一层（见 [operations.md](operations.md)）：gpu-lock 只记账，不限制实际占用。
