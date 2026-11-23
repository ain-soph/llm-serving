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
- **预计用时**：提交时可加 `--eta-min MINUTES`，填写本次 GPU 会话预计占用的整数分钟数。例如
  `gpu-lock acquire eval 8000 --eta-min 25`。拿到份额后才开始倒计时。可用
  `gpu-lock heartbeat eval --eta-min 10` 更新为从现在起预计还要 10 分钟。预计用时供 agent 安排工作参考，调度继续按显存和队列规则执行。
- **不写 MB（或 `all`）= 独占**整个池。**凡是要报告的速度测量都必须独占**（decode/prefill tok/s、吞吐、延迟、TTFT、ASR
  实时率、图像/视频渲染时间）：共享 GPU 和内存带宽的邻居会扭曲结果。持有较小份额时先扩到整个池
  （`gpu-lock acquire NAME all`）。`lsw-apply` 也要求独占。质量评测、功能检查可以共享；共享时测得的速度不得写进报告。
- **优先级与回填**。提交时用 `--priority low|normal|high`，默认 `normal`。每累计 30 分钟升一档，最高到 `high`。
  先按有效优先级，再按原始入队顺序。任务靠等待升档并获得份额后，从排在它后面的票据扣掉它已用掉的等待加成，最低扣到零。
  原始入队时间和回填保护时钟照常走。前面的请求显存不足时，后面的任务可以回填，直到前面的请求等超过 20 分钟。
  轮到后有 5 分钟来取走份额，过时让出这轮。旧式 `acquire` 或 `try` 票据超过 15 分钟未轮询会被清理。
- **单片时限与轮转**。新 GPU 任务使用 `gpu-lock run NAME [MB|all] [options] -- COMMAND`。默认每片最多 30 分钟，
  可用 `--lease-min` 设置为 1 到 60 分钟。运行器按预约显存监测进程组，临近时限发 TERM，到期仍运行则发 KILL，
  等进程组退出才释放份额。让工作留在托管进程组内，命令可读取 `GPU_LOCK_DEADLINE_EPOCH_S` 提前保存进度。
  `--slices N` 提前提交 N 片，同一任务一次只参与一个排队位置。每片运行相同的命令，它可用 `GPU_LOCK_SLICE` 和
  `GPU_LOCK_SLICES` 取到片号和总片数。一片完成后，下一片重新入队。有其它任务能运行时先让它拿到一轮。
  用 `--from-slice K` 从检查点恢复。超时退出码为 124。直接 `acquire` / `try` 仍由调用方管理进程和释放锁，
  硬时限由 `run` 执行。
- **只在真正使用 GPU 时持有份额**：GPU 工作开始前一刻再 acquire；分析结果、写报告、下载或等待时不得持有；GPU 工作一结束
  立即 release。会话脚本必须在退出 trap 里释放（正常结束、出错或中断都要释放）。
- **等待：一条后台命令，不轮询**（用户 2026-11-21）：agent 每次唤醒都要重读整个上下文（H3 subagent 曾每 3 分钟醒一次，
  每次 400–470k token）。在一条后台命令中提交 `gpu-lock run`，由它负责轮询和释放。使用旧式调用的脚本应在同一条后台命令中
  在 exit 1 时重复
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

## 查看占用：gpu-top

`~/llm-serving/bin/gpu-top`（源码 [`serving/bin/gpu-top`](../serving/bin/gpu-top)，同样由 `install_serving.py` 安装）像
`top` / `nvidia-smi` 一样显示池子、持有者、队列、llama-swap 和系统实测内存。**严格只读**：不调用 gpu-lock、不写
`.locks`、不拿 gpu-lock 的状态互斥锁，也就不能代替 acquire/release，等待时也不要用它轮询。

```sh
gpu-top            # 全屏，每 2 s 刷新（-d 秒数可改）；q 退出，空格立即刷新
gpu-top --once     # 打印一次后退出（stdout 不是终端时自动如此）
gpu-top --json     # 一次快照的 JSON，给 agent 用；--no-swap 不问 llama-swap
```

- 顶部：池总量 / 已预约 / 空闲（MB 与 GB）。超过 60 分钟没心跳的份额单列（gpu-lock 下次调用时丢弃，不计入）。
- **持有者**：名字、MB（整池显示 `all`）、已持有时长（HELD）、距上次心跳（BEAT）、拿份额的 shell pid、是否还活着
  （`yes`；`agent` = 那个 shell 已退出但 agent 进程还在；`DEAD` = 都已退出，份额漏了，需 release 或等它过期）、agent 进程、
  命令行（截断）。ETA 显示预计还要占用多久（`+` 表示超过了申报的预计时间，`-` 表示未申报）。
- **队列**：顺序、名字、请求量（持有者扩容显示 `旧->新`）、已等待、距上次轮询、正在等待的 `gpu-lock acquire|try` 进程 pid，
  以及按 gpu-lock 同一套规则（先到先得、插队、20 分钟后禁止插队、5 分钟轮次、死锁拒绝）在当前快照上推算的状态：
  `ready`（下次轮询就会拿到，前提是下面的内存检查通过）、`waiting`（附原因）、`deadlock`（下次轮询会被 exit 4 拒绝）、
  `stale`（超过 15 分钟没轮询，下次调用被清掉）。这是推算，不是 gpu-lock 的决定。
  EST RUN 显示任务申报的 GPU 占用时长。显存不足的较早请求会显示回填保护倒计时或已生效。保护触发后，
  `gpu-lock acquire` / `try` 会显示挡住回填的请求和它还差的显存。
  PRI 是有效优先级，BOOST 是离下次升档的时长。托管任务也会显示片号和持锁时限。
- JSON 中，持有者的 `estimated_release_epoch_s` 和 `estimated_remaining_s` 分别是预计释放时间和剩余秒数。
  排队任务的 `estimated_duration_s` 是申报时长。`backfill_protect_in_s` 是离回填保护生效的秒数，`0` 表示已生效。
- llama-swap：是否暂停、已加载的模型、在途请求数（其中来自本机之外、gpu-lock 暂停前会等的有几个，规则同 `lsw-inflight`）。
- **系统实测**（不是记账）：`vm_stat` 的 wired、`sysctl` 的内存压力 / 可用百分比 / 压缩 / swap，以及 gpu-lock 发放份额前的
  内存检查（wired ≤ 6000 + 已预约）此刻是否通过。

读状态的方式：gpu-lock 每次改状态都在 `.locks/mutex` 目录互斥锁里进行。gpu-top 不拿这个锁，而是像 seqlock 一样读：读前读后都
没有 mutex、目录列表和各文件 mtime/大小前后一致，才算一致快照；否则隔 50 ms 重读，最多 6 次。互斥锁一直被占（第一个持有者
暂停 llama-swap 时可占几十秒）就显示上次读到的内容并标 `STATE BUSY`，不会卡住。读的途中消失的文件跳过；只写了一半的文件
标 `?`，按 gpu-lock 的规则计（没有数字 = 整个池）。JSON 里 `state.consistent` / `state.busy` 表示这一点。

示例（2026-11-21，一个独占持有者、6 个排队；排第一的 `mpm_tests` 已等满 20 分钟，后面的都不能插队）：

```text
gpu-top  2026-11-21 23:54:05   pool 82032 MB (80.1 GB)   reserved 82032 MB (80.1 GB)   free 0 MB (0.0 GB)
[############################################################] 100%
llama-swap: PAUSED   loaded: none   in flight: 0 (0 from outside)
memory (measured, not reservations): wired 7690 MB (7.5 GB) of limit 88064 MB; pressure normal, 89% available; compressed 0.5 GB; swap used 0.8 GB
gpu-lock memory check (wired <= 6000 + reserved = 88032 MB): ok

HOLDERS (1)
NAME                     MB    HELD    BEAT    PID ALIVE  AGENT          COMMAND
a1121                   all  15m31s      6s  38809 yes    73217 claude   /bin/zsh ./s4.sh

QUEUE (6)
 # NAME                        WANTS  WAITED    POLL    PID STATUS
 1 mpm_tests                    8000  46m37s     25s  81578 waiting: needs 8000 MB, 0 MB free for it
 2 qwen3mlxaudio                 all  41m03s     22s  57876 waiting: nobody may pass mpm_tests (waiting > 20 min)
 3 strata_rebase                 all  35m47s      8s  30124 waiting: nobody may pass mpm_tests (waiting > 20 min)
 4 mflux_fusion_pr_g3           8000   9m16s     13s  83580 waiting: nobody may pass mpm_tests (waiting > 20 min)
 5 lora_compat                 30000   5m10s      8s  51545 waiting: nobody may pass mpm_tests (waiting > 20 min)
 6 lora_compat_h3              12000   3m10s      9s  78413 waiting: nobody may pass mpm_tests (waiting > 20 min)
```

`gpu-lock status` 也能看，但它会拿互斥锁并顺带清理过期份额和票据；只想看时用 gpu-top。测试：`tests/test_gpu_top.py`
（假状态目录，不调用 gpu-lock）。

## 相关规则

- 规则 2：只有持有份额时才能为直接测试（在 llama-swap 之外起服务）腾内存。脚本已替你暂停 llama-swap，再暂停一次无害；
  别人持有份额时不要自己 resume llama-swap。自测服务用 18100–18199 端口，释放份额前先停掉。
- 大内存任务用 `tools/memguard` 包一层（见 [operations.md](operations.md)）：gpu-lock 只记账，不限制实际占用。
