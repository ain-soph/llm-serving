# 多 agent 分工（全局规则）

本段由 llm-serving 仓库 `agents/claude-code/global-CLAUDE.md` 经 `tools/install_agents.py` 装入；改仓库里的版本后重装。
详细说明：`~/llm-serving/portable/docs/multi-agent.md`；本地 worker：`~/llm-serving/portable/agents/local-worker/README.md`。

- **主窗口只编排**：与用户交流、拆分任务、派发、跟进、回答进度；实际工作交给 subagent（统一 Opus 5.5 high）。分工可按判断调整。
- **按难度分级**
  - L1 打杂（按清单改代码、样板代码、跑测试、收集输出）→ Codex（gpt-6.2-sol medium，经 llm-collab bridge）或本地 worker。
  - L2 常规开发、调试、分析 → Claude subagent。
  - L3 高难（内核融合、算子编排、难优化）→ Claude subagent 与 Antigravity（Orchestrator：Neon planner xhigh + Hydrogen
    executor xhigh）各自独立出方案，再互审；必要时 Fable 5.2 high 做顾问和审阅。
  - Antigravity 只在具体的高难技术设计/分析（内核、算子编排、性能优化）时引入；普通工程（仓库整理、安装/恢复链、
    配置、发布）由 Claude 独自规划，不走 agy 审阅轮。
- **性能分歧用 benchmark 决定，不靠投票**；Codex 不参与高难设计投票。
- **测试清单**（命令、参数、输出路径、验收标准）由 Claude/agy 设计。Codex 和本地 worker 只执行并收集原始结果，
  不得设计实验、挑选或跳过测试、改阈值、下结论；结果由 Claude subagent 分析。
- **写审分离**：L3 任务中 agy 写的由 Claude 审，Claude 写的由 agy 审。work 模式下每个任务只有一个写入者，并行任务用不同的 `taskBranch`。
- **llm-collab bridge**（MCP `llm-collab`）：`start_collaboration` 的 review 模式用于讨论（结论用 `record_decision` 记下），
  work 模式用于实现（给出计划、验收检查和固定的 `resumeKey`）；`get_collaboration` 看进度、证据和 `latestHandoff`，
  `continue_collaboration` 继续同一任务 ID，`list_collaborations` 按任务找。
- **等待长任务**（用户 2026-11-21）：不用短间隔轮询等长任务或 gpu-lock 队列，每次唤醒都要重读整个上下文。每个 GPU 阶段写成
  **一条**后台命令（如 Claude Code `run_in_background` Bash），依次：排队拿锁（`gpu-lock acquire NAME all|MB`，exit 1 就再调）、
  跑 memguard 包住的任务、在脚本**顶层**的 `trap` 里 release（不要放在 zsh 函数里：那里的 EXIT trap 提前触发，立刻放了锁）、
  结果写进文件。然后结束回合，等退出通知唤醒。可选的兜底心跳 ≥30 分钟（如后台 `sleep 1800`，查一次再重设）；不用更短的
  sleep 或轮询（`get_collaboration` 也一样）。过了预计结束时间还没完，每约 30 分钟查一次。有未完成的工作时，结束回合前
  必须有会唤醒你的东西。编排者也不主动查，靠完成通知。
- **上下文卫生**（不靠自动压缩）：长任务在每个阶段边界把进度写进运行目录下的 checkpoint 文件（状态、数字、commit、下一步）。
  subagent 上下文到约 300k token 以上、又要进入长等待或新阶段时，先写 checkpoint 再汇报；编排者另起新 agent 读 checkpoint
  接着做，不让一个巨大的 agent 一直挂着。额度 3% 交接也从这些 checkpoint 出发。
- **本地 worker**（本机 Strata 上的 Qwen 3.8 Flash-Next，Codex 之下一层；编排者用 Bash 调用，不是 bridge provider）：
  `~/llm-serving/platform/agent-bridge/local-worker/run --cwd <目录> [--harness claude|codex] [--variant normal|uncensored]`
  `[--wait] [--timeout 60m] [--task-file F | -- "任务"]`（默认 claude harness、normal）。stdout 是一个 JSON 摘要，日志在 `logs/<run>/`。
  - 退出码：0 harness 结束（是否完成要自己验）、2 参数错、3 模型/环境错、**75 GPU 忙**（没加 `--wait`）或 gpu-lock 拒绝、124 超时。
  - 只做清单式机械活：无网络，只能写 `--cwd`，不装依赖、不测速、不碰生产配置。结果必须由 Claude 自己跑测试、看 diff 验收。
  - 运行期间占约 70 GB gpu-lock（70000 MB），别人的 benchmark 会排队；用完即释放，不要长时间挂着。
  - `run`（尤其带 `--wait`）放在后台命令里跑，等它的退出通知，不轮询。
