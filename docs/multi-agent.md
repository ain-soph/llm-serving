# 多 agent 分工与恢复

本机开发用多个编码 agent：Claude Code、Antigravity（agy）、Codex，加一个跑在本机 Strata 上的本地 worker。
它们经 MCP `llm-collab` 共享任务、交接和决议。本页说明分工规则放在哪、怎么装到新机器，以及恢复链条里各部分来自哪个仓库。

## 分工规则

- 正文：[`agents/claude-code/global-CLAUDE.md`](../agents/claude-code/global-CLAUDE.md)，装进 `~/.claude/CLAUDE.md`，对所有
  Claude Code 会话生效。要点：主窗口只编排；L1 打杂交给 Codex 或本地 worker，L2 交给 Claude subagent，L3 由 Claude
  与 agy 各自出方案再互审（必要时 Fable 审阅）；性能分歧看 benchmark；测试清单由 Claude/agy 设计，执行者只收集原始结果；
  写审分离，work 模式每任务一个写入者；长任务和 gpu-lock 排队用一条后台命令等退出通知，不短间隔轮询（兜底 ≥30 分钟），
  阶段边界写 checkpoint，上下文约 300k token 时换新 agent 读 checkpoint 接着做。同一条等待规则也在生产 `AGENTS.md` 规则 1
  和 [gpu-lock.md](gpu-lock.md)。
- 同一套角色规则也写在 [llm-collab](https://github.com/ain-soph/llm-collab) 的 MCP 策略里，每个 bridge 任务都会带上，
  所以 Codex 和 agy 即使不读 CLAUDE.md 也会收到。

## 安装

`tools/bootstrap.sh` 最后一步会运行 `tools/install_agents.py`（设置 `LLM_SERVING_SKIP_AGENTS=1` 可跳过）。也可以单独运行：

```sh
.venv/bin/python tools/install_agents.py            # 安装或更新
.venv/bin/python tools/install_agents.py --check    # 只报告，不写；有差异时退出码 1
```

- `~/.claude/CLAUDE.md`：规则写在 `BEGIN/END llm-serving agents` 标记块里。文件不存在就新建；已有文件只替换或追加这个块，
  块外的内容保持不变。改动已有文件前先备份到 `~/llm-serving/.backup/.claude/CLAUDE.md.<时间>-install-agents`
  （按家目录下的相对路径存放，`--backup-dir` 可改）。标记不成对时报错，不写。
- 本地 worker：把 `~/llm-serving/platform/agent-bridge/local-worker` 建成指向本仓库 `agents/local-worker` 的符号链接，
  生产位置直接用仓库版本。那里已有目录或指向别处的链接时不动它，只提示。
- `tools/doctor.py` 会报告这两项是否一致（只打印 WARN，不影响部署检查结果）。

本地 worker 的运行状态（`logs/`、`claude-config/` 内容、`codex-home/` 里除 `config.toml` 和 `rules/default.rules`
以外的文件）留在仓库目录里，但被 `agents/local-worker/.gitignore` 忽略；API key 从部署目录的 `.api_key` 运行时读入，
不进仓库。用法、沙箱和验收结果见 [worker README](../agents/local-worker/README.md)。

## 恢复链条

| 部分 | 来源 | 新机器上怎么恢复 |
| --- | --- | --- |
| 全局 CLAUDE.md 规则 | 本仓库 `agents/claude-code/` | `tools/bootstrap.sh` 或 `tools/install_agents.py` |
| 本地 worker | 本仓库 `agents/local-worker/` | 同上（建符号链接） |
| gpu-lock、llama-swap 配置与 key、网关、Strata 路由、LaunchAgent | 本仓库 `serving/` | `tools/bootstrap.sh` 或 `tools/install_serving.py` |
| llm-collab MCP 服务 | [ain-soph/llm-collab](https://github.com/ain-soph/llm-collab)，commit 固定在 `agents/llm-collab/pins.json` | `tools/install_agents.py`（clone、`npm ci`、注册三个客户端） |
| Bridge 运行时 | [ain-soph/local-agent-bridge](https://github.com/ain-soph/local-agent-bridge)（分支 `serving`），同上固定 | 同上 |
| 启动器、配置、各 provider 的模型 | 本仓库 `agents/llm-collab/` | 同上（复制到 bridge 目录，旧文件先备份） |
| Claude Code / agy / Codex CLI 与登录 | 各自官方安装 | 手动安装并登录 |

模型、开关、Codex 可执行文件的选择、回滚步骤见 [agents/llm-collab/README.md](../agents/llm-collab/README.md)。

### 旧私有 wrapper 与 llm-collab + fork 的对照

旧 bridge 目录里手写的 `mcp`、`server.mjs`、`provider.mjs`、`team.mjs` 与已发布的 llm-collab（`src/` 同名文件）逐行对比后，
只剩下表中的差异；每项都在启动器或配置里补齐，不需要改 llm-collab 本身。

| 旧 wrapper 的行为 | 现在由谁提供 |
| --- | --- |
| 策略文本：协调者顺序、3% 交接、VOTE、GPU 行、Permissions 行、Roles 行 | llm-collab `src/server.mjs`，文字相同（`fullControl` 开时带 Permissions 行）；2026-11-21 起另加等待行和上下文行 |
| 委派回合完全控制（Claude/agy/Codex 去沙箱、去审批）；`continue_collaboration` 升级为 implement/yolo | llm-collab `src/provider.mjs`、`src/server.mjs`；`llm-collab.json` 的 `fullControl`、`codexYolo` |
| `workspace` 可为任意目录（`BRIDGE_WORKSPACE_ROOT=/`）、`taskBranch` 私有 checkout | llm-collab；`llm-collab.json` 的 `workspaceRoot: "/"` |
| `AGY_MODEL` 作为 Antigravity 默认路由 | llm-collab `src/server.mjs`；值在 `llm-collab.env` |
| `ANTHROPIC_MODEL`、`CLAUDE_CODE_EFFORT_LEVEL`、`BRIDGE_CODEX_MODEL`、`BRIDGE_CODEX_REASONING_EFFORT` | fork（模型策略、`prepare-codex-home`）；llm-collab 原样传递环境；值在 `llm-collab.env` |
| 配额、交接、投票（`team.mjs`）；投票/交接/状态四个本地工具 | llm-collab `src/team.mjs`（配额收集器从 fork 动态加载） |
| `BRIDGE_LLM_COLLAB_CODEX`：Gatekeeper 卡住时改用 daemon 的 Codex 副本 | `codex-select`（`llm-collab.json` 的 `providers.codex`），每次启动时自动检测 |
| provider 写死 `/opt/homebrew/bin/{claude,agy,codex}`；`PATH` 写死 | 启动器设固定 `PATH`（Homebrew、`~/.local/bin`、系统目录），provider 按名字查找 |
| 私有 Node（`runtime/node`） | 启动器优先用它，否则用安装时的 node 或 Homebrew `node`（`Brewfile`） |
| state、委派 Codex home 在 `state/`、`state/codex/`；`umask 077`；`KEEP_MODELS_WARM=0`；cwd 固定 | llm-collab `bin/llm-collab.mjs`；`stateDirectory: "./state"`，原有 state 直接沿用 |
| `workspace_status` 读 `../../logs/health/STATUS` | `productionRoot: "../.."`；文件缺失时 llm-collab 返回 `null`（旧版会报错） |
| 生产切换说明里点名 `gpu-lock`、`memguard`、`lsw-apply` | llm-collab 改为“按该部署自己的操作规程”；这些规程本来就写在生产目录的 `AGENTS.md`，GPU 行也指向它。接受此文字差异 |

本地 worker 依赖的 `gpu-lock`、Strata 服务配置和 llama-swap key 在本机部署目录 `~/llm-serving`；脚本和配置模板公开在
`serving/`，由 `tools/install_serving.py`（bootstrap 最后一步）装回，key 在安装时生成，见 [serving-restore.md](serving-restore.md)。
GPU 锁规则见 [gpu-lock.md](gpu-lock.md)。缺 key 时 worker 以退出码 3 停下，缺 `gpu-lock` 时按 GPU 忙以 75 退出。
