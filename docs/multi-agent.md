# 多 agent 分工与恢复

本机开发用多个编码 agent：Claude Code、Antigravity（agy）、Codex，加一个跑在本机 Strata 上的本地 worker。
它们经 MCP `llm-collab` 共享任务、交接和决议。本页说明分工规则放在哪、怎么装到新机器，以及恢复链条里各部分来自哪个仓库。

## 分工规则

- 正文：[`agents/claude-code/global-CLAUDE.md`](../agents/claude-code/global-CLAUDE.md)，装进 `~/.claude/CLAUDE.md`，对所有
  Claude Code 会话生效。要点：主窗口只编排；L1 打杂交给 Codex 或本地 worker，L2 交给 Claude subagent，L3 由 Claude
  与 agy 各自出方案再互审（必要时 Fable 审阅）；性能分歧看 benchmark；测试清单由 Claude/agy 设计，执行者只收集原始结果；
  写审分离，work 模式每任务一个写入者。
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
| llm-collab MCP 服务 | [ain-soph/llm-collab](https://github.com/ain-soph/llm-collab) | 按其 README：`npm ci`、写 `config.json`、在三个客户端 `mcp add` |
| Bridge 运行时 | [ain-soph/local-agent-bridge](https://github.com/ain-soph/local-agent-bridge)（分支 `serving`） | clone 后装依赖，`config.json` 的 `bridgeRoot` 指向它 |
| 各 provider 的模型 | 启动 llm-collab 时的环境变量 | 见下 |
| Claude Code / agy / Codex CLI 与登录 | 各自官方安装 | 手动安装并登录 |

llm-collab 把启动环境原样传给 bridge。当前使用的模型（bridge 委派的回合）：

```sh
export AGY_MODEL="orchestrator:gemini-4-neon-xhigh+gemini-4-hydrogen-xhigh"  # Antigravity Orchestrator
export ANTHROPIC_MODEL="claude-opus-5-5" CLAUDE_CODE_EFFORT_LEVEL="high"      # Claude
export BRIDGE_CODEX_MODEL="gpt-6.2-sol" BRIDGE_CODEX_REASONING_EFFORT="medium" # Codex
exec /path/to/node /path/to/llm-collab/bin/llm-collab.mjs --config /path/to/config.json
```

本地 worker 依赖的 `gpu-lock`、Strata 服务配置和 llama-swap key 属于本机部署目录 `~/llm-serving`，不在本仓库；
缺 key 时 worker 以退出码 3 停下，缺 `gpu-lock` 时按 GPU 忙以 75 退出。
