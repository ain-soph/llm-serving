# llm-collab：bridge 启动器与配置

Claude Code、Codex、Antigravity 共用的 MCP 服务 `llm-collab`，由两部分组成：

- [ain-soph/llm-collab](https://github.com/ain-soph/llm-collab)：MCP 协调服务（投票、交接、配额、完全控制等策略）；
- [ain-soph/local-agent-bridge](https://github.com/ain-soph/local-agent-bridge) 分支 `serving`：Bridge 运行时
  （agy Orchestrator/xhigh/`--settings`、所有 origin 分支可见的 writer checkout、`BRIDGE_CODEX_MODEL`）。

本目录只放启动器和配置；`tools/install_agents.py` 按 `pins.json` 把两个仓库装到 bridge 目录并注册 MCP。

| 文件 | 装到 `~/llm-serving/platform/agent-bridge/` | 作用 |
| --- | --- | --- |
| `pins.json` | （不复制） | 两个仓库的 URL、分支、固定 commit 和安装位置（`release/`、`runtime/upstream/`） |
| `launcher.sh` | `mcp` | MCP stdio 入口。沿用旧私有 wrapper 的文件名，客户端里已有的注册无需改动 |
| `llm-collab.json` | `llm-collab.json` | llm-collab 配置，路径全部相对 bridge 目录（workspace、state、运行时、生产根目录 `../..`） |
| `llm-collab.env.sh` | `llm-collab.env` | 委派回合的模型和 effort（见下） |
| `codex-select` | `codex-select` | 选择 Codex 可执行文件（见下） |

不受管理、安装器从不改动的：`state/`（任务账本、交接、`state/codex/` 委派 Codex 的 home）、`workspace/`（已存在时）、
`llm-collab.local.env`（本机覆盖，启动器在 `llm-collab.env` 之后读取）。

## 模型与开关

| 设置 | 值 | 位置 |
| --- | --- | --- |
| `AGY_MODEL` | `orchestrator:gemini-4-neon-xhigh+gemini-4-hydrogen-xhigh` | `llm-collab.env` |
| `ANTHROPIC_MODEL` / `CLAUDE_CODE_EFFORT_LEVEL` | `claude-opus-5-5` / `high` | `llm-collab.env` |
| `BRIDGE_CODEX_MODEL` / `BRIDGE_CODEX_REASONING_EFFORT` | `gpt-6.2-sol` / `medium` | `llm-collab.env`（Bridge 每次启动 Codex 时写入 `state/codex/config.toml`） |
| 完全控制、Codex 无沙箱 | `fullControl: true`、`codexYolo: true` | `llm-collab.json` |
| 任意目录可作工作区 | `workspaceRoot: "/"` | `llm-collab.json` |

## Codex 可执行文件

`codex-select` 的顺序：环境变量 `LLM_COLLAB_CODEX_BIN`（写在 `llm-collab.local.env`）→ Codex app-server daemon 在
`~/.codex/packages/app-server-daemon/releases/<版本>-*/bin/codex` 装的同版本副本 → `PATH` 上的 `codex`。
原因：Homebrew Caskroom 里那份 `codex-code-mode-host` 曾卡在 `_dyld_start`（去掉 quarantine 也没用，复制一份同一文件就能启动），
委派 Codex 的每次工具调用都以 “gRPC code-mode host timed out waiting for session opening response” 超时。
daemon 副本与 Homebrew 版本号相同、内容一致，所以优先用它没有副作用；`brew reinstall --cask codex` 修好后两者等价。

## 安装、检查、回滚

```sh
.venv/bin/python tools/install_agents.py --dry-run   # 列出将要做的改动，不写
.venv/bin/python tools/install_agents.py             # 安装或更新（含 CLAUDE.md 规则和本地 worker）
.venv/bin/python tools/install_agents.py --check     # 有差异时退出码 1
.venv/bin/python tools/install_agents.py --home /tmp/x   # 装到另一个 home（测试用）
```

- 仓库：缺失时 clone；干净但不在固定 commit 时 fetch 后 `checkout --detach`；有本地改动时报错，不动。
  依赖用 `npm ci --ignore-scripts`，`node_modules` 已与 `package-lock.json` 一致时跳过。`--source NAME=URL` 可换来源。
- Node.js 22+：`LLM_COLLAB_NODE`、bridge 自带的 `runtime/node`，否则用安装时找到的 node（记成 `<bridge>/node` 符号链接），
  再否则 Homebrew 的 `node`。
- 注册：Claude Code（user scope）、Codex（`$CODEX_HOME` 或 `~/.codex`）、Antigravity（`~/.gemini/config/mcp_config.json`），
  都通过各自的 CLI 写入，已指向 `<bridge>/mcp` 时不动。委派 Codex 的 `state/codex/config.toml` 由 Bridge 每次按白名单重写，
  不能也不应该在那里注册（否则委派回合会递归调用 llm-collab）。在委派会话里运行安装器时，继承来的 `CODEX_HOME`、`CLAUDE_CONFIG_DIR` 若指向 bridge 目录内部会被忽略，改用 `<home>` 下的默认位置。`--no-register` 跳过注册，`--no-bridge` 跳过整个 bridge。
- 回滚：被替换的文件先备份到 `~/llm-serving/.backup/llm-serving/platform/agent-bridge/<文件名>.<时间>-install-agents`。
  恢复旧启动器：`cp -p <备份的 mcp> ~/llm-serving/platform/agent-bridge/mcp`，然后在客户端里重启 MCP 服务。
  安装器不删除旧 wrapper 的 `server.mjs`、`provider.mjs`、`team.mjs`，所以恢复 `mcp` 即完整回到旧版本。
