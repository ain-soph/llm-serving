# local-worker：本地 Flash-Next 打杂 worker

用本机 Strata 上的 Qwen 3.8 Flash-Next（llama-swap `qwen3.8-flash-next` / `-uncensored`）驱动一个无头编码 agent，
harness 可选 **Claude Code CLI** 或 **Codex CLI**（两者平等可选，共用同一套加锁、沙箱、超时、日志逻辑）。
定位：团队里 Codex 之下的一层，只做**清单式机械活**，结果必须由 Claude/调度者自己核对。

代码在公开仓库 [ain-soph/llm-serving](https://github.com/ain-soph/llm-serving) 的 `agents/local-worker/`；
`tools/install_agents.py` 把 `~/llm-serving/platform/agent-bridge/local-worker` 链接到这里（见 `docs/multi-agent.md`）。
依赖本机部署：`~/llm-serving/bin/gpu-lock`、`services/text/strata/configs`、`platform/llama-swap/.api_key`。

## 用法

```sh
W=~/llm-serving/platform/agent-bridge/local-worker/run

# 默认：claude harness、normal 变体、60 分钟硬超时；GPU 有人持锁或排队就立即退出（exit 75）
$W --cwd ~/some/scratch/repo -- "把 foo/*.py 里的 print 改成 logging，并运行 pytest 修到通过"

# Codex harness、等待 GPU、30 分钟、任务写在文件里
$W --harness codex --wait --timeout 30m --cwd /path/to/repo --task-file task.md

# uncensored 变体（llama-swap ID qwen3.8-flash-next-uncensored；结束时卸载它，不留额外加载状态）
$W --variant uncensored --cwd /path/to/dir -- "..."
```

参数：`--harness claude|codex`（默认 claude）、`--variant normal|uncensored`（默认 normal）、`--cwd DIR`（必填，
worker 只能写这里）、`--timeout 90s|45m|1h`（30 s–60 min，默认 60m，到点杀掉整个进程组）、`--wait`（等 GPU，
否则忙就退出）、`--max-turns N`（仅 claude，默认 100）、`--label TEXT`（日志目录名后缀）、`--task-file FILE` 或
`-- "任务"`。`--under-lock NAME` 仅供已经独自持有 ≥69500 MB 份额的脚本使用（如 `accept/run_acceptance.sh`）。

输出：stdout 只有一个 JSON 摘要（`status`、`exit_code`、各阶段秒数、`num_turns`、`tool_calls(_by_name)`、
`tool_errors`、`tool_format_errors`、`engine`（本次运行期间 Strata 日志里的每请求 prefill/decode 速度汇总）、
`external_network_attempts`、`git` 状态、`final_message`）。进度在 stderr。完整记录在 `logs/<时间>-<harness>-<variant>[-label]/`：
`events.jsonl`（Claude Code `stream-json` / Codex `exec --json` 原始事件）、`stderr.log`、`run.log`、`prompt.txt`、
`net-attempts.jsonl`、`summary.json`。

编排者在后台命令里调用 `run`（如 Claude Code `run_in_background` Bash；带 `--wait` 时尤其如此），结束回合等退出通知，
不用短间隔轮询（规则见 `agents/claude-code/global-CLAUDE.md` 的“等待长任务”）。

退出码：0 harness 正常结束（任务是否真的完成要自己验）；2 参数错；3 模型/环境错误；75 GPU 忙（未加 `--wait`）
或 gpu-lock 拒绝；124 硬超时；其他 = harness 自身退出码。

## 每次运行做了什么

1. `gpu-lock status`：有持锁者或排队者且没有 `--wait` → 打印 `{"status":"busy",...}`，exit 75。
2. `gpu-lock try local_worker_<pid> 70000`（Strata 入口 69500 + 余量）。gpu-lock 给第一个持锁者暂停 llama-swap
   并等外部用户请求（最多 120 s）结束，这段总是等。拿到后若发现还有别的持锁者，则立即释放（不在别人持锁时
   resume llama-swap，AGENTS.md 规则 2）。拿到后每 10 分钟 heartbeat。
3. `POST /api/models/resume`，再用 `/upstream/<模型>/health` 把模型加载好（约 30–60 s）。
4. 在 macOS `sandbox-exec`（`lib/sandbox.sb`）里启动 harness：出站网络只允许 `localhost:8080`（网关）和本次的
   拒绝代理端口；文件写入只允许 `--cwd`、harness 自己的配置目录、本次日志目录（含 TMPDIR）、`/dev`。环境变量
   `env -i` 清空后只给必要的；`HTTP(S)_PROXY` 指向 `lib/deny_proxy.py`，它对所有请求回 403 并记录目标，
   因此 `external_network_attempts` 能看到谁想上网（沙箱本身已在内核层拒绝）。
5. EXIT/INT/TERM trap：杀掉 harness 进程组、停 heartbeat 和代理；uncensored 变体先卸载；`gpu-lock release`
   （最后一个持锁者离开时 llama-swap 自动 resume）；写摘要。模型默认常驻 normal 版，normal 运行结束后它保持已加载。

## 两种 harness 的配置与差异

| | Claude Code（`--harness claude`） | Codex CLI（`--harness codex`） |
|---|---|---|
| 版本（验收时） | 2.1.323 | codex-cli 0.171.0 |
| 配置目录 | `claude-config/` = `CLAUDE_CONFIG_DIR`（运行时状态，git 忽略） | `codex-home/` = `CODEX_HOME`（只提交 `config.toml`、`rules/default.rules`） |
| 协议 | `/v1/messages`（Anthropic） | `/v1/responses`（`wire_api="responses"`） |
| 鉴权 | `ANTHROPIC_API_KEY` = llama-swap key（运行时从 `.api_key` 读入环境，不写进文件；`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` 使 Bash 子进程看不到它） | `env_key = "LOCAL_WORKER_API_KEY"`（同上；`shell_environment_policy` 排除 `*KEY*`/`*TOKEN*`） |
| 模型映射 | `ANTHROPIC_MODEL`、`ANTHROPIC_SMALL_FAST_MODEL`、`ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL`、`CLAUDE_CODE_SUBAGENT_MODEL` 全部 = 本地模型 | `-m <模型>`，provider `strata` |
| 关闭外联 | `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`、`DISABLE_{AUTOUPDATER,TELEMETRY,ERROR_REPORTING}=1`、`--strict-mcp-config`、`--setting-sources ""`（不读用户/项目设置）、`--no-session-persistence` | `analytics/feedback` 关、`web_search="disabled"`、关闭 plugins/apps/浏览器/图像等 features（plugins 同步曾试图连 github.com）、`--ephemeral` |
| 工具 | `--tools Bash,Read,Edit,Write,Glob,Grep`、`ENABLE_TOOL_SEARCH=false` | `exec_command`、`write_stdin`、`request_user_input`、`view_image`（无 apply_patch 工具，改文件靠 shell） |
| 权限 | `lib/claude-settings.json`：`acceptEdits` + `--permission-prompts none`（cwd 外编辑自动拒绝）；允许 Bash；拒绝 `git push`/`git remote`/`git config`、curl/wget/ssh/scp/rsync/nc、sudo/launchctl/brew、pip/uv/npm 安装、gpu-lock/lsw-apply、WebFetch/WebSearch、读 `.api_key` | Codex 自带沙箱与外层 sandbox-exec 不能嵌套，所以用 `--dangerously-bypass-approvals-and-sandbox`，由外层沙箱限制文件和网络；`rules/default.rules` 禁止 `git push`/`git remote`/curl/wget/ssh/gpu-lock |
| 上下文 | `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`、`CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768` | `model_context_window=262144`、自动压缩 120000 |
| 已知提示 | stderr `[claude-code:unrecognized_model]`，无害 | 事件里 `Model metadata ... not found`，无害（摘要已过滤） |

验收对比与默认值建议见下文“验收结果”。默认 harness 为 **claude**。

## 适用 / 不适用

适用（清单式机械活，验收标准由派活方写清）：按明确清单做小范围代码修改、重命名、格式/样板代码、补 docstring、
按给定命令跑测试并修到通过、收集命令输出、批量文本替换、整理日志。

不适用：实验设计、决定跑哪些测试、判定结果/结论、任何 GPU 工作（worker 自己就占着 70 GB 锁，任务里不能再
加载模型或跑 benchmark）、生产配置（llama-swap/网关/LaunchAgent）、git push 或任何联网操作、安装依赖
（没有网络；需要的环境由派活方事先在 `--cwd` 里准备好，如 `.venv`）。

## 限制与注意

- 运行期间占 70000 MB gpu-lock：其他 agent 的 benchmark 会排队（用户已接受）。默认遇忙即退，不抢队。
- gpu-lock 进入时会暂停 llama-swap，因此 worker 先拿锁再 resume；任何别的 agent 在 worker 运行中 pause llama-swap
  都会让本次请求中断（harness 会报错退出，trap 照常释放锁）。
- Bash 子进程能访问 `localhost:8080`（harness 本身要用），理论上能调用网关 API；靠权限规则和任务约束，不是硬隔离。
- 沙箱只限制写，不限制读；不要把 worker 指向含机密的目录。`--cwd` 拒绝 `~`、`~/llm-serving` 根目录及
  `platform/`、`services/`、`bin/`、`~/models`、`~/.claude`、`~/.codex`。
- worker 自述不可信：派活方必须自己跑测试、看 diff（参考 `accept/verify.sh`）。

## 验收

`accept/`：`fixture/`（带 2 个 bug、缺一个函数的小 Python 包 + pytest）、`task.md`（任务）、`hidden_test.py`
（不给 worker 看的额外检查）、`make_repo.sh`（新建 git 副本 + `.venv`）、`verify.sh`（独立验证）、
`run_acceptance.sh BASE [claude codex]`（一次独占 gpu-lock 内依次跑各 harness，释放后再验证）。

### 验收结果（2026-11-21）

任务（`accept/task.md`）：读 README、包代码和测试；修 `parsing.py`、`report.py` 各 1 个 bug；在 `report.py` 新增
`top_n` 并从 `__init__.py` 导出；README 加一行；跑 pytest 修到通过；不得改 `tests/`。每次一个全新仓库副本。
结果由 `accept/verify.sh` 独立检查：可见测试 7 个、隐藏测试 5 个（不给 worker 看）、`tests/` 未改、人工看 diff。

环境：M5 Ultra，`qwen3.8-flash-next`（AD-4.27，主体约 3.57 bpw，INT8 KV，Q2_0 MTP `--spec 3`，262144 × 2 槽，
GPU 预约 69,500 MiB，实测峰值 wired +67,695 MiB 见 `reports/strata-compact-dual-20261119.md`），并发 1。
A1/A2 在独占 gpu-lock（82,032 MB）下运行；S 为 `run` 自己拿 70,000 MB 份额（期间无其他持锁者）。
prefill = 本次所有请求新读 token / 读取时间（括号内只算新读 ≥1024 token 的请求）；decode 含 MTP 接受的草稿。

| 轮次 | harness | 结果 | harness 用时 | 模型请求 | 工具调用 | 工具错误 / 格式错误 | 首个 / 最大 prompt | 新读 / 生成 token | prefill tok/s | decode tok/s | 外网尝试 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | claude | 7/7 + 隐藏 5/5 | 19 s | 6 | 12（Read 6、Edit 4、Bash 2） | 0 / 0 | 4,720 / 8,202 | 6,355 / 2,187 | 1,764（2,005） | 174.5（143.6–211.3） | 0 |
| A1 | codex | 7/7 + 隐藏 5/5 | 16 s | 8 | 9（命令 5、文件修改 4） | 0 / 0 | 6,102 / 9,740 | 8,498 / 1,617 | 1,680（1,964） | 153.6（136.9–171.6） | 0 |
| A2 | claude | 7/7 + 隐藏 5/5 | 21 s | 7 | 13（Read 6、Edit 5、Bash 2） | 0 / 0 | 4,720 / 8,738 | 6,457 / 2,533 | 1,759（2,131） | 170.5（138.8–209.3） | 0 |
| A2 | codex | 7/7 + 隐藏 5/5 | 29 s | 12 | 14（命令 9、文件修改 5） | 2 / 0 | 6,102 / 11,852 | 8,981 / 3,205 | 1,484（1,917） | 160.9（141.2–188.2） | 0 |
| S | claude，uncensored | 7/7 + 隐藏 5/5 | 18 s（换载模型 19 s） | 6 | 13 | 0 / 0 | 4,726 / 8,459 | 6,466 / 2,217 | 1,553（1,702） | 177.1（136.0–207.4） | 0 |
| S | codex | 7/7 + 隐藏 5/5 | 23 s（加载 10 s） | 8 | 7 | 0 / 0 | 6,102 / 10,683 | 8,793 / 2,195 | 1,667（1,821） | 151.1（141.3–161.0） | 0 |

- 两种 harness 6 次全部完成，diff 都只动 4 个该动的文件，修法与参考答案一致；没有任何工具调用格式错误，
  没有请求被记录为发往外网（deny proxy 0 条），llama-swap 只收到本机请求。
- A2 codex 的 2 个工具错误是 zsh here-document 写 `/tmp/zsh*` 被沙箱拒绝（worker 换用别的写法后完成）；
  已修：`run` 设置 `TMPPREFIX` 到本次日志目录，Codex `shell_environment_policy.inherit="all"`，离线复测通过。
- Strata 会话缓存让多轮请求几乎只读新增部分（每轮新读几十到上千 token），所以 harness 的系统提示大小对速度
  影响很小：精简工具后 Claude Code 首请求约 4.7K token，Codex 约 6.1K。
- 选择默认 **claude**：模型请求更少（6–7 对 8–12）、新读 token 更少、decode 略高、0 工具错误；权限可以按命令
  细分拒绝，且 `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` 让 Bash 看不到 key。Codex 同样可用、结果同样正确，
  适合想要 Codex 工具语义（exec_command/apply_patch 式改文件）的任务。样本只有一个小任务、各 3 次，差距不大。

离线策略测试（假 API 服务器按脚本发工具调用，不占 GPU）：cwd 外写文件被沙箱拒绝；Claude 的 `git push`、
`curl` 被权限规则拒绝，Write 到 cwd 外被自动拒绝；Codex 的 `git push` 被 execpolicy 禁止（本地 bare 远端
未收到提交）；Python `urllib` 外联被代理 403 并记录；Bash/shell 子进程环境里没有 API key；硬超时 30 s 时
整个进程组被杀，无残留进程；对持锁中的 worker 发 SIGTERM 后 trap 释放了 gpu-lock；GPU 被占时第二个
`run`（无 `--wait`）立即以 75 退出。
