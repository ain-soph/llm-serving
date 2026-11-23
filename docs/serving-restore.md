# 恢复 ~/llm-serving 的服务胶水

本地 worker（`agents/local-worker`）和各 agent 依赖部署目录 `~/llm-serving` 里的几样东西：GPU 锁脚本、llama-swap
启动脚本、主配置和 key、:8080 网关、Strata 两个文本入口的路由与引擎配置，以及开机启动它们的 LaunchAgent。公开版本放在
本仓库 `serving/`，由 `tools/install_serving.py` 安装；`tools/bootstrap.sh` 最后会运行它（`LLM_SERVING_SKIP_SERVING=1` 跳过）。

```sh
.venv/bin/python tools/install_serving.py --dry-run     # 只打印计划
.venv/bin/python tools/install_serving.py               # 创建缺失文件；与仓库不同的已有文件只报告 drift，不动
.venv/bin/python tools/install_serving.py --update      # 先备份到 ~/llm-serving/.backup/<相对路径>.<时间>-install-serving 再替换
.venv/bin/python tools/install_serving.py --check       # 不写；有缺失或差异时退出码 1（--diff 显示差异）
.venv/bin/python tools/install_serving.py --render DIR  # 只把模板渲染到 DIR（相对 home 的路径），用来和线上文件 diff
```

`--home DIR` 把目标换成 `DIR/llm-serving` 和 `DIR/Library/LaunchAgents`（测试用临时目录）。安装不加载、不卸载、不重启任何
服务，也不需要 GPU：写入或替换 LaunchAgent 后只打印要执行的 `launchctl bootstrap gui/$(id -u) <plist>`（替换时先
`launchctl bootout`）。备份路径：`~/llm-serving` 内的文件按其内的相对路径，`~/Library/LaunchAgents` 的按相对 home 的路径
（`.backup/Library/LaunchAgents/...`）。`tools/doctor.py` 用同一套逻辑报告这些文件是否与仓库一致、key 是否存在且为 0600
（只打印 OK/WARN，不打印 key）。

## 清单

| 线上文件（相对 `~`） | 仓库位置（`serving/` 下） | 方式 | 原因 |
| --- | --- | --- | --- |
| `llm-serving/bin/gpu-lock` | `bin/gpu-lock` | 原样 | 多 agent 显存预约锁；路径都从 `$HOME` 推出。规则见 [gpu-lock.md](gpu-lock.md) |
| `llm-serving/bin/gpu-lock-run` | `bin/gpu-lock-run` | 原样 | 执行托管 GPU 分片、进程组和持锁时限 |
| `llm-serving/bin/gpu-pool` | `bin/gpu-pool` | 原样 | gpu-lock、serve.sh、lsw-apply 用它从 `iogpu.wired_limit_mb` 算池大小 |
| `llm-serving/bin/gpu-top` | `bin/gpu-top` | 原样 | 只读查看 gpu-lock 池子、持有者、队列和 llama-swap（类似 top），见 [gpu-lock.md](gpu-lock.md) |
| `llm-serving/bin/memguard` | `../tools/memguard` | 原样 | 监测系统内存压力和进程组内存占用 |
| `llm-serving/bin/lsw-inflight` | `bin/lsw-inflight` | 原样 | gpu-lock 暂停 llama-swap 前统计外部在途请求 |
| `llm-serving/bin/lsw-apply` | `bin/lsw-apply` | 原样 | 校验 llama-swap 配置并重启（只许独占 gpu-lock 持有者运行）；只用 `$HOME` 路径 |
| `llm-serving/bin/gpu-lock-watchdog` | `bin/gpu-lock-watchdog` | 原样 | 强制释放空闲份额；只用 `Path.home()` |
| `llm-serving/bin/com.local.gpu-lock-watchdog.plist.disabled` | `LaunchAgents/com.local.gpu-lock-watchdog.plist` | 模板 | watchdog 的 LaunchAgent 按用户决定（2026-11-15）停用，所以只装到脚本旁，不进 `~/Library/LaunchAgents` |
| `llm-serving/platform/llama-swap/serve.sh` | `platform/llama-swap/serve.sh` | 原样 | 读 `.api_key` 设 `LLAMA_API_KEY`，用 gpu-pool 启动 llama-swap（127.0.0.1:8081） |
| `llm-serving/platform/llama-swap/start.sh`、`stop.sh` | `platform/llama-swap/` 同名 | 原样 | 手动加载/卸载 llama-swap LaunchAgent |
| `llm-serving/platform/llama-swap/config.yaml` | `platform/llama-swap/config.yaml` | 模板 | 主配置；key 只以 `${env.LLAMA_API_KEY}` 出现，运行时由 serve.sh 从 `.api_key` 读入，文件里没有 key |
| `llm-serving/platform/llama-swap/conf.d/strata.yaml` | `platform/llama-swap/conf.d/strata.yaml` | 模板 | `qwen3.8-flash-next` / `-uncensored` 两个入口 |
| `llm-serving/platform/gateway/Caddyfile` | `platform/gateway/Caddyfile` | 模板 | :8080 网关（worker 和 gpu-lock 都调用 :8080）；受信代理地址来自取值文件 |
| `llm-serving/platform/gateway/com.local.llm-gateway.plist` | `LaunchAgents/com.local.llm-gateway.plist` | 模板 | 网关 LaunchAgent 在网关目录里的副本（与下面那份相同） |
| `llm-serving/services/text/strata/configs/strata-ad427-c262k-p2-compact.json` | `services/text/strata/configs/` 同名 | 模板 | normal 变体引擎参数 |
| `llm-serving/services/text/strata/configs/strata-uncensored-c262k-p2-compact.json` | 同上 | 模板 | uncensored 变体引擎参数 |
| `Library/LaunchAgents/com.local.llama-server.plist` | `LaunchAgents/com.local.llama-server.plist` | 模板 | 登录时运行 serve.sh |
| `Library/LaunchAgents/com.local.llm-gateway.plist` | `LaunchAgents/com.local.llm-gateway.plist` | 模板 | 登录时运行 Caddy 网关 |
| `llm-serving/platform/llama-swap/.api_key` | 无 | 安装时生成 | 不存在时生成随机 key（mode 0600）；已存在绝不覆盖（只把权限收紧到 0600）；被 `.gitignore` 忽略，`privacy_check` 拒收 |
| 本机取值 | `.local/serving-values.json` | 私有 | 模型、Strata 构建路径、受信代理地址；`.local/` 和文件名 `serving-values.json` 都被忽略且被 `privacy_check` 拒收 |
| `llm-serving/AGENTS.md` 规则 1 | [docs/gpu-lock.md](gpu-lock.md) | 公开改写 | `AGENTS.md` 其余部分含本机和网络细节，不公开（`privacy_check` 拒收该文件名） |

不在本工具范围、仍只在本机的：其他 10 个 `conf.d` 片段（仓库已有 `templates/conf.d/` 版本，由 `tools/configure.py` 按仓库
布局渲染）、Strata 的实验配置（`configs/` 下其他 JSON，不被路由）、`services/text/strata/config/`（Strata 自己写的设置）、
其他 LaunchAgent（health、decisions、h3-video-files）和它们的服务代码。

**已知缺口**：`config.yaml` 启动时预加载 `embeddinggemma-2`（定义在 `conf.d/embedding.yaml`），`minimax-h3-image` 用
`~/llm-serving/services/image/h3-image-mlx/server.py`；网关把 `/v1/decisions` 和媒体读取转给 :8095 / :8090 上的服务。这些
不由本工具恢复。全新机器上只装本工具时，Flash-Next 两个入口、`/v1/messages`、`/v1/responses` 可用；预加载列表中缺失的
模型和上述路由要等对应片段与服务就位。

## 模板取值

模板里的 `@NAME@` 只做文本替换，渲染结果与线上文件逐字节比较。`@SERVING_ROOT@` 固定为 `<home>/llm-serving`；其余来自
`.local/serving-values.json`（`--values` 可换路径；路径里的 `~` 表示目标 home），缺省值对应只用本仓库重建的机器：

| 名称 | 缺省值 |
| --- | --- |
| `MODELS` | `~/models` |
| `STRATA_RELEASE` | `<仓库>/.local/src/strata-<native.lock.json 中的 12 位 commit>`（`tools/native.py strata` 的产物） |
| `STRATA_EXE` | `<STRATA_RELEASE>/build-metal-serving/strata` |
| `STRATA_VISION_EXE` | `<STRATA_RELEASE>/build-vision/bin/strata-vision` |
| `TRUSTED_PROXY` | `127.0.0.1`（网关前没有反向代理时；填一个 IPv4 地址，Caddyfile 中写成 `/32`） |

例（按本机实际情况填写）：

```json
{
  "MODELS": "~/models",
  "STRATA_RELEASE": "~/llm-serving/services/text/strata/releases/<release>",
  "STRATA_EXE": "~/llm-serving/services/text/strata/releases/<release>/build-metal-compact-native/strata",
  "STRATA_VISION_EXE": "~/llm-serving/services/text/strata/repo/build-vision/bin/strata-vision",
  "TRUSTED_PROXY": "<反向代理的局域网地址>"
}
```

路径取值只能含字母、数字和 `_ . / + -`：llama-swap 的 `cmd` 不加引号（与线上文件一致），带空格的路径会被拆开。

模型权重、Strata 构建产物、Homebrew 的 `llama-swap` 与 `caddy` 不由这个脚本处理：权重见 `models.required.json`，构建见
[strata.md](strata.md)，依赖见 `Brewfile`。
