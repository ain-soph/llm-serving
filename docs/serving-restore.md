# 恢复 ~/llm-serving 的部署

这个仓库就是部署目录：clone 到 `~/llm-serving` 后，GPU 锁脚本、llama-swap 启动脚本与全部配置、:8080 网关配置、服务代码、
LaunchAgent 模板都已在运行位置。每台机器不同的只有几样，由 `tools/install_serving.py` 从本机取值文件
`.local/serving-values.json`（不进库）生成；`tools/bootstrap.sh` 最后会运行它（`LLM_SERVING_SKIP_SERVING=1` 跳过）。

```sh
.venv/bin/python tools/install_serving.py --dry-run     # 只打印计划
.venv/bin/python tools/install_serving.py               # 创建缺失文件；与渲染结果不同的已有文件只报告 drift，不动
.venv/bin/python tools/install_serving.py --update      # 先备份到 ~/llm-serving/.backup/<相对路径>.<时间>-install-serving 再替换
.venv/bin/python tools/install_serving.py --check       # 不写；有缺失或差异时退出码 1（--diff 显示差异）
.venv/bin/python tools/install_serving.py --render DIR  # 只把模板和 env 文件渲染到 DIR（相对 home 的路径），用来和线上文件 diff
```

`--home DIR` 把目标换成 `DIR/llm-serving` 和 `DIR/Library/LaunchAgents`（测试用临时目录；这时仓库自己的文件会被复制过去，
正常部署里它们是“in place”）。安装不加载、不卸载、不重启任何服务，也不需要 GPU：写入或替换 LaunchAgent 后只打印要执行的
`launchctl bootstrap gui/$(id -u) <plist>`（替换时先 `launchctl bootout`）。`tools/doctor.py` 用同一套逻辑报告这些文件是否
与渲染结果一致、key 是否存在且为 0600（只打印 OK/WARN，不打印 key）。

## 清单

| 线上文件（相对 `~`） | 来源 | 方式 | 说明 |
| --- | --- | --- | --- |
| `llm-serving/bin/*`（gpu-lock、gpu-lock-run、gpu-pool、gpu-top、memguard、lsw-inflight、lsw-apply、gpu-lock-watchdog、archive-restore、model-archive、regress、health-check） | 仓库 | 原位 | 规则见 [gpu-lock.md](gpu-lock.md)；archive-restore 在入口启动时按清单从归档机拷回缺失的权重 |
| `llm-serving/platform/llama-swap/{config.yaml,conf.d/*.yaml}` | 仓库 | 原位 | 本机路径写成 `${env.LLM_SERVING}`、`${env.LLM_MODELS}`、`${env.STRATA_RELEASE}`；key 只以 `${env.LLAMA_API_KEY}` 出现 |
| `llm-serving/platform/llama-swap/{env.sh,serve.sh,start.sh,stop.sh}` | 仓库 | 原位 | `env.sh` 读 `.local/serving.env` 和 `bin/gpu-pool`；serve.sh 和 lsw-apply 都 source 它 |
| `llm-serving/platform/gateway/Caddyfile` | 仓库 | 原位 | `{$LLM_SERVING}`、`{$LLM_TRUSTED_PROXY}`，Caddy 用 `--envfile .local/serving.env` 读取 |
| `llm-serving/.local/serving.env` | 取值文件 | 生成（0600） | `LLM_SERVING`、`LLM_MODELS`、`STRATA_RELEASE`、`LLM_TRUSTED_PROXY`、`H3_VIDEO_SHARE_BASE`、`H3_VIDEO_FETCH_HOSTS` |
| `llm-serving/services/text/strata/configs/strata-{ad427,uncensored}-c262k-p2-compact.json` | `*.json.in` | 模板 | Strata 读 JSON 不展开环境变量，所以仍渲染 |
| `Library/LaunchAgents/com.local.{llama-server,llm-gateway,decisions,h3-video-files,health}.plist` | `platform/launchd/*.plist.in` | 模板 | 登录时启动 llama-swap、网关、`/v1/decisions` 翻译服务（:8095）、只读媒体服务（:8090）、每 10 分钟健康检查 |
| `com.local.gpu-lock-watchdog.plist` | `platform/launchd/` | 只 `--render` | 按用户决定（2026-11-15）停用，不写进 `~/Library/LaunchAgents` |
| `llm-serving/platform/llama-swap/.api_key` | 无 | 安装时生成 | 不存在时生成随机 key（mode 0600）；已存在绝不覆盖（只把权限收紧到 0600）；被忽略，`privacy_check` 拒收 |
| `llm-serving/AGENTS.md` 规则 1 | [docs/gpu-lock.md](gpu-lock.md) | 公开改写 | `AGENTS.md` 其余部分含本机和网络细节，不公开（`privacy_check` 拒收该文件名） |

不进库、也不由本工具恢复：Strata 的实验配置（`configs/` 下其他 JSON，不被路由）、`services/text/strata/config/`（Strata
自己写的设置）、模型权重、各服务的 fork checkout（OneJev 等，见各服务文档）。

## 取值

模板里的 `@NAME@` 只做文本替换，渲染结果与线上文件逐字节比较。`@SERVING_ROOT@` 固定为 `<home>/llm-serving`；其余来自
`.local/serving-values.json`（`--values` 可换路径；路径里的 `~` 表示目标 home），缺省值对应只用本仓库重建的机器：

| 名称 | 缺省值 |
| --- | --- |
| `MODELS` | `~/models` |
| `STRATA_RELEASE` | `<仓库>/.local/src/strata-<native.lock.json 中的 12 位 commit>`（`tools/native.py strata` 的产物） |
| `STRATA_EXE` | `<STRATA_RELEASE>/build-metal-serving/strata` |
| `STRATA_VISION_EXE` | `<STRATA_RELEASE>/build-vision/bin/strata-vision` |
| `TRUSTED_PROXY` | `127.0.0.1`（网关前没有反向代理时；填一个 IPv4 地址，Caddyfile 中写成 `/32`） |
| `SHARE_BASE` | `http://127.0.0.1:8080`（`/media/<id>` 链接的公开地址） |
| `FETCH_HOSTS` | 空（视频渲染可以下载输入文件的主机，逗号分隔） |

例（按本机实际情况填写）：

```json
{
  "MODELS": "~/models",
  "STRATA_RELEASE": "~/llm-serving/services/text/strata/releases/<release>",
  "STRATA_EXE": "~/llm-serving/services/text/strata/releases/<release>/build-metal-serving/strata",
  "STRATA_VISION_EXE": "~/llm-serving/services/text/strata/releases/<release>/build-vision/bin/strata-vision",
  "TRUSTED_PROXY": "<反向代理的局域网地址>",
  "SHARE_BASE": "https://<公开域名>",
  "FETCH_HOSTS": "<上传文件所在主机>"
}
```

路径取值只能含字母、数字和 `_ . / + -`：llama-swap 的 `cmd` 不加引号（与线上文件一致），带空格的路径会被拆开。

手动校验 llama-swap 配置：

```sh
source ~/llm-serving/platform/llama-swap/env.sh
LLAMA_API_KEY=x llama-swap -config ~/llm-serving/platform/llama-swap/config.yaml \
  -config-dir ~/llm-serving/platform/llama-swap/conf.d -validate
```

模型权重、Strata 构建产物、Homebrew 的 `llama-swap` 与 `caddy` 不由这个脚本处理：权重见 `models.required.json`，构建见
[strata.md](strata.md)，依赖见 `Brewfile`。
