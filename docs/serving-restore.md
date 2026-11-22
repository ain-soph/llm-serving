# 恢复 ~/llm-serving 的服务胶水

本地 worker（`agents/local-worker`）和各 agent 依赖部署目录 `~/llm-serving` 里的几样东西：GPU 锁脚本、llama-swap
启动脚本和 key、Strata 两个文本入口的路由与引擎配置。它们的公开版本放在本仓库 `serving/`，相对路径与 `~/llm-serving`
下一致，由 `tools/install_serving.py` 安装；`tools/bootstrap.sh` 最后会运行它（`LLM_SERVING_SKIP_SERVING=1` 跳过）。

```sh
.venv/bin/python tools/install_serving.py --dry-run   # 只打印计划
.venv/bin/python tools/install_serving.py             # 创建缺失文件；与仓库不同的已有文件只报告 drift，不动
.venv/bin/python tools/install_serving.py --update    # 先备份到 ~/llm-serving/.backup/<相对路径>.<时间>-install-serving 再替换
.venv/bin/python tools/install_serving.py --check     # 不写；有缺失或差异时退出码 1（--diff 显示差异）
.venv/bin/python tools/install_serving.py --render DIR  # 只把模板渲染到 DIR，用来和线上文件 diff
```

`--home DIR` 把目标换成 `DIR/llm-serving`（测试用临时目录）。安装不碰运行中的服务，不重启 llama-swap，也不需要 GPU。
`tools/doctor.py` 用同一套逻辑报告这些文件是否与仓库一致、key 是否存在且为 0600（只打印 OK/WARN，不打印 key）。

## 清单

| 线上文件（`~/llm-serving/` 下） | 仓库位置 | 方式 | 原因 |
| --- | --- | --- | --- |
| `bin/gpu-lock` | `serving/bin/gpu-lock` | 原样 | 多 agent 显存预约锁；路径都从 `$HOME` 推出，无本机内容。规则见 [gpu-lock.md](gpu-lock.md) |
| `bin/gpu-pool` | `serving/bin/gpu-pool` | 原样 | gpu-lock 和 serve.sh 用它从 `iogpu.wired_limit_mb` 算池大小；缺了 gpu-lock 退回固定 77000 |
| `bin/lsw-inflight` | `serving/bin/lsw-inflight` | 原样 | gpu-lock 暂停 llama-swap 前统计外部在途请求 |
| `platform/llama-swap/serve.sh` | `serving/platform/llama-swap/serve.sh` | 原样 | 读 `.api_key` 和 gpu-pool 启动 llama-swap（127.0.0.1:8081，网关 :8080 在前） |
| `platform/llama-swap/conf.d/strata.yaml` | `serving/platform/llama-swap/conf.d/strata.yaml` | 模板 | `qwen3.8-flash-next` / `-uncensored` 两个入口；含本机绝对路径 |
| `services/text/strata/configs/strata-ad427-c262k-p2-compact.json` | `serving/services/text/strata/configs/` 同名 | 模板 | normal 变体引擎参数；含模型和构建路径 |
| `services/text/strata/configs/strata-uncensored-c262k-p2-compact.json` | 同上 | 模板 | uncensored 变体引擎参数 |
| `platform/llama-swap/.api_key` | 无 | 安装时生成 | 不存在时生成随机 key（mode 0600）；已存在绝不覆盖（只把权限收紧到 0600）；被 `.gitignore` 忽略，`privacy_check` 拒收 |
| 本机取值 | `.local/serving-values.json` | 私有 | 模型、Strata 构建路径；`.local/` 和文件名 `serving-values.json` 都被忽略且被 `privacy_check` 拒收 |
| `AGENTS.md` 规则 1 | [docs/gpu-lock.md](gpu-lock.md) | 公开改写 | `AGENTS.md` 其余部分含本机和网络细节，不公开（`privacy_check` 拒收该文件名） |

不在本次范围、仍只在本机的：llama-swap 主配置 `config.yaml` 及其他 `conf.d` 片段（仓库已有 `templates/` 版本，由
`tools/configure.py` 按仓库布局渲染）、`bin/lsw-apply`（会重启生产服务）、`bin/gpu-lock-watchdog`（LaunchAgent 已停用）、
Strata 的实验配置（`configs/` 下其他 JSON）和 `services/text/strata/config/`（Strata 自己写的设置）。

## 模板取值

模板里的 `@NAME@` 只做文本替换，渲染结果与线上文件逐字节比较。`@SERVING_ROOT@` 固定为 `<home>/llm-serving`；其余来自
`.local/serving-values.json`（`--values` 可换路径；`~` 表示目标 home），缺省值对应只用本仓库重建的机器：

| 名称 | 缺省值 |
| --- | --- |
| `MODELS` | `~/models` |
| `STRATA_RELEASE` | `<仓库>/.local/src/strata-<native.lock.json 中的 12 位 commit>`（`tools/native.py strata` 的产物） |
| `STRATA_EXE` | `<STRATA_RELEASE>/build-metal-serving/strata` |
| `STRATA_VISION_EXE` | `<STRATA_RELEASE>/build-vision/bin/strata-vision` |

例（路径按本机实际情况填写）：

```json
{
  "MODELS": "~/models",
  "STRATA_RELEASE": "~/llm-serving/services/text/strata/releases/<release>",
  "STRATA_EXE": "~/llm-serving/services/text/strata/releases/<release>/build-metal-compact-native/strata",
  "STRATA_VISION_EXE": "~/llm-serving/services/text/strata/repo/build-vision/bin/strata-vision"
}
```

取值只能含字母、数字和 `_ . / + -`：llama-swap 的 `cmd` 不加引号（与线上文件一致），带空格的路径会被拆开。

模型权重、Strata 构建产物和 LaunchAgent 不由这个脚本处理：权重见 `models.required.json`，构建见 [strata.md](strata.md)。
