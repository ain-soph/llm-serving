# Strata Metal 部署

主要文本模型通过 Strata Metal 运行。当前路由包含 `qwen3.8-flash-next` 和 `qwen3.8-flash-next-uncensored`。两个入口均使用 2 个槽位，每槽 262144 token，启用 Metal 省内存双槽，GPU 预约均为 69,500 MiB。这些参数来自目标 M5 Ultra 96 GiB 部署，不代表所有 Apple Silicon 都适用。

## 源码与优化

源码在自己的 [ain-soph/Strata](https://github.com/ain-soph/Strata/tree/serving) fork 的 `serving` 分支维护。部署实际固定到 [4a6415f4e0dad46d096a22b2de5e0223348922a7](https://github.com/ain-soph/Strata/commit/4a6415f4e0dad46d096a22b2de5e0223348922a7)（2026-11-23 20:38 起；上游 main `73d4f4196`（v0.2.20+13）+ 我们的提交 + Metal down+combine 合并 + Metal 并发 MTP；前两版的 `serving` 头留有标签 `serving-pre-cmtp-20261123`（`64091c8e2`）和 `serving-pre-decodeopt-20261123`（`53a6becb8`），rebase 前的 `82d63e19e` 是 `serving-pre-rebase-20261122`），由主仓库的 `native.lock.json` 记录，不会在每次安装时跟随分支移动。

该快照包含 Metal `--prefill auto` 修复。引擎 JSON 的 `parallel: 2` 生成 `--batch 2`，`STRATA_METAL_COMPACT_BATCH=1` 让预填与解码共用两套序列状态。MTP 草稿头为 Q2_0；`STRATA_METAL_BATCH_MTP=1` 让两路同时 decode 时每个槽也带 1 个草稿，在共享批窗口里验证（上游在 Metal 上关掉了 `--batch-mtp`；去掉这个 env 就只有单路请求做草稿）。KV 为 INT8。`STRATA_METAL_PROMPT_CHUNK=2048`：Metal 的 prompt 每块读 2048 行，有意偏离上游默认的 1024 行。长提示 prefill 快约 13%，输出逐位相同，wired 多约 1.9 GiB；去掉这个 env 就回到 1024。其余按上游默认：图片由 Strata 自有的 Qwen 视觉塔处理；draft-cost 表开启，每台机器从前约 100 个真实请求被动学到（存于 `XDG_CONFIG_HOME/strata/draft-costs/`）；GPU keep-alive 开启。对话缓存不设 `--conversation-cache-mib`，由引擎在模型加载后按空闲内存自动定大小（可用内存减 2560 MiB 后的 40%，最多 8192 MiB，以 256 取整；本机常规加载为 7680–8192 MiB，引擎日志会写出取值）。不设大小时，服务端不加 `--conversation-cache-min-tokens 1024`，短会话也会停放。从内存缓存挤出的会话会写进权重目录下的 `conversation-spill/`（SSD 层，上限为 min(32 GiB, 空闲盘的 ¼)），续问时再读回。后续经过验证的 macOS 性能改动应先提交到引擎 fork，再更新部署锁定的 commit。本仓库不复制第三方引擎代码。

Strata 自带 `terminal/serve/server.py`，主仓库直接复用它提供 HTTP 接口；无需再写一个 Python 包装服务器。其 Python 运行依赖纳入主仓库的统一 uv 环境。

## 文件对应关系

| 文件 | 内容 |
| --- | --- |
| [`native.lock.json`](../native.lock.json) | Strata fork commit、Metal C++ 头文件校验值（ggml 与视觉代码随 Strata 源码自带） |
| [`tools/native.py`](../tools/native.py) | 拉取固定源码，构建 Strata Metal 与视觉编码器 |
| [`platform/llama-swap/conf.d/strata.yaml`](../platform/llama-swap/conf.d/strata.yaml) | llama-swap 的两个文本模型 ID、HTTP 启动命令和路由容量 |
| [`services/text/strata/configs/strata-ad427-c262k-p2-compact.json.in`](../services/text/strata/configs/strata-ad427-c262k-p2-compact.json.in) | AtomicChat AD-4.27 的引擎参数与视觉配置 |
| [`services/text/strata/configs/strata-uncensored-c262k-p2-compact.json.in`](../services/text/strata/configs/strata-uncensored-c262k-p2-compact.json.in) | Uncensored AD-4.27 的引擎参数与视觉配置 |
| [`models.required.json`](../models.required.json) | 需要另行迁移的模型目录与权重布局 |
| [`tools/install_serving.py`](../tools/install_serving.py) | 由本机取值渲染两个引擎 JSON 和 `.local/serving.env`（`STRATA_RELEASE` 等） |

## 构建与启动路径

完成 `tools/bootstrap.sh` 后，可以只构建 Strata：

```sh
.venv/bin/python tools/native.py strata
```

`tools/native.py all` 也包含这一步。构建脚本准备 Metal C++ 头文件，用 Strata 自带的 ggml 构建 `engine/metal` 与 `terminal/tools/vision`，生成：

```text
.local/src/strata-4a6415f4e0da/
├── terminal/serve/server.py
├── build-metal-serving/strata
├── build-vision/bin/strata-vision
└── serving-build.json
```

这些是本机构建产物，被 Git 忽略。`platform/llama-swap/conf.d/strata.yaml` 直接在线上使用，release 路径写成 `${env.STRATA_RELEASE}`（来自 `.local/serving.env`）；两个引擎 JSON 由 `*.json.in` 渲染到 `services/text/strata/configs/`。路由器按请求中的模型 ID 启动 HTTP 服务，再由 HTTP 服务启动 Metal 引擎。

模型 pack、Metal GGUF、PLE、MTP 和视觉投影权重独立存储。迁移需要保留经过准备的目录布局，单独下载原始模型 checkpoint 不能替代现有转换产物。

本仓库已校验统一 Python 依赖及路由配置。全新系统上的源码重编译与完整推理恢复尚未演练；这份说明不将构建脚本误标为已完成冷安装验证。
