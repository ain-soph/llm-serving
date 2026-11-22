# Strata Metal 部署

主要文本模型通过 Strata Metal 运行。当前路由包含 `qwen3.8-flash-next` 和 `qwen3.8-flash-next-uncensored`。两个入口均使用 2 个槽位，每槽 262144 token，启用 Metal 省内存双槽，GPU 预约均为 69,500 MiB。这些参数来自目标 M5 Ultra 96 GiB 部署，不代表所有 Apple Silicon 都适用。

## 源码与优化

源码在自己的 [ain-soph/Strata](https://github.com/ain-soph/Strata/tree/serving) fork 的 `serving` 分支维护。部署实际固定到 [82d63e19e55cf674e7ed9487e0606c36b7ecbc91](https://github.com/ain-soph/Strata/commit/82d63e19e55cf674e7ed9487e0606c36b7ecbc91)（2026-11-22 起；上游 main `6e38f588a` + 我们的提交，rebase 后覆盖 `serving`，旧版本留有标签 `serving-pre-cleanup-20261121`），由主仓库的 `native.lock.json` 记录，不会在每次安装时跟随分支移动。

该快照包含 Metal `--prefill auto` 修复。引擎 JSON 的 `parallel: 2` 生成 `--batch 2`，`STRATA_METAL_COMPACT_BATCH=1` 让预填与解码共用两套序列状态。单请求使用 Q2_0 MTP，KV 为 INT8。其余按上游默认：图片由 Strata 自有的 Qwen 视觉塔处理；draft-cost 表开启，每台机器从前约 100 个真实请求被动学到（存于 `XDG_CONFIG_HOME/strata/draft-costs/`）；GPU keep-alive 开启。上游的 SSD 对话溢出层会在权重目录下建 `conversation-spill/`，但本配置下从不写盘（会话指纹要求 `--mtp` 目录有 `dense.bin`，Metal MTP 目录没有）。后续经过验证的 macOS 性能改动应先提交到引擎 fork，再更新部署锁定的 commit。本仓库不复制第三方引擎代码。

Strata 自带 `terminal/serve/server.py`，主仓库直接复用它提供 HTTP 接口；无需再写一个 Python 包装服务器。其 Python 运行依赖纳入主仓库的统一 uv 环境。

## 文件对应关系

| 文件 | 内容 |
| --- | --- |
| [`native.lock.json`](../native.lock.json) | Strata fork commit、Metal C++ 头文件校验值（ggml 与视觉代码随 Strata 源码自带） |
| [`tools/native.py`](../tools/native.py) | 拉取固定源码，构建 Strata Metal 与视觉编码器 |
| [`templates/conf.d/strata.yaml`](../templates/conf.d/strata.yaml) | llama-swap 的两个文本模型 ID、HTTP 启动命令和路由容量 |
| [`templates/strata/strata-ad427-c262k-p2-compact.json`](../templates/strata/strata-ad427-c262k-p2-compact.json) | AtomicChat AD-4.27 的引擎参数与视觉配置 |
| [`templates/strata/strata-uncensored-c262k-p2-compact.json`](../templates/strata/strata-uncensored-c262k-p2-compact.json) | Uncensored AD-4.27 的引擎参数与视觉配置 |
| [`models.required.json`](../models.required.json) | 需要另行迁移的模型目录与权重布局 |
| [`tools/configure.py`](../tools/configure.py) | 将机器无关模板渲染成本机配置 |

## 构建与启动路径

完成 `tools/bootstrap.sh` 后，可以只构建 Strata：

```sh
.venv/bin/python tools/native.py strata
```

`tools/native.py all` 也包含这一步。构建脚本准备 Metal C++ 头文件，用 Strata 自带的 ggml 构建 `engine/metal` 与 `terminal/tools/vision`，生成：

```text
.local/src/strata-82d63e19e55c/
├── terminal/serve/server.py
├── build-metal-serving/strata
├── build-vision/bin/strata-vision
└── serving-build.json
```

这些是本机构建产物，被 Git 忽略。生成配置后，`templates/conf.d/strata.yaml` 对应 `.local/config/conf.d/strata.yaml`；两个引擎 JSON 对应 `.local/config/strata/`。路由器按请求中的模型 ID 启动 HTTP 服务，再由 HTTP 服务启动 Metal 引擎。

模型 pack、Metal GGUF、PLE、MTP 和视觉投影权重独立存储。迁移需要保留经过准备的目录布局，单独下载原始模型 checkpoint 不能替代现有转换产物。

本仓库已校验统一 Python 依赖及生成的路由配置。全新系统上的源码重编译与完整推理恢复尚未演练；这份说明不将构建脚本误标为已完成冷安装验证。
