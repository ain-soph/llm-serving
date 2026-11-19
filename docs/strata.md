# Strata Metal 部署

主要文本模型通过 Strata Metal 运行。当前路由包含 `qwen3.8-flash-next` 和 `qwen3.8-flash-next-uncensored`；两者各使用 262144 token 上下文、1 个引擎槽位和 MTP，GPU 预约分别为 63,900 和 65,200 MiB。这些参数来自目标 M5 Ultra 96 GiB 部署，不代表所有 Apple Silicon 都适用。

## 源码与优化

源码在自己的 [ain-soph/Strata](https://github.com/ain-soph/Strata/tree/serving) fork 的 `serving` 分支维护。部署实际固定到 [004d0d1197292b7819c9b3ed52c5bb8f29c18514](https://github.com/ain-soph/Strata/commit/004d0d1197292b7819c9b3ed52c5bb8f29c18514)，由主仓库的 `native.lock.json` 记录，不会在每次安装时跟随分支移动。

该快照包含 Metal `--prefill auto` 的修复。后续经过验证的 macOS 性能改动应先提交到引擎 fork，再更新部署锁定的 commit。本仓库不复制第三方引擎代码。

Strata 自带 `serve/server.py`，主仓库直接复用它提供 HTTP 接口；无需再写一个 Python 包装服务器。其 Python 运行依赖纳入主仓库的统一 uv 环境。

## 文件对应关系

| 文件 | 内容 |
| --- | --- |
| [`native.lock.json`](../native.lock.json) | Strata fork commit、所需 ggml/llama.cpp commit、Metal C++ 头文件校验值 |
| [`tools/native.py`](../tools/native.py) | 拉取固定源码，构建 Strata Metal 与视觉编码器 |
| [`templates/conf.d/strata.yaml`](../templates/conf.d/strata.yaml) | llama-swap 的两个文本模型 ID、HTTP 启动命令和路由容量 |
| [`templates/strata/strata-ad427-c262k-p1.json`](../templates/strata/strata-ad427-c262k-p1.json) | AtomicChat AD-4.27 的引擎参数与视觉配置 |
| [`templates/strata/strata-uncensored-c262k-p1.json`](../templates/strata/strata-uncensored-c262k-p1.json) | Uncensored AD-4.27 的引擎参数与视觉配置 |
| [`models.required.json`](../models.required.json) | 需要另行迁移的模型目录与权重布局 |
| [`tools/configure.py`](../tools/configure.py) | 将机器无关模板渲染成本机配置 |

## 构建与启动路径

完成 `tools/bootstrap.sh` 后，可以只构建 Strata：

```sh
.venv/bin/python tools/native.py strata
```

`tools/native.py all` 也包含这一步。构建脚本准备 Metal C++ 头文件及固定版本的 ggml，再生成：

```text
.local/src/strata-004d0d119729/
├── serve/server.py
├── build-metal-serving/strata
├── build-vision/bin/strata-vision
└── serving-build.json
```

这些是本机构建产物，被 Git 忽略。生成配置后，`templates/conf.d/strata.yaml` 对应 `.local/config/conf.d/strata.yaml`；两个引擎 JSON 对应 `.local/config/strata/`。路由器按请求中的模型 ID 启动 HTTP 服务，再由 HTTP 服务启动 Metal 引擎。

模型 pack、Metal GGUF、PLE、MTP 和视觉投影权重独立存储。迁移需要保留经过准备的目录布局，单独下载原始模型 checkpoint 不能替代现有转换产物。

本仓库已校验统一 Python 依赖及生成的路由配置。全新系统上的源码重编译与完整推理恢复尚未演练；这份说明不将构建脚本误标为已完成冷安装验证。
