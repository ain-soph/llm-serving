# llm-serving

面向 Apple Silicon 的本地模型接口与部署工具。当前参数来自 **M5 Ultra、96 GiB 统一内存**的单机部署，包含文本、嵌入、重排、语音识别、图片和视频。

模型服务独立运行；不依赖另一台电脑、外部推理 API、聊天前端、MCP 或公网反向代理。

```text
客户端 → Caddy :8080 → llama-swap 127.0.0.1:8081 → 按需加载模型
                 └→ 只读媒体服务 127.0.0.1:8090
```

## 安装与迁移

需要 Apple Silicon、macOS 及适用的 Apple Command Line Tools。先安装 [Homebrew](https://brew.sh)，再执行：

```sh
git clone https://github.com/ain-soph/llm-serving.git
cd llm-serving
./tools/bootstrap.sh
.venv/bin/python tools/native.py all
```

`Brewfile` 统一安装 uv、Caddy、llama-swap、llama.cpp、FFmpeg、CMake、Ninja。uv 管理 Python 3.12 和项目唯一的 `.venv`；所有 Python 服务使用同一个解释器。`uv.lock` 固定 Python 依赖，包括图片引擎 fork 的完整 commit。

原生引擎从 `native.lock.json` 的完整 commit 构建到 `.local/src/`。编译后的可执行文件、动态库、Metal shader 和构建信息留在同一个本地构建树，不能只搬一个可执行文件。

将已验证的模型目录复制到新机器上的任意位置，再生成本机配置：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models"
.venv/bin/python tools/doctor.py
.venv/bin/python tools/install.py --activate
```

`configure.py` 只生成本机文件，不启动服务。默认监听 `127.0.0.1:8080`；提供局域网接口时添加 `--listen 0.0.0.0`。可用 `--state`、`--output` 自定义运行数据和配置目录。代码可以放在任意目录，路径含空格也支持。

当前完整聊天配置要求 GPU 池至少 **71000 MiB**。默认从当前 `iogpu.wired_limit_mb` 减去系统与常驻嵌入预留的 4872 MiB；系统上限为 0 时要求显式给出已验证的 `--pool-mb`。工具不会因为识别出一台 Mac 就自动提高内存上限，也不会把这台机器的上下文/并发参数缩放成未经验证的其他机型参数。

若迁移机器需要持久化 GPU 内存上限，可以显式指定，例如针对已经验证过的 96 GiB 机器：

```sh
.venv/bin/python tools/configure.py --models "$HOME/models" --pool-mb 83192
.venv/bin/python tools/install.py --wired-limit-mb 88064 --activate
```

此选项通过 `sudo` 安装 root 拥有的 LaunchDaemon；普通模型服务由当前用户的 LaunchAgent 运行，在用户登录后启动。若旧部署仍占用 8080/8081/8090，应先停掉旧部署，再激活新服务。

## 模型与状态

- 模型目录完全独立，必要布局见 `models.required.json`。转换过的 Strata pack、INT8 视频权重和页对齐图片权重应一并迁移；原始 Hub 权重不一定能直接替代这些文件。
- 运行数据默认在 `~/Library/Application Support/llm-serving`：API key、活动数据库、日志、图片与视频。模型权重和运行数据都不提交到 Git。
- 迁移时可以保留旧 API key，放到新状态目录的 `api_key` 文件，权限设为 `0600`；否则配置工具会生成一个新凭证，供客户端使用。不要将密钥放到命令行参数或公开配置中。
- 有权重文件后，推理不需要外部 API key 或下载 token。个别模型首次下载可能需要模型托管站点的授权；这属于准备阶段。

## 验证范围

已在开发机器完成统一依赖安装与导入、配置生成、llama-swap/Caddy 配置校验、FFmpeg 编码器检查，以及安装路径和隐私扫描测试。当前生产服务仍使用原来的目录和环境。

尚未在干净系统完成全套原生重编译和端到端推理恢复演练。本仓库提供可审查的迁移工具和当前配置，首次切换前仍需运行真实 API 冒烟检查。`doctor.py` 不加载模型，不替代正确性/性能验证。

更多说明：[设计与硬件优化](docs/design.md)、[发布与隐私](docs/privacy.md)。
