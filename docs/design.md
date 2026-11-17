# 部署与维护设计

## 路径

源代码、模型权重、运行状态分别管理。源码根目录由工具自己的位置推导；模型位置由 `--models` 指定；状态目录与生成配置由 `--state`、`--output` 指定。仓库内只有模板，绝对路径只出现在本机生成的 `.local/config` 中。

LaunchAgent 使用解释器与脚本的完整路径和参数数组，由 `plistlib` 写入。不同用户名、代码位置、空格和非英文目录名不需要手动替换。环境变量由统一的服务启动器传入，不依赖交互 shell 或 Homebrew 的固定安装前缀。

## 一个 uv 环境

部署的 Python 依赖统一声明在 `pyproject.toml`，解析结果保存在 `uv.lock`。部署使用 **Homebrew Python 3.15**，`requires-python` 限定 3.15 系列，`.python-version` 记录该系列；所有服务进程都使用根目录 `.venv/bin/python`。已在 3.15.2 验证依赖安装、导入和相关 CPU 基础调用，未发现需要降到 3.12 的阻碍；完整模型推理回归仍是生产切换前的必要步骤。独立进程仍各自管理模型和 GPU 内存，共享 venv 只统一解释器及库版本。

不要把临时评测、训练和模型转换工具任意装入部署环境。若未来出现真正的版本冲突，应先解决依赖约束；确实无法兼容的工具可另设非运行时环境。

Homebrew 管理 Python、uv 和原生工具，uv 管理项目虚拟环境与 Python 包。Brewfile 是工具清单，**不是永久锁定历史 bottle 的锁文件**；`native.lock.json` 的 `observed_tools` 记录已使用的版本。Homebrew 更新后应重新检查接口兼容性。源码固定 commit 也不代表跨 SDK、编译器的二进制逐字节可复现。

参考：[uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

## 原生构建

第三方引擎不复制进主仓库；改动留在相应 fork，部署引用不可变的完整 commit，分支 `serving` 用于维护。每次构建使用独立版本目录，保留旧目录即可回退。

`tools/native.py` 使用 Homebrew CMake/Ninja 和 Apple clang，从固定源码构建 Strata、视觉编码器、H3 视频、NeMo Speech。Metal C++ 头文件单独下载并验证 SHA-256；嵌套源码按对应项目固定的版本准备。图片引擎和 Whisper 从 `uv.lock` 中的各自 fork commit 安装。

每个构建树生成 `serving-build.json`，记录 commit、架构、macOS/SDK/编译器、构建参数、可执行文件 SHA-256 和动态库链接。该文件是本机产物，不提交公开仓库。未来发布二进制包时，应以引擎 commit、架构、最低系统版本、SDK/工具链和构建参数标记产物，并一并提供依赖库、shader、许可证和校验值。

## 本机凭证与可选外部入口

当前服务使用的凭证是本机 llama-swap 的 Bearer API key。视频异步任务向本机路由器发送 hold 请求，防止任务未完成时被卸载；回调也使用这个凭证。视频分享签名由本机 key 派生。

公网域名、TLS、聊天前端和客户端配置属于可选访问层。服务不需要在另一台机器配置任何东西才能提供模型 API。媒体 URL 的公共前缀用 `--public-base` 指定，远程参考媒体的下载主机用 `--fetch-host` 明确配置，可信反向代理用 `--trusted-proxy` 指定；默认没有个人域名、外部下载白名单或可信代理。

普通 API 与媒体读取都经过 Bearer 鉴权。只有签名的分享链接也不会自动绕过网关的 Bearer 检查；需要无请求头播放器时，可选入口须单独处理认证，这是入口功能，不是推理依赖。

## 系统配置

安装器可重复生成和安装用户 LaunchAgent，覆盖前保留上一个文件。`--activate` 才立即启动；不加则在下次登录时启动。GPU 内存上限由显式的 root LaunchDaemon 选项持久化；不把某台机器的数值写成所有 Mac 的默认值。

参考：[Apple launchd 配置](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)。

## macOS 性能优化

优先把优化实现为原引擎里的 macOS/Metal 执行路径，在当前 fork 上维护，成熟后再考虑提交上游。主部署仓库负责版本、配置、编排，不承担多份引擎代码的长期复制。

能力分派应依次检查 OS/API 可用性、Metal GPU family、内核能否编译/执行和结果验证，失败回到基线。设备型号可用来选择已经实测的 tile、缓存预算等参数；不能用型号字符串代替能力检测。M5 Ultra 的内存量和并发配置应与算子实现分开。

已有的部分基础：

- H3 视频按 M5 设备选择 TensorOps，编译失败时回到普通 Metal/MPS 路径；其中型号字符串判定还可以逐步替换成能力检测。
- H3 图片包含整数 TensorOps 的实际小规模执行探测和 portable 路径。
- Strata Metal 已有设备相关的 kernel/tile 与内存预算选择。

这些机制并不等于所有新优化都已经在旧机型验证。部署默认值只对当前测量配置负责；另一台 Mac 上仍需要验证正确性、内存峰值与性能。精确优化与稀疏/裁剪等近似选项分开，硬件识别不应自动启用近似算法。

若上游不接收特定优化，继续在现有 fork 维护 macOS 性能分支即可，无需先删除其他平台实现。

参考：[Metal 设备能力检测](https://developer.apple.com/documentation/metal/device-inspection)。
