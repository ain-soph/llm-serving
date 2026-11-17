# 决策模型：Perplexity Decider 与 Base Labs elephant-1

服务 `decisions/decider/server.py` 是 Perplexity Decider Flash v1.3（Apache-2.0；Qwen3.5-4B 骨干）的 MLX 移植
（`decisions/decider/decider_mlx.py`），提供 SystemOne 协议 `POST /v1/systemone`（llama-swap 按 `model` 路由）。
提示词、预处理、概率校准用官方 `pplx_decider` 包，它随检查点发布在 `source/` 目录里；服务从那里导入，不安装
（它固定的 torch / transformers 版本比本仓库旧）。路由模板：`templates/conf.d/decisions.yaml`。

| 模型 ID | 输入 | 精度 |
| --- | --- | --- |
| `pplx-decider-flash` | 文本 | FP16 权重与激活（从官方 BF16 逐张量转换） |
| `pplx-decider-flash-multimodal` | 文本 + 图片（base64 data URL） | 同上，含视觉塔 |
| `pplx-decider-flash-int8` | 文本 | 语言模型投影 W8A8，加载时量化（[说明](w8a8.md)） |

## 准备步骤

```sh
hf download perplexity-ai/pplx-decider-flash-v1.3 --local-dir "$HOME/models/pplx-decider-flash-v1.3"
```

权重原样使用，不需要转换（9.6 GB，含 `source/`）。


## Base Labs elephant-1

服务 `decisions/jev/server.py` 是 Base Labs elephant-1（MIT；Qwen3.5-9B 后训练，块内双向注意力 + 字母槽读出）的 MLX 移植
（`decisions/jev/jev_mlx.py`，骨干用 mlx-vlm 的 qwen3_5），SystemOne `POST /v1/systemone`。提示词渲染、前缀／后缀切分、答案槽、
校准和答案组装直接调用官方 OneJev `qev` 的 CPU 代码。路由模板：`templates/conf.d/decisions-baselabs.yaml`。

| 模型 ID | 精度 |
| --- | --- |
| `elephant-1` | 解码层线性层 8-bit（MLX affine，组 32），BF16 激活；加载时量化 |
| `elephant-1-int8` | W8A8；加载时量化 |

词表和 LM 头不进 GPU（从映射的 BF16 文件读行）。文本输入；state + 一个问题最多 16,384 token。

### 准备步骤

```sh
hf download baselabs/elephant-1 --local-dir "$HOME/models/elephant-1"      # BF16，18.8 GB
# 官方 OneJev（Apache-2.0）固定到 81ce62f，打上检查点自带的 serve.patch（与官方 setup.sh 相同）
git clone https://github.com/OmniJev/OneJev decisions/OneJev
git -C decisions/OneJev checkout 81ce62f
git -C decisions/OneJev apply "$HOME/models/elephant-1/serve.patch"
```

OneJev 不随本仓库分发（`decisions/OneJev/` 在 `.gitignore` 里）；`tools/doctor.py` 会检查它是否就位。
官方运行时只支持 CUDA，这里只用它的 CPU 部分，不需要安装它的依赖。

## 未包含

Drex v1.5（需要 nace-ai 的 llama.cpp 分支）和把 OpenAI `POST /v1/decisions` 翻译成 SystemOne 的服务。
