# Perplexity Decider：权重准备与服务

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

未包含：Drex v1.5（需要 nace-ai 的 llama.cpp 分支）和把 OpenAI `POST /v1/decisions` 翻译成 SystemOne 的服务。
