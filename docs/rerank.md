# Qwen3 重排部署

当前三个模型 ID 保持兼容，后端由 llama.cpp GGUF 改为 MLX。服务代码在
[`rerank/server.py`](../rerank/server.py)，没有修改 MLX 或 MLX-LM 的库源码。
所有进程使用部署仓库统一的 Python 3.15 环境：MLX 0.32.4、MLX-LM 0.32.2、
Transformers 5.23.0，具体依赖由 `uv.lock` 固定。

| ID | 模型目录（相对于模型根目录） | 权重／激活 | 运行文件逻辑大小 |
| --- | --- | --- | ---: |
| `qwen3-reranker-0.6b` | `qwen3-reranker/mlx-0.6b-bf16` | BF16／BF16 | 1.203 GB |
| `qwen3-reranker-4b` | `qwen3-reranker/mlx-4b-bf16` | BF16／BF16 | 8.055 GB |
| `qwen3-reranker-8b` | `qwen3-reranker/mlx-8b-q8_0` | Q8_0 等价 MLX affine 8-bit、group 32、FP16 scale/bias／FP16 | 9.228 GB |

迁移应复制这三个已验证目录，包括配置、tokenizer、索引及全部 safetensors 分片。
`models.required.json` 检查必需文件，避免空目录被误判为完整模型。权重不上传 GitHub。

## 服务层优化

- 把同一查询下的文档右侧补齐后批量计算，默认最多 8192 个补齐后 batch token。
  这是批量预算，不是上下文长度限制；单篇更长的文档仍会独立计算，可能提高峰值内存。
- 使用官方提示格式，取每篇最后一个真实 token 的 yes/no logits，计算 `sigmoid(yes-no)`。
  只保留输出头的两行；8B 的其余独立输出头加载后释放，约少 0.6 GB。
- 请求间保留的 MLX 缓存限制为 256 MiB，单 worker 串行计算。
- 没有生成阶段，也不复用生成 KV cache；每次请求完整前向计算。

这些是自写服务和权重准备的改动。0.6B/4B 同时从 Q8_0 改为 BF16，
因此与旧服务的速度差包含后端、批处理及精度选择的共同影响，不能归因于某一项修改。

## 8B 转换与代价

[`q8_0_to_mlx.py`](../rerank/q8_0_to_mlx.py) 将原始浮点权重按 Q8_0 的
32 元素分组和 FP16 scale 舍入，编码为 MLX affine 8-bit。
额外的 `bias = -128 * scale` 使反量化权重对应 Q8_0；它不是把 GGUF 直接改名。
保存原始浮点来源、固定转换代码和参数，才能重新生成同一套准备权重。

```sh
.venv/bin/python rerank/q8_0_to_mlx.py "$SOURCE_HF_DIR" "$OUTPUT_MLX_DIR" --act-dtype float16
```

转换会加载完整浮点来源及中间张量，应先预留足够 GPU 内存，输出到新目录再验证。
服务环境已包含所需依赖。默认 MLX 8-bit 的量化规则不同，不能替代这里的转换。
相对旧 8B GGUF 的 8.047 GB，当前目录为 9.228 GB，增加 1.181 GB（14.7%）；
其中包含 affine bias 和 tokenizer/配置等格式差异。
0.6B/4B 则分别由旧 GGUF 的 0.639/4.280 GB 增至 1.203/8.055 GB。

## 检查与复测

先预留 GPU 内存，再在测试端口启动单个模型：

```sh
.venv/bin/python rerank/server.py --port 18189 \
  --model-id qwen3-reranker-8b --model "$MODELS/qwen3-reranker/mlx-8b-q8_0"
```

另一个终端运行检查；`--benchmark` 仅在独占 GPU 时使用：

```sh
.venv/bin/python tools/check_rerank.py --size 8b --benchmark
```

检查包括 health、模型 ID、三个公开查询共 11 对文档对照官方 BF16 分数及排序，
字符串/对象文档、top_n、return_documents、三个重排 URL，以及空文档的 HTTP 400。
基准沿用旧测试的 20 篇文档、并发 1；先预热一次再测三次。
工具使用响应中实际 prompt token 计算端到端输入吞吐，历史报告约 400 token/篇的估算值
单独标注，不与实测 token 数混算。无生成 decode，tok/s 也不是单独算子吞吐。

已保存的官方分数及公开输入在 [`rerank-reference.json`](../tests/fixtures/rerank-reference.json)。
小样本数值和排序一致不代表所有查询准确率相同。历史结果见
[性能与资源报告](performance.md)；新环境检查结果见 [验证记录](rerank-validation.json)。

## Python 3.15.2 独占复测

2026-11-16，同一 M5 Ultra、96 GiB、macOS 27；每档单独启动服务，
20 文档实际共 9950 prompt token，c1，预热一次后三次均值：

| 模型 | 平均耗时 | HTTP 端到端输入吞吐 | 峰值 RSS | 峰值 wired 增量 | 官方 BF16 最大分数差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6b | 0.137 s | 72466 tok/s | 1.657 GB | +2772 MiB | 0.001223 |
| 4b | 0.847 s | 11753 tok/s | 8.453 GB | +8694 MiB | 0.010256 |
| 8b | 1.793 s | 5548 tok/s | 9.680 GB | +9869 MiB | 0.004420 |

三个公开查询的排序全部一致，全部 API 检查通过。没有生成 decode，单独内核 prefill 未测。
RSS 与系统 wired 每 0.3 秒采样，覆盖加载、正确性检查、预热及测速全过程；
wired 以各模型启动前为基线，包含系统噪声，不能与 RSS 相加。
路由预留调整为 3500/10000/11000 MiB，给本轮约 2772/8694/9869 MiB 的峰值留余量；
这些是迁移配置的新值，当前生产路由未切换。更长文档、不同 batch 或共存组合仍需重新测量。

历史报告按每篇约 400 token 估算吞吐，实际本轮 tokenizer 返回 9950 token，
因此历史 61000/9500/4400 tok/s 与新值的差不能当作额外提速；
0.137/0.847/1.793 s 的延迟与历史 0.13/0.84/1.80 s 基本一致。
旧 llama.cpp 基线未重新运行，未从这些结果推断 Python 3.15 比其他版本快。
本轮只验证直接 HTTP 服务，没有做新机器冷恢复或完整网关推理验收。
