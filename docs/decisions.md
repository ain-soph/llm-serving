# 决策模型：Perplexity Decider 与 Base Labs elephant-1 / mammoth-1

服务 `decisions/decider/server.py` 是 Perplexity Decider Flash v1.3（Apache-2.0；Qwen3.5-4B 骨干）的 MLX 移植
（`decisions/decider/decider_mlx.py`），提供 SystemOne 协议 `POST /v1/systemone`（llama-swap 按 `model` 路由）。
提示词、预处理、概率校准用官方 `pplx_decider` 包，它随检查点发布在 `source/` 目录里；服务从那里导入，不安装
（它固定的 torch / transformers 版本比本仓库旧）。路由模板：`templates/conf.d/decisions.yaml`。

| 模型 ID | 输入 | 精度 |
| --- | --- | --- |
| `pplx-decider-flash` | 文本 | FP16 权重与激活（从官方 BF16 逐张量转换） |
| `pplx-decider-flash-multimodal` | 文本 + 图片（base64 data URL） | 同上，含视觉塔 |
| `pplx-decider-flash-int8` | 文本 | 语言模型投影 W8A8，加载时量化；`down_proj` 保持 FP16（`--w8a8-names`，[说明](w8a8.md)） |

## 准备步骤

```sh
hf download perplexity-ai/pplx-decider-flash-v1.3 --local-dir "$HOME/models/pplx-decider-flash-v1.3"
```

权重原样使用，不需要转换（9.6 GB，含 `source/`）。


## Base Labs elephant-1 与 mammoth-1

服务 `decisions/jev/server.py` 是 Base Labs elephant-1（MIT；Qwen3.5-9B 后训练，块内双向注意力 + 字母槽读出）和 mammoth-1（MIT；Qwen3.5-27B，结构相同）的 MLX 移植
（`decisions/jev/jev_mlx.py`，骨干用 mlx-vlm 的 qwen3_5），SystemOne `POST /v1/systemone`。提示词渲染、前缀／后缀切分、答案槽、
校准和答案组装直接调用官方 OneJev `qev` 的 CPU 代码。路由模板：`templates/conf.d/decisions-baselabs.yaml`。

每个模型一个入口，精度按投影层类选择：能提速的层类用 W8A8（int8 权重 + int8 激活，M5 INT8 TensorOps），提速很小但误差占大头的层类换回
8-bit 只量化权重（W8A16），太窄的 `in_proj_a/b` 保持 BF16。量化结果预先存盘（`save_mix.py`），服务零拷贝映射，不再在加载时从 BF16 量化。

| 模型 ID | 精度 | 硬盘 |
| --- | --- | --- |
| `elephant-1` | W8A8；线性注意力 `in_proj_z` 用 8-bit 只量化权重；MLX 缓存 1024 MB；能与聊天模型共存 | `elephant-1-mix8` 10 GB |
| `mammoth-1` | 27B，同一移植；W8A8；线性注意力 `out_proj` 用 8-bit 只量化权重；MLX 缓存 4096 MB；不能与聊天模型共存 | `mammoth-1-mix8` 28 GB |

词表和 LM 头不进 GPU（从映射的 BF16 表读行，存在 `embeds.safetensors`）。文本输入；state + 一个问题最多 16,384 token。

### 准备步骤

```sh
# 官方 BF16 检查点只在生成预量化副本时需要，生成后可删
hf download baselabs/elephant-1 --revision b3f8a93462fb1051a9de031e94a7ff380e254072 --local-dir /tmp/elephant-1   # 18.8 GB
hf download baselabs/mammoth-1 --revision 61cec44c281c693992e3d6ebeb44dd3402dfa186 --local-dir /tmp/mammoth-1     # 54.7 GB
.venv/bin/python decisions/jev/save_mix.py /tmp/elephant-1 "$HOME/models/elephant-1-mix8" --quant w8a8 --w8a8-skip in_proj_z
.venv/bin/python decisions/jev/save_mix.py /tmp/mammoth-1 "$HOME/models/mammoth-1-mix8" --quant w8a8 --w8a8-skip out_proj
rm -rf /tmp/elephant-1 /tmp/mammoth-1
# 官方 OneJev（Apache-2.0）固定到 81ce62f，打上检查点自带的 serve.patch（与官方 setup.sh 相同；预量化副本里也有）
git clone https://github.com/OmniJev/OneJev decisions/OneJev
git -C decisions/OneJev checkout 81ce62f
git -C decisions/OneJev apply "$HOME/models/elephant-1-mix8/serve.patch"
```

`save_mix.py` 需要 Metal GPU（量化走 MLX），elephant 约 10 s、mammoth 约 30 s；生成的副本与加载时量化逐位相同。
mammoth-1 与 elephant-1 的 `serve.patch` 相同，共用这一个 OneJev checkout。OneJev 不随本仓库分发（`decisions/OneJev/` 在 `.gitignore` 里）；`tools/doctor.py` 会检查它是否就位。
官方运行时只支持 CUDA，这里只用它的 CPU 部分，不需要安装它的依赖。

## 未包含

Drex v1.5（需要 nace-ai 的 llama.cpp 分支）和把 OpenAI `POST /v1/decisions` 翻译成 SystemOne 的服务。
