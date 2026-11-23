# MiniMax H3 Image：权重准备与开关

服务 `services/image/h3-image-mlx/server.py` 调用 [h3-image-mlx fork](https://github.com/ain-soph/h3-image-mlx/tree/serving)（commit 固定在
`uv.lock`），入口 `minimax-h3-image`，所有尺寸都与聊天模型共存，路由在 `platform/llama-swap/config.yaml`；可选入口
`minimax-h3-image-pe` 先做官方提示词扩写（见文末），路由在 `platform/llama-swap/conf.d/h3-image-pe.yaml`。
速度与内存：[性能与资源报告](performance.md#图片minimax-h3-image)。

## 权重

```sh
M="$HOME/models/minimax-h3-image"     # Comfy-Org 打包的 MiniMax H3 Image
.venv/bin/h3-image-mlx align --models "$M"
```

`align` 把 DiT 和文本编码器换成页对齐的 `*.aligned16k.safetensors`（逐字节校验），加载时直接从页缓存映射（`H3_ZERO_COPY=1`）。
目录只保留一份 DiT，约 49.3 GB。

## 环境变量

| 变量 | 模板值 | 作用 |
| --- | --- | --- |
| `H3_ZERO_COPY` | 1 | 页对齐权重零拷贝映射，DiT 逐层流式 |
| `H3_ASYNC` | 1 | 异步提交 GPU 命令 |
| `H3_GPU_BUDGET_MB` | 8000 | DiT 阶段 MLX 缓存的内存预算 |
| `H3_FUSION` | 不设（= 全开） | AdaLN 调制与门控残差、SwiGLU、q/k RoPE 编译成融合内核；可设为 `modulate,swiglu,rope` 的子集，空 = 全关。输出逐字节相同 |
| `H3_ROTATE` | 不设（= 新内核） | 激活旋转量化用快速 Hadamard Metal 内核；`reference` 恢复原来的 float32 矩阵乘写法 |
| `H3_GEMM_TILES` | 不设 | GEMM 分块；`128,64,64` 恢复旧分块（输出相同，慢） |
| `H3_EMBED_CACHE` | `${env.LLM_SERVING}/data/cache/h3-image/text` | 文本阶段输出的磁盘缓存（按提示词 + 参考图；键含本包代码指纹、MLX 构建、`H3_ROTATE`、`H3_GEMM_TILES`、`dit_mode`）。不设 = 关 |
| `H3_EMBED_CACHE_MB` | 不设（= 1024） | 文本缓存上限，超出按最久未用删除 |
| `H3_REF_CACHE` | `${env.LLM_SERVING}/data/cache/h3-image/refs` | 每张参考图的 VAE 潜变量缓存（键为图的字节 + VAE 文件 + 代码指纹）。不设 = 关 |
| `H3_REF_CACHE_MB` | 不设（= 256） | 参考图缓存上限，按最久未用删除 |
| `H3_ROW_SLICE_FROM` | 不设（= 35000） | DiT 序列超过这么多行时按行切片（QKV、输出投影、MLP；注意力仍看整段） |
| `H3_ROW_SLICE` | 不设（= 16384） | 切片行数；`0` = 从不切片 |

**同一 seed 的图与 2026-11-17 之前不同**：fork `a42e001` 的旋转量化内核与旧写法相比，各有约 0.0001% 的 INT8 值差 1
（两者与 float64 精确结果的差距相同），但 6 步 DiT 会放大任何舍入差异，同一 seed 会得到另一张同样有效的图
（PSNR 1024² 22.7 dB、2048² 15.2 dB、4096² 18.6 dB）。要复现旧图，设 `H3_ROTATE=reference`（仍保留约 10% 的提速）。

**缓存与切片**（fork `971caf7`）：两种缓存的结果与不开缓存逐字节相同，只在命中时省时间（编辑 1 张图全命中约 −29%），
未命中不变慢；缓存在磁盘上，进程重启后仍有效，删除目录即清空。里面是提示词的文本编码器输出和参考图的 VAE 潜变量，
不是原文或原图，但仍属于用户数据，只放在本机（`data/` 不进仓库）。切片只在 35000 行以上（约 2048² + 16 张参考图）启用，
把 DiT footprint 压回 10000 MB 以内，代价约 +5% 耗时，输出逐字节相同。数字见 [性能与资源报告](performance.md#图片minimax-h3-image)。

## 官方提示词扩写（`minimax-h3-image-pe`）

```sh
hf download MiniMaxAI/MiniMax-H3-Image-PE --revision 35603d4b63949afa12cb098227fffefedf081afa \
  --local-dir "$HOME/models/minimax-h3-image-pe"     # 官方 NVFP4，17.9 GB，附带 prompt/ 模板
```

`minimax-h3-image-pe-expander` 用 `mlx_vlm.server` 直接加载官方 NVFP4（W4A16，BF16 KV，前缀缓存开），
`minimax-h3-image-pe` 是同一个图片服务加 `--default-enhance official`：文生图按官方 T2I 模板、编辑按官方 edit 模板
（附参考图）先扩写，`size=auto` 用扩写给的比例，每项返回 `revised_prompt`。请求可用 `prompt_enhance`
（`off` / `official` / `llm`）覆盖默认值。

| 变量 | 模板值 | 作用 |
| --- | --- | --- |
| `H3_PE_EFFORT` | `low` | 官方模板的 `reasoning_effort`（`xhigh` / `medium` / `low`；Bench 上三档无差别，`low` token 最少） |
| `H3_PE_MODEL` | 不设（= `minimax-h3-image-pe-expander`） | 扩写用的 llama-swap 模型 |
| `H3_PE_URL` | 不设（= `http://127.0.0.1:8080/v1/chat/completions`） | 扩写请求地址 |
| `H3_PE_DIR` | 不设（= `~/models/minimax-h3-image-pe`） | 官方 `prompt/` 模板所在目录 |

扩写模型约占 20–24 GB wired（容量 30000），放不下时路由器卸下聊天模型；出图入口 `evict_cost` 1，保证让位的是聊天模型
而不是正在等扩写结果的出图进程。`mlx_vlm.server` 在权重加载完之前 `/health` 就返回 200，加载中的请求得到 503
`model_loading`，服务端按 `Retry-After` 等待（最多 600 s）。
