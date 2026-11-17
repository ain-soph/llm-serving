# MiniMax H3 Image：权重准备与开关

服务 `h3-image-mlx/server.py` 调用 [h3-image-mlx fork](https://github.com/ain-soph/h3-image-mlx/tree/serving)（commit 固定在
`uv.lock`），一个入口 `minimax-h3-image`，所有尺寸都与聊天模型共存。路由在 `templates/config.yaml`。
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

**同一 seed 的图与 2026-11-17 之前不同**：fork `a42e001` 的旋转量化内核与旧写法相比，各有约 0.0001% 的 INT8 值差 1
（两者与 float64 精确结果的差距相同），但 6 步 DiT 会放大任何舍入差异，同一 seed 会得到另一张同样有效的图
（PSNR 1024² 22.7 dB、2048² 15.2 dB、4096² 18.6 dB）。要复现旧图，设 `H3_ROTATE=reference`（仍保留约 10% 的提速）。
