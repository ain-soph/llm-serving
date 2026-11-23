# Z-Image-Turbo：权重准备与服务

服务 `services/image/z-image/server.py` 用 [mflux fork](https://github.com/ain-soph/mflux/tree/serving)（commit 固定在 `uv.lock`）运行
Tongyi-MAI 的 Z-Image-Turbo（6.15 B 单流 DiT，蒸馏 9 步文生图），提供 OpenAI 兼容的 `/v1/images/generations`。
路由模板：`platform/llama-swap/conf.d/z-image.yaml`。速度与内存：[性能与资源报告](performance.md#图片z-image-turbo)。

**许可**：Apache-2.0。

## 入口

| 模型 ID | 步数 | DiT | 文本编码器（Qwen3-4B） | VAE |
| --- | ---: | --- | --- | --- |
| `z-image-turbo` | 9 | 注意力＋前馈 W8A8（ConvRot int8），其余 BF16 | BF16 | BF16 |

请求字段：`prompt`、`size`（每边 256–2048，面积 ≤ 2048²；不是 16 倍数的边按下一个 16 倍数渲染后居中裁回）、`n`（≤ 4）、
`seed`、`steps`（默认 9）、`output_format`（png／jpeg／webp）。Turbo 没有 CFG，所以没有 `negative_prompt`；不支持编辑、遮罩、透明背景。

## 权重目录

```text
<models>/z-image-turbo/serve/
  model_index.json  scheduler/  tokenizer/  transformer/config.json
  text_encoder/model.safetensors                 官方 BF16，页对齐单文件（8.04 GB）
  vae/diffusion_pytorch_model.safetensors        官方 BF16，页对齐（0.17 GB）
  dit-w8a8.safetensors                           W8A8 DiT（6.30 GB，8.19 bpw）
```

准备步骤（官方 DiT 是 FP32，24.6 GB；准备完可删除或归档 `official/`）：

```sh
hf download Tongyi-MAI/Z-Image-Turbo model_index.json 'scheduler/*' 'tokenizer/*' 'text_encoder/*' 'transformer/*' 'vae/*' \
  --local-dir <models>/z-image-turbo/official
O=<models>/z-image-turbo/official; S=<models>/z-image-turbo/serve; mkdir -p $S/transformer
cp -c $O/model_index.json $S/; cp -cR $O/scheduler $O/tokenizer $O/text_encoder $O/vae $S/; cp $O/transformer/config.json $S/transformer/
# 页对齐（合并分片，逐字节核对后删除 $S 里的原分片）
python -m mflux.models.common.weights.loading.zero_copy $S/text_encoder $S/vae
# W8A8 DiT：直接从 FP32 权重在 CPU 上量化（不需要 GPU；M5 Ultra 约 15 s，峰值约 8 GB；运行时 W8A8 GEMM 需要 Apple M5 的 TensorOps）
python -m mflux.models.z_image.weights.z_image_w8a8 $O $S/dit-w8a8.safetensors
```

环境变量（模板里已设）：`MFLUX_ZERO_COPY=1`、`MFLUX_ZERO_COPY_STREAM=1`（文本编码器和 DiT 只在各自运行时映射权重，
锁定峰值约 8.8 GiB）、`MFLUX_ZIMAGE_W8A8=<文件>`、`ZIMAGE_CACHE_MB=2048`。
