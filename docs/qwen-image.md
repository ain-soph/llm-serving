# Qwen-Image-2.1：权重准备与服务

服务 `qwen-image/server.py` 用 [mflux fork](https://github.com/ain-soph/mflux/tree/serving)（commit 固定在 `uv.lock`）运行
Qwen-Image-2.1，提供 OpenAI 兼容的 `/v1/images/generations` 和 `/v1/images/edits`。路由模板：`templates/conf.d/qwen-image.yaml`。
速度与内存：[性能与资源报告](performance.md#图片qwen-image-21)。

**许可**：Qwen Research License，仅限非商用研究；商用需另行向 Qwen 申请授权。

## 入口

| 模型 ID | 步数 | DiT | 文本编码器 |
| --- | ---: | --- | --- |
| `qwen-image-2.1-turbo` | 8 | W8A8 | BF16 |
| `qwen-image-2.1-turbo-int8` | 8 | W8A8 | INT8（页缓存少 8.8 GB，误差略大） |
| `qwen-image-2.1` | 40 | W8A8 | BF16（支持 `negative_prompt` + `guidance`） |
| `qwen-image-2.1-int8` | 40 | W8A8 | INT8 |
| `qwen-image-2.1-turbo-pe` | 8 | W8A8 | BF16；不传 `prompt_enhance` 时默认 `llm`（`QWEN_IMAGE_DEFAULT_ENHANCE=llm`，需要下面的系统提示词） |

请求扩展字段：`seed`、`steps`、`negative_prompt` + `guidance`（仅 40 步版）、`reference_resolution`（编辑时每张参考图的边长；
默认 1–3 张 1024，更多张时按总面积约 3 MP 缩小，最小 512）。`background=transparent` 返回 RGBA PNG／WebP；
`mask` 按 OpenAI 约定（完全透明的像素重绘）。尺寸每边 256–2752，面积不超过 2400×1792。

提示词增强（默认关）：
- `prompt_enhance: mflux`：文本编码器自己逐 token 改写提示词（权重逐层流式）。
- `prompt_enhance: llm`：请聊天模型（默认 `qwen3.8-flash-next`，经本机 :8080）用 Qwen 官方改写系统提示词改写；编辑时附参考图。
  需要先下载官方系统提示词（上游未写许可证，所以本仓库不附带）：

  ```sh
  mkdir -p qwen-image/prompt_rewrite/prompts
  for f in system_prompt_t2i.txt system_prompt_edit.txt; do
    curl -fsSL "https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/prompt_rewrite/prompts/$f" \
      -o "qwen-image/prompt_rewrite/prompts/$f"
  done
  ```
  其他位置可用 `QWEN_IMAGE_SYSTEM_PROMPTS` 指定；聊天地址和模型用 `QWEN_IMAGE_LLM_URL`、`QWEN_IMAGE_LLM_MODEL`。
- 返回 `revised_prompt`；文生图且 `size=auto` 时按改写给出的宽高比选画布。
- `auto_mask: "<物体>"`：编辑时由文本编码器定位物体，只重绘它所在的矩形。

## 权重目录

```text
<models>/qwen-image-2.1/
  base/      官方 Qwen/Qwen-Image-2.1：model_index.json、processor/、scheduler/、text_encoder/、vae/、transformer/config.json
  turbo/     官方 Qwen/Qwen-Image-2.1-Turbo：同上；text_encoder、vae 是指向 ../base 的符号链接
  w8a8/base/model.safetensors    Comfy-Org qwen_image_2.1_int8_convrot（页对齐）
  w8a8/turbo/model.safetensors   Comfy-Org qwen_image_2.1_turbo_int8_convrot（页对齐）
  w8a8/te/model.safetensors      Comfy-Org qwen3vl_8b_int8_convrot（页对齐，只有 -int8 入口用）
```

运行时设置了 `MFLUX_QWEN21_W8A8`，不会读取官方 BF16 DiT（`transformer/*.safetensors`，每份约 14.2 GB），只读它的
`config.json`。五个入口合计约 42.8 GB。

## 准备步骤

```sh
M="$HOME/models/qwen-image-2.1"     # 换成实际的模型目录
PY=.venv/bin/python

# 1. 官方权重（不需要 BF16 DiT 时可以只下 transformer/config.json）
hf download Qwen/Qwen-Image-2.1 --local-dir "$M/base" --exclude "transformer/*.safetensors"
hf download Qwen/Qwen-Image-2.1 transformer/config.json --local-dir "$M/base"
hf download Qwen/Qwen-Image-2.1-Turbo --local-dir "$M/turbo" \
  --exclude "transformer/*.safetensors" "text_encoder/*" "vae/*"
hf download Qwen/Qwen-Image-2.1-Turbo transformer/config.json --local-dir "$M/turbo"

# 2. Turbo 与 base 共用文本编码器（750 个张量逐位相同）和 VAE（Turbo 的 VAE = base 的 FP32 VAE 舍入到 BF16）
ln -s ../base/text_encoder "$M/turbo/text_encoder"
ln -s ../base/vae "$M/turbo/vae"

# 3. 文本编码器重写为一个页对齐的标准 safetensors（逐字节校验后才删除原分片）
$PY -m mflux.models.common.weights.loading.zero_copy "$M/base/text_encoder"

# 4. Comfy-Org 的 INT8 ConvRot 权重
hf download Comfy-Org/Qwen-Image-2.1 \
  diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
  diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors --local-dir "$M/comfy"
mkdir -p "$M/w8a8/base" "$M/w8a8/turbo" "$M/w8a8/te"
mv "$M/comfy/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" "$M/w8a8/base/model.safetensors"
mv "$M/comfy/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors" "$M/w8a8/turbo/model.safetensors"
mv "$M/comfy/text_encoders/qwen3vl_8b_int8_convrot.safetensors" "$M/w8a8/te/model.safetensors"
$PY -m mflux.models.common.weights.loading.zero_copy "$M/w8a8/base" "$M/w8a8/turbo" "$M/w8a8/te"
```

然后运行 `tools/doctor.py`，它会按 `models.required.json` 检查这些文件。

- 页对齐后仍是普通 safetensors，官方工具可以照常读取。
- Comfy 的 INT8 DiT 与我们从官方 BF16 自己转换的结果 99.99999% 逐值相同，可以直接使用。
- 本机目录里还留着官方 BF16 DiT（页对齐），用于正确性对照；服务本身不需要它们。

## 服务设置（模板里的环境变量）

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `MFLUX_ZERO_COPY` | 1 | 页对齐文件用 mmap 映射，不拷贝到 MLX 缓冲区 |
| `MFLUX_ZERO_COPY_STREAM` | 1 | DiT block 和文本编码器层只在计算时持有权重（生成期 wired 44–52 → <10 GB） |
| `MFLUX_QWEN21_W8A8` | `w8a8/base` 或 `w8a8/turbo` | DiT 用 INT8 ConvRot 权重和 W8A8 GEMM |
| `MFLUX_QWEN21_TE_W8A8` | `w8a8/te` | 仅 `-int8` 入口：INT8 文本编码器 |
| `MFLUX_QWEN21_VAE_STRIP_FROM`、`_STRIP_BUDGET` | 2、16384 | VAE 两段条带解码，行数按图宽选 |
| `MFLUX_QWEN21_VAE_CACHE_MB` | 2048 | VAE 的 MLX 缓存上限 |
| `QWEN_IMAGE_CACHE_MB`、`QWEN_IMAGE_REF_CACHE_MB` | 6144、2048 | 去噪阶段 MLX 缓存上限（不带／带参考图） |
| `QWEN21_FUSION` | 不设 | 逐元素融合（不设 = 全开，空 = 全关）；输出与不融合逐字节相同 |

W8A8 GEMM 使用 M5 的 Metal TensorOps INT8 指令；其他机型未验证。
