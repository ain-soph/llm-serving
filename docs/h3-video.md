# MiniMax H3 Video：权重准备与服务

服务 `services/video/h3-video/server.py` 调用 [h3.c fork](https://github.com/ain-soph/h3.c/tree/serving)（commit 固定在 `native.lock.json`，
`tools/native.py h3` 构建 `h3` 和 `h3_pack_int8`），每次渲染启动一个 `h3` 子进程，空闲时不占模型内存。
路由模板：`platform/llama-swap/conf.d/h3-video.yaml`。速度与内存：[性能与资源报告](performance.md#视频音频minimax-h3-video)。

2026-11-22 fork 整理：`serving` 重建为上游 `8974cc0` + 24 个签名提交（不再带未使用的上游 PR #77），渲染输出逐字节不变；本文提到的旧 commit（`0ff19ee`、`67a014a`、`0feaa4c` 等）在 tag `serving-pre-cleanup-20261122` 下仍可取。

## 入口

只有一个入口 `minimax-h3-video`（常驻模式，INT8 DiT 零拷贝映射）。请求体的 `model` 可以带预设后缀
`-best`／`-standard`／`-turbo`／`-fast`；旧名 `minimax-h3-video-stream` 仍被接受，作为别名走同一模式。

| 预设 | 计算 | 1344×768 5 s |
| --- | --- | ---: |
| `best` | 50 层、每步完整、密集注意力，精确 | 约 12 分钟 |
| `standard`（默认） | 45 层、reuse 2、放宽块稀疏注意力（`t0=1,r0=12,rmin=2,sink=1`） | 约 4.5–5 分钟 |
| `turbo`（可选，需另备权重） | Turbo 蒸馏 LoRA 折进 DiT，6 步、50 层、reuse 1、放宽块稀疏 | 约 3.3–3.6 分钟 |
| `fast` | 45 层、reuse 2、紧块稀疏 + token 削减 | 约 3 分钟 |

`turbo` 只用于纯文本和首尾帧请求、最长 8 s；带参考（Ref2VA）、续写或更长的请求自动改用 `standard`，
任务的 `settings` 里写明 `requested_preset: "turbo"` 和 `fallback` 原因。权重放在 `--models` 旁边的
`minimax-h3-turbo/`（或 `H3_VIDEO_TURBO_MODELS`），不存在时同样退回 `standard`。准备方法见下文“Turbo 权重”。

2026-11-17 起合并为一个入口：页对齐后的 INT8 DiT 用 `newBufferWithBytesNoCopy` 直接映射，常驻模式的 DiT 加载
2.29 → 0.33 s，速度与原 SSD 流式相同，三种方式输出逐字节相同。

h3.c `70f657e` 起，零拷贝 DiT 默认按 block 建、放 Metal buffer（每 2 个 block 提交一次），只在编码该 block 时持有它的权重：
768p 生成期 wired 21.9 → 14.3 GB，去噪时间不变，输出逐字节相同。开关：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `H3_DIT_STREAM_WEIGHTS` | 开 | `0` 恢复常驻（整个 DiT 的 buffer 一直存活） |
| `H3_DIT_STREAM_BLOCKS` | 2 | 每次提交的 block 数 |
| `H3_DIT_STREAM_INFLIGHT` | 2 | 最多在途的 command buffer 数 |

权重不是零拷贝（非 M5 或未页对齐）时自动保持常驻。

稀疏档（`standard`／`turbo`／`fast`）的稀疏评估总是走 h3.c 的 NAX 块稀疏内核（由稀疏计划选择）；2026-11-21 起包装器不再设
`H3_ATTN=nax`，前几次评估和首尾层这些密集注意力改用默认 MPS SDPA（NAX 密集慢 4–5%）：1344×768 5 s 去噪快 1.4–2.1%，输出变化为舍入级。

**视频 VAE（h3.c `67a014a` 起）**：解码器融合 BF16 输入与 tile 流水线、去噪期间预读 VAE 权重页，编码器 buffer arena 与融合 GroupNorm+SiLU+pad；
输出逐字节不变，768p 解码 24.5 → 21.2 s，首帧编码每张 5.0 → 3.3 s，Ref2VA 编码快约 19%。默认全开，`H3_VAE_FUSED=0`、`H3_VAE_PREFETCH=0`、
`H3_VAE_ENC_OPT=0` 分别退回原路径；`H3_VAE_ENC_BF16=1` 为可选的非逐字节变体（默认关）。
h3.c `243d6ab` 起，单张图（首／尾帧）编码只跑因果 3x3x3 卷积最后一个时间 tap 的 1x3x3 卷积（前两个 tap 只乘零帧），输出逐字节不变，768p 每张 3.3 → 1.25 s；原 `H3_VAE_ENC_FAST` 开关已删除，视频上下文（Ref2VA、续写）不受影响。

**旋转 W8A8 DiT（h3.c `0feaa4c` 起）**：`FL2VA/transformer-int8-rot256` 是 qkv／fc1／fc2 的 K 维乘 Hadamard H256 后再 INT8 的 pack
（`h3_pack_int8 --rot qkv,fc1,fc2`，再 `tools/align_int8_pack.py`），DiT 对官方 BF16 的误差约降 27%、去噪快约 1.5%。包装器对 fast／standard／best 的文本与首尾帧渲染设
`H3_DIT_VARIANT=rot256` 并加 `--use-int8-row-fc2`；Ref2VA 用自己的原 pack。turbo 也可用旋转 pack（`minimax-h3-turbo/FL2VA/transformer-int8-rot256`，见下文“Turbo 权重”）。没有这个 pack 时自动用原 pack；`H3_VIDEO_DIT_ROT=0` 关闭。
旋转类别写在 marker 和每个权重文件的元数据里，h3 加载时比对，不一致直接报错。不要把它改名成 `transformer-int8`：旧版 h3 不读旋转信息。

**提示词条件磁盘缓存**（h3.c `0ff19ee` 起，`H3_COND_CACHE_DIR`）：每次渲染是新进程，同一提示词原来每次都要重跑 tokenizer 和 INT8 文本编码器
（约 2.5 s、读约 24 GB），首帧／参考请求还要重跑 VAE、视觉和音频编码。缓存保存种子增强之前的全部条件，命中时输出逐字节相同。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `H3_COND_CACHE_DIR` | 关（包装器设为 `services/video/h3-video/cond-cache`，可用 `H3_VIDEO_COND_CACHE` 改） | 缓存目录 |
| `H3_COND_CACHE_MAX_MB` | 2048 | 总大小上限，按最近使用淘汰 |

键是 SHA-256：提示词与条件参数、首帧／参考文件内容、文本编码器／tokenizer／VAE 文件身份、着色器源码、可执行文件内容、系统版本、GPU、
ffmpeg（有媒体时）、除 DiT／采样器／诊断专用以外的 `H3_*` 环境变量。条目带 magic、版本、完整键和载荷哈希，任何不一致都重新计算；
写入先写临时文件再改名。删掉目录即清空。

## 权重目录

```text
<models>/minimax-h3/
  FL2VA/    text_encoder-int8/、video_vae/、audio_vae/、processor/、tokenizer/、transformer-int8/（页对齐）
  Ref2VA/   processor/、tokenizer/、transformer-int8/（页对齐）；video_vae、audio_vae 链接到 ../FL2VA
```

运行时只读 `transformer-int8/` 和 `text_encoder-int8/`，不需要官方 BF16 `transformer/`（每份约 66 GB）和 BF16 `text_encoder/`（62 GB）。
`text_encoder-int8/` 里的权重是 H3 图片模型的 INT8 convrot 文本编码器文件（同一份 Qwen3-VL 权重，符号链接，不另占空间），
h3.c `6c3a53e` 起按 W8A16 运行（8 位权重、BF16 激活）；`H3_TE=bf16` 可强制用 BF16 文本编码器（要先下载）。

## 准备步骤

```sh
M="$HOME/models/minimax-h3"                 # 换成实际的模型目录
H3=".local/src/h3-<commit 前 12 位>"          # tools/native.py h3 构建出的源码目录

# 1. 官方权重（含 BF16 transformer，打包要用）
hf download MiniMaxAI/MiniMax-H3 --local-dir "$M"

# 2. （可选，省空间）Ref2VA 的 text_encoder、video_vae、audio_vae 与 FL2VA 相同：先比对，再换成符号链接
for d in video_vae audio_vae; do diff -rq "$M/FL2VA/$d" "$M/Ref2VA/$d"; done
#    比对无差异后：mv "$M/Ref2VA/$d" 到别处，再 ln -s "../FL2VA/$d" "$M/Ref2VA/$d"

# 3. DiT 打包为 INT8（逐位校验），然后页对齐（逐字节校验后原子替换）
for v in FL2VA Ref2VA; do
  "$H3/h3_pack_int8" "$M/$v/transformer" "$M/$v/transformer-int8" --max-steps 20 --verify
done
python3 "$H3/tools/align_int8_pack.py" "$M/FL2VA/transformer-int8" "$M/Ref2VA/transformer-int8"

# 4. 文本编码器：直接用 H3 图片模型已页对齐的 INT8 文件（见 docs/h3-image.md），配置文件取自官方 text_encoder/
I="$HOME/models/minimax-h3-image/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.aligned16k.safetensors"
mkdir -p "$M/FL2VA/text_encoder-int8"
cp "$M/FL2VA/text_encoder/config.json" "$M/FL2VA/text_encoder-int8/"
ln -s "$I" "$M/FL2VA/text_encoder-int8/model.safetensors"
echo '{"source": "minimax-h3-image INT8 convrot text encoder", "precision": "W8A16"}' > "$M/FL2VA/text_encoder-int8/h3-te-int8.json"

# 5. 确认无误后可删除 BF16 transformer（每份约 66 GB）和 BF16 text_encoder（62 GB）；需要时再下载：
#    hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/transformer/*" "Ref2VA/transformer/*" "FL2VA/text_encoder/*" --local-dir "$M"
```

- 有 `text_encoder-int8/h3-te-int8.json` 时默认用 INT8 文件；`H3_QWEN_W8A16` 可按层类改走 W8A8（实测都不更快，不建议）。

- `align_int8_pack.py` 把每个文件改写为普通 safetensors：整页大小的张量排在前面，每个都从 16 KiB 页开始，可以零拷贝；
  每行 scale 等小张量放在后面，加载时照常拷贝。已对齐的文件会跳过。
- 未对齐的 `transformer-int8` 也能运行（自动回退到拷贝），只是加载慢一些。
- 打包和对齐后 doctor 只检查目录是否存在（`models.required.json` 的 `minimax-h3`），不校验内容。

## Turbo 权重（可选，`turbo` 预设）

[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)（Apache-2.0）的
`minimax_h3_turbo_v4_step600_ema.safetensors` 离线折进 FL2VA 的 BF16 DiT，再照常打包成 INT8。它的视频/音频 shift（12/3）
与 h3.c 的采样调度一致，引擎不需要改。只训练了 FL2VA，所以没有 Ref2VA 版本。

```sh
M="$HOME/models/minimax-h3"; T="$HOME/models/minimax-h3-turbo"
mkdir -p "$T/src" "$T/FL2VA"
hf download larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors --local-dir "$T/src/lora"
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/transformer/*" --local-dir "$T/src"     # 约 62 GB，折完即删
python3 "$H3/tools/fold_lora.py" "$T/src/FL2VA/transformer" "$T/src/lora/minimax_h3_turbo_v4_step600_ema.safetensors"
"$H3/h3_pack_int8" "$T/src/FL2VA/transformer" "$T/FL2VA/transformer-int8" --max-steps 20
python3 "$H3/tools/align_int8_pack.py" "$T/FL2VA/transformer-int8"
for f in audio_vae processor text_encoder-int8 tokenizer video_vae model_index.json; do ln -s "$M/FL2VA/$f" "$T/FL2VA/$f"; done
rm -rf "$T/src/FL2VA"                                                                       # 删 BF16 中间文件
```

- `fold_lora.py` 原地改写 BF16 分片，所以要折进一份单独下载的副本。它处理两处容易出错的地方：
  - 官方分片的 QKV 行按 head 交错存放，LoRA 是在 `[q;k;v]` 布局上训练的，折叠前先把增量交错回去；
  - LoRA 增量比半个 BF16 最小精度单位还小，四舍五入会把大部分增量舍掉（实测只剩 10–55%），默认改用按张量设种子的随机舍入（保留约 100%）。
- 磁盘：INT8 包约 22 GB，其余部件是符号链接。
- 旋转 W8A8（推荐，2026-11-23 起本机用它）：打包一步改为 `"$H3/h3_pack_int8" "$T/src/FL2VA/transformer" "$T/FL2VA/transformer-int8-rot256" --rot qkv,fc1,fc2 --max-steps 20`，再对该目录跑 `align_int8_pack.py`。包装器在非 stream 模式下优先用它；只有它而没有 `transformer-int8` 时，`H3_VIDEO_DIT_ROT=0` 或 stream 模式的 turbo 请求退回 standard。实测（对 BF16 turbo 参考，6 步 124 帧，3 种子 × 2 提示词）：平均 PSNR 512² 18.57 → 19.41 dB、768p 17.48 → 17.68 dB，个别种子更低（轨迹分岔）；768p 去噪 160.7 → 158.0 s，加载与内存不变。
