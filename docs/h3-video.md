# MiniMax H3 Video：权重准备与服务

服务 `h3-video/server.py` 调用 [h3.c fork](https://github.com/ain-soph/h3.c/tree/serving)（commit 固定在 `native.lock.json`，
`tools/native.py h3` 构建 `h3` 和 `h3_pack_int8`），每次渲染启动一个 `h3` 子进程，空闲时不占模型内存。
路由模板：`templates/conf.d/h3-video.yaml`。速度与内存：[性能与资源报告](performance.md#视频音频minimax-h3-video)。

## 入口

只有一个入口 `minimax-h3-video`（常驻模式，INT8 DiT 零拷贝映射）。请求体的 `model` 可以带预设后缀
`-best`／`-standard`／`-fast`；旧名 `minimax-h3-video-stream` 仍被接受，作为别名走同一模式。

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
