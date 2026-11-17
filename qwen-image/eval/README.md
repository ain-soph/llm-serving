# Qwen-Image 提示词评测

32 条短提示词 × 6 类（文字、计数、空间、简短、风格、编辑），1024²、seed 11，经 API 出图后由聊天模型看图两两评判。

```sh
# 出图：每个变体一个目录（可续跑）
.venv/bin/python qwen-image/eval/run_eval.py http://127.0.0.1:8080 qwen-image-2.1-turbo off qwen-image/eval/runs/turbo-off
.venv/bin/python qwen-image/eval/run_eval.py http://127.0.0.1:8080 qwen-image-2.1-turbo llm qwen-image/eval/runs/turbo-llm
# 评判：A、B 两种顺序各问一次，平局各计半票
.venv/bin/python qwen-image/eval/judge.py http://127.0.0.1:8080 qwen-image/eval/runs turbo-off turbo-llm
.venv/bin/python qwen-image/eval/summarize.py
```

- 编辑类的 6 条要用 `qwen-image/eval/src/` 下的参考图（`cafe.png`、`bicycle.png`、`figure.png`），仓库不附带，换成自己的图即可。
- API key 从 `LLM_API_KEY_FILE` 或状态目录的 `api_key` 读取。结果数据见 [性能报告](../../docs/performance.md#图片qwen-image-21)。
