# Whisper 对齐模块按需导入

**结论：模块导入中位耗时从0.530降到0.134秒，约少0.396秒；导入进程RSS峰值中位数约少96 MiB。没有证据表明GPU解码吞吐因此提高。**

代码在 [ain-soph/mlx-examples 的 serving 分支](https://github.com/ain-soph/mlx-examples/tree/serving/whisper)，固定提交 [43308d48](https://github.com/ain-soph/mlx-examples/commit/43308d480b4c647b3c542b8478054706ce1a3795)。本项目的 `pyproject.toml` / `uv.lock` 从该提交的 `whisper` 子目录安装。

## 改动

原版 `transcribe.py` 在模块顶层导入 `timing.add_word_timestamps`，连带导入Numba和SciPy。修改后只在执行 `word_timestamps` 分支时导入它，保留相同函数和对齐算法。

采用函数内导入，所以兼容既有Python版本，不要求全局lazy-import开关，也不要求第三方包只支持3.15。Numba和SciPy仍是依赖；首次词级时间戳请求仍支付初始化成本，导入后也会保留在该进程中。

## 基线和条件

- 源码基线：[mlx-examples c1af8c46](https://github.com/ml-explore/mlx-examples/commit/c1af8c46bd4c4704cff1561736a634cee38efc6c)，与当前PyPI `mlx-whisper 0.4.3` 的运行包源码逐文件一致。未混入上游后续模型默认值或加载路径变更。
- 2026-11-16，Apple M5 Ultra，96 GiB统一内存，macOS27；性能测量独占GPU预约。
- 包导入：Homebrew Python3.15.2，同一venv、同一依赖，仅源码目录不同。先各预热一次文件/字节码缓存，再运行7组交替的独立进程；不在同一进程内删除模块后假装重新导入。
- 推理：FP16 Whisper large-v3，KV实测FP16，allocator缓存上限512 MiB，单worker/c1；公开 `whisper.cpp` 的 `jfk.wav`，11秒、16kHz单声道；language=en、temperature=0、condition_on_previous_text=False。
- 同样的普通/词级转写及HTTP格式检查另在现有Python3.12.16环境运行；没有把跨Python版本差异归入这个改动的收益。

## 测量结果

| 指标 | 原版 | 修改版 | 解读 |
| --- | ---: | ---: | --- |
| 包导入耗时，中位数，n=7 | 0.52983 s | 0.13383 s | 3.96倍；耗时减少74.7% |
| 包导入耗时，7个样本范围 | 0.5247–0.5317 s | 0.1326–0.1362 s | 原始数值见[JSON摘录](whisper-startup.json) |
| 导入进程RSS峰值，中位数 | 167.30 MiB | 71.36 MiB | 减少95.94 MiB；不是模型GPU占用降低57% |
| 11秒音频普通转写，热态中位数，n=3 | 0.2335 s | 0.2280 s | 小样本，视为基本持平，不宣称推理提速 |
| 词级时间戳转写，热态中位数，n=3 | 0.2868 s | 0.2877 s | 基本持平 |
| 同轮MLX分配峰值 | 4.0689 GB | 4.0689 GB | 完全相同；不包含所有CPU RSS/文件页 |
| 同轮进程RSS峰值 | 3.422 GB | 3.397 GB | 单次全流程记录，包含已用过的对齐模块，不计算固定内存收益 |
| 同轮wired增量峰值 | 4173.91 MiB | 4072.08 MiB | 单次系统采样，有基线波动，不宣称固定节省值 |
| 模型权重与配置磁盘 | 3.084 GB | 3.084 GB | 不改变权重，也不删除Numba/SciPy依赖 |

首次模型加载/推理受到shader编译、文件缓存和运行先后顺序影响，没有足够配对样本，因此没有把该轮更大的首次请求时间差算作收益。也没有把推迟到首次词级请求的成本当作已经消失。

独立的同步阶段计数得到decoder prefill约49.92→49.86 tok/s、decode约159.95→159.08 tok/s，基本持平。这轮是两次转写合计6个decoder前缀token、58个增量decode token，不含encoder音频token，不能与LLM长prompt prefill或其他ASR模型的audio tokens/s直接比较。文本上下文448 token，音频按Whisper的30秒窗口处理；本样本不验证最长音频或高并发边界。

## 正确性

- Python3.15和3.12中，原版/修改版的普通转写及词级转写完整结果对象一致，包括segments、words和时间戳。
- 两个环境中，HTTP的JSON、verbose JSON（含words）、SRT、VTT响应内容分别一致。
- 两项不下载模型的回归测试检查：普通模型API导入不加载timing/Numba/SciPy；显式导入对齐API仍可用。
- 普通转写执行后修改版仍未加载这些对齐依赖；词级转写执行后它们正常加载。

这是一段公开英语音频上的回归检查，不替代多语言、长音频的完整准确率评测。改动没有改变数值计算，但不据此声称所有输入都已验证。

## 采样范围

系统自带Tachyon外部采样器在本机受到macOS进程内存访问权限限制，未取得外部采样结果；没有为此修改系统权限。补充的进程内Python栈采样和函数级剖析已经完成。12次热态普通转写循环中，原版/修改版分别采到1912/1902条worker栈，主要位置都在时间戳token规则的 `tokens.tolist()`、生成循环的 `mx.async_eval(...)` 与完成状态判断处，分布接近。导入阶段减少的依赖工作会在首次词级请求阶段出现，与按需导入的预期一致。

这类进程内采样有GIL调度偏差；函数级计时也包含原生调用等待及额外观测开销。它们只帮助定位调用路径，不用来计算CPU/GIL占比、GPU耗时或与上述无采样基准混算。逐token同步是后续独立评估的候选，当前改动没有调整解码循环。
