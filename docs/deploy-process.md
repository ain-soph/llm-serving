# 标准部署流程（2026-11-16 起，用户要求）

每个新模型（以及复查已部署的模型）都按这个顺序做：正确性 → 测量 → 瓶颈 → 优化 → 清理 → 发布与维护。
结果写进 `reports/<area>.md`，进度记在 `WORKLOG.md`。
参考做法：Flash-Next（`reports/strata.md`）、H3 视频（`reports/h3-optimization.md`）、H3 图片（`reports/h3-image.md`）。

## 0. 环境（用户 2026-11-17）
- 开发新模型时在模型目录建 `.venv-dev`，随便装、不影响线上。
- 上线时把依赖并入 `~/llm-serving/pyproject.toml` 与 `uv.lock`（只写下限 `>=`，不写上限；`uv.lock` 精确锁定每个包），
  用 `cd ~/llm-serving && uv sync --frozen` 更新统一环境（`.venv` 就是这个项目的环境；改了 `pyproject.toml` 必须 `uv lock`，按上线处理），
  conf.d 的 cmd 改用 `~/llm-serving/.venv/bin/python`，:8080 验证后删除 `.venv-dev`。
- 依赖冲突优先在 fork 里修；fork 里上游写的上限先用最新版实测，能跑就去掉。只有确实修不了的才保留上限并写明原因。

## 1. 正确性基线
- 用官方实现（transformers / 官方推理代码）在同一组输入上得到参考输出，部署版本（GGUF / MLX / 自写）必须对得上：
  数值或分数误差、排序 / 识别结果一致；生成类模型看确定性输出是否逐字节一致，做不到时用 PSNR / 抽帧 / 人工比对。
- 量化、精度、内核改动后都要重新过一遍；结果写明“逐字节一致”还是“数值接近（附误差）”。
- 先量出“参照本身的误差”再判断：原实现两次运行之间的差异、换一种等价精度 / 等价内核的差异（H3 视频：等价 BF16 内核
  就有约 26 dB），部署版本的误差和它比，而不是和 0 比。
- 批量路径要和逐条结果对比（右侧补齐 + 每行位置从 0 开始，应等于逐条运行；嵌入 / 决策模型的对照脚本都两种都测）。
- 官方代码在非 CUDA 设备上常有降级（Decider 强制 FP32、线性注意力退回逐 token 的 PyTorch 循环）：适合做正确性参考，
  不能用来判断速度。

## 2. 测量（独占 gpu-lock，按真实负载）
- 速度：真实形状下的端到端时间，并拆到各阶段（加载、预处理、主计算、后处理）；文本模型给 prefill / decode tok/s。
- 内存：锁定内存（vm_stat wired 增量）、进程 footprint、页缓存依赖；和其他常驻模型（Flash-Next）共存时的压缩区 / 交换。
- 硬盘：每个文件的大小，是否有重复副本、转换源文件、评测残留。
- **测模型接受的最长输入**：容量必须覆盖最坏情况（嵌入 32K、Decider 32K 都是在这一步才发现峰值超出共存余量）。
- **按真实用法测共存**：对话与出图 / 检索交替时，模型切换和页缓存替换的延迟往往比常驻内存更重要（H3 图片：交替使用时
  共存只多 2–3 s，重新加载 Flash-Next 要 ~26 s）。共存满载时以压缩区 / swap 增长为门限，超限自动回滚（Strata C 方案）。
- 冷启动和热态分开报：新二进制或新形状第一次会编译 Metal 内核（Strata 7.3 s 计入了第一次 prefill），基准先预热。
- 连续测试之间等 wired 回落再取基线（否则增量会是负数）；直接读程序自己的缓冲区统计（llama.cpp 启动日志、
  `mx.get_peak_memory()`）作为交叉验证。

## 3. 找瓶颈
- 运行时与量化：比较可用的后端（llama.cpp / MLX / 自写）和量化级别的速度、内存、精度。**MLX 和 llama.cpp 必须都测**：
  重排模型上 llama-server 逐个文档计算，比 MLX 慢 1.4–2.8 倍（`reports/rerank.md`）；不要只看量化级别就定引擎。
- 读代码找明显低效：缓存上限、权重多余拷贝、每次请求重新加载 / 重启进程、批量过小、未融合的逐元素操作、同步等待。
- 用微基准确认占比（某个内核占多少时间、和硬件峰值差多少），再决定值不值得动。

## 4. 优化（按收益从大到小）
- 加载与内存：零拷贝 / 文件映射、页对齐、常驻还是按需、缓存预算、并行槽与批量。
- 量化：默认用标准的 Q8_0（或模型原生的 INT8 / BF16）；只有模型大、内存或硬盘确实紧张时才降到 Q6_K / Q4 等，并过精度检查。
- **W8A8 / INT8 计算只用在能提速的地方（用户 2026-11-17 定的原则）**：逐层（或逐类投影）实测，换成 INT8 后确实变快的才用 W8A8；
  提速不了的部分（小投影、只跑一次的层、注意力 / 递推这类非权重矩阵乘、文本编码器等占比很小的阶段）保持 BF16 / FP16 维护精度。
  反例：Qwen-Image 的 INT8 文本编码器（不提速、只省页缓存）已撤掉。只省内存的权重量化（Q8_0 等）是另一个理由，要单独写明。
  **补充（用户 2026-11-18）**：降精度不只看提速——能明显省硬盘 / 内存且误差很小，也应该降（例如 W8A8 不提速的层类改 W8A16 存储，而不是 BF16）。
  每个已部署模型、每类层都比较 W8A8 / W8A16 / BF16 / FP16（速度、误差、存储），按表选择。同一模型保留了多个精度入口的
  （`-int8` 与精确版等），要检查能否合并成一个又快、误差又小、占用又小的入口；只有多模态变种可以保留单独入口。不保留旧名字的别名。
  **检查标准（Decider 2026-11-17 的做法，`reports/optimization-audit-20261117.md` 第 3 节）**：先按部分 profile（短 / 中 / 长输入）；
  每类层逐类开关量化，测这一类的提速、它在端到端的占比、误差贡献（最大误差、答案 / 排序变化）；逐类决定——提速明显且误差小留 W8A8，
  不提速换回浮点，提速很小但误差占大头也换回浮点（Decider 的 down_proj：只快 2–5%，误差几乎全来自它）；报告给一张
  “层类 | 实测提速 | 端到端占比 | 误差贡献 | 最终选择”的表；不量化的部分（注意力、递推）另找更快的现成内核。
  小模型（几 GB）不要为省 1–2 GB 用非标准量化（用户 2026-11-16）。近似算法（稀疏、token 削减）只放在明确标为“近似”的档位。
  MLX 上要 8-bit 时，不用 mlx_lm 默认的 8-bit（非对称、BF16 缩放，8B 重排误差是 Q8_0 的 9 倍），
  用 `services/rerank/q8_0_to_mlx.py` 写成与 Q8_0 逐位相同的 MLX 格式（含词嵌入和输出层，FP16 激活；BF16 激活走慢路径）。
- 内核 / 算子：只有占比足够大时才写（M5 TensorOps 等）。
- 长序列的峰值可以**精确分块**压下来：逐 token 的部分（MLP、投影）按 token 分块；因果递推（线性注意力）分块时带上
  状态（用解码缓存）；注意力按查询分块（softmax 按查询独立）。Decider 32K：15.7 → 9.8 GiB，结果到小数点后 6 位相同。
- 只保留需要的代码路径：不同内存档位只体现在 llama-swap 容量上，代码里不保留多余模式（H3 图片去掉 resident）。
  同一模型的不同能力（纯文本 / 多模态）注册成两个入口，各自登记容量（用户 2026-11-16）。
- 每项优化都回到第 1 步验证。

## 5. 清理与维护
- 模型目录只保留一份权重；转换验证后删除源文件（先问用户）；评测用的其他版本不留。**删除要等配置最终确定**（用户认可之后），
  不要在对比测试刚出结果时就删掉备选版本。
- llama-swap：capacities = 实测锁定峰值 + 余量；配置注释写明依据（测试目录）。
- 报告写全规定指标：prefill / decode 速度、内存、权重量化、KV 量化（AGENTS.md 规则 8），格式见下面“报告怎么写”。
- 下载或写入 `~/models/<name>` 之前先 `ls -la`：可能已经下好（候选模型 2026-11-16 凌晨批量下载过）；我们的说明写进
  `SERVING.md`，不要覆盖模型自带的 `README.md`。

### 报告怎么写（用户 2026-11-17：简洁、清晰易懂，但内容详细完全）
`reports/*.md` 和公开仓库的性能报告都按这个结构：
1. **总览表**：每个模型 / 入口一行，只放关键数（速度、内存、磁盘、权重精度）。读者看完这一张表就知道结论。
2. **分模型小节**：测试条件一行 → 结果小表 → “我们的修改及收益”小表（修改前 → 后、倍数）→ 注意事项 1–3 条。
3. **测量方法与口径**放在最后统一说明（wired / RSS / footprint 的区别、GB 与 GiB、单次样本、冷热态），不要在每张表里重复。
- 表格单元格只写数字和短语，不写长句；长解释放到脚注或方法一节。
- 数字不丢：改写旧报告时，原有的每个数和测试条件都要保留（可以挪位置）。
- 写清楚是“逐字节一致”还是“数值接近（附误差）”；近似方案（稀疏、token 削减、INT8 激活）明确标出。
- 过程记录（试过但放弃的方案、调试经过）放在报告末尾的“过程与放弃的方案”，不要穿插在结论里。

## 6. 发布与维护

上线并清理完之后，代码要落到 GitHub，公开部署仓库要同步。两处都是公开的：提交前确认不含密钥、个人路径、邮箱、域名。

### 6.1 第三方库：ain-soph 公开 fork
- 任何对第三方代码的修改（库、引擎、包，包括 venv 里的文件）都在 git checkout 里做：`origin` = 上游，`fork` =
  `https://github.com/ain-soph/<repo>`（没有就 `gh repo fork OWNER/REPO --remote-name fork`），提交到分支 `serving` 并推送，
  然后从这个 commit 部署（Python 用 `pip install git+https://github.com/ain-soph/<repo>@<commit>` 或从 checkout 安装）。
  不在 site-packages 打补丁，不只留在本地磁盘。
- 现有 fork（本地目录 → 仓库）：

  | 本地目录 | fork | 部署方式 |
  | --- | --- | --- |
  | `services/image/qwen-image/src` | ain-soph/mflux | `services/image/qwen-image/.venv` 从 fork commit 安装 |
  | `services/video/h3-video/h3.c` | ain-soph/h3.c | 本地构建 `h3`（测试版另存其他文件名，不覆盖线上二进制） |
  | `services/image/h3-image-mlx/src` | ain-soph/h3-image-mlx | 从 fork commit 安装 |
  | `services/text/strata/repo` | ain-soph/Strata | 每次部署从已推送的 fork commit 建 detached release worktree `services/text/strata/releases/<名称>`，在其中构建 `build-metal-serving/strata`（视觉 `build-vision/bin/strata-vision`）；`conf.d/strata.yaml` 与 `configs/strata-*.json` 指向它；上一个 worktree 保留作回滚（`repo/build-*` 不是线上二进制） |
  | `services/asr/mlx-whisper/src` | ain-soph/mlx-examples | 从 fork commit 安装 `whisper` 子目录 |
  | `services/asr/mlx-audio` | ain-soph/mlx-audio | SOV 语音转写从 fork commit 安装（TTS 线上用 PyPI 0.6.6，公开仓库统一用 fork） |
  | `services/asr/mlx-qwen3-asr-src` | ain-soph/mlx-qwen3-asr | 从 fork commit 安装 |

- 提交规范：一个提交做一件事；标题写“改了什么”（英文，与上游风格一致），正文写为什么、怎么开（环境变量 / 参数）、
  验证结果（逐字节一致或误差、速度前后）；默认关闭的改动写明开关。实验分支（如 `fusion`、`decode-fusion`）单独推送，不合进
  `serving`，直到上线。
- fork README（用户 2026-11-17 定的统一格式，英文）：开头一个引用块，列出与上游的全部区别，一项一行；后面接上游原 README，不改上游内容。

  ```
  > **ain-soph fork (`serving` branch, based on upstream <tag/commit>).** What differs from upstream:
  >
  > | Change | Why | Enable / default | Effect (M5 Ultra) |
  > |---|---|---|---|
  > | Zero-copy page-aligned weights | … | `ENV=1` (off) | load 2.29 → 0.33 s; footprint −X GB; byte-identical |
  >
  > Full numbers: <ain-soph/llm-serving docs/performance.md 对应小节的链接>. Upstream PRs: #… (open/merged).
  ```
  - 单元格不写长句；Effect 只写最关键的一两个数（速度／内存），加上输出是否不变（byte-identical／误差）。
  - 数字只来自 `reports/*.md` 和公开仓库的 `docs/performance.md`，不为 README 新测；没有数据写 “—”。
  - 每次往 `serving` 加改动都同时加一行；不另留长段落说明（旧段落改写进表格）。依赖改动（如去掉版本上限）也列一行。
  - 在干净的 clone 上改（先拿 `serving` 最新版本），只改 README，签名提交并推送；不要在别人正在用的 checkout 里改。
- 合并上游新版本：在新分支上合并 → 构建 → 与合并前做 A/B（同输入、同 seed，输出逐字节一致；速度、内存不退步；参考
  h3-image-mlx 合并 v0.1.0a9：先确认上游改动只影响哪些路径，再在本机型上 A/B）→ 通过后快进 `serving` 并重新部署。用 rebase 跟上游时（例：Strata 2026-11-22，`serving-next` = 上游 main + 我们的提交），新分支单独推送、从它的 commit 建 release worktree 验收和部署；上线并重录金标准后，先在旧 `serving` 头打备份标签（`serving-pre-cleanup-<日期>`）并推送，再用 `git push --force-with-lease=serving:<旧头> fork <新 commit>:refs/heads/serving` 让 `serving` 指向新版本（用户 2026-11-22 决定：用新版本覆盖，不保留旧的合并历史）。
  有冲突时保留上游的结构，把我们的改动重新放上去，不反向改上游代码。
- 向上游开 PR 或 issue 需要用户同意。
- **mflux 的上游检查**（用户 2026-11-22 同意；维护者在 mflux #1019 要求）：往 `serving` 推送、合并上游新版本之后、给上游开 PR 之前，
  在开发 worktree 里跑 `just check`（pre-commit：ruff 检查和格式化、typos 等，**会改文件**，提交前跑）和 `just test-fast`
  （约 1,500 个 `fast` 测试，不出图、不下载）。`just test-fast` 先执行 `uv sync --all-extras`，会把当前虚拟环境同步成上游
  `uv.lock`：只在该 worktree 自己的 `.venv` 里跑，**不要在生产统一环境（`~/llm-serving/.venv`）或
  `services/image/qwen-image/.venv` 里跑**。测试用 GPU，先按规则拿一小份 gpu-lock。它只是第一道关：真实权重的逐字节对比、
  速度、内存和生产配置（W8A8、流式加载）仍按本文第 1、2 节自己测。平时改代码只跑相关测试文件即可。

### 6.2 llm-serving 公开部署仓库
`~/llm-serving` 就是 `https://github.com/ain-soph/llm-serving` 的工作树（分支 `main`）：**工作树 = 线上**。上线 = 在这里提交，
发布 = 推送（推送前问用户）。本机私有的文件（权重链接、venv、日志、key、实验、报告）被默认忽略，只有 `public-paths.txt`
列出的路径进库。

每个模型上线或改动后：
1. 服务代码直接在线上位置改（先备份到 `.backup/`），验证后提交。代码里不写本机绝对路径和个人域名：默认路径由文件位置推导
   （`Path(__file__).resolve().parents[N]`），本机取值走环境变量；ffmpeg 等外部程序按名字从 PATH 找。还在测试、未上线的功能
   放在 `git worktree add <临时目录> -b <topic>` 里开发，不放进 `~/llm-serving` 工作树。
2. llama-swap 片段 `platform/llama-swap/conf.d/<area>.yaml` 直接入库：本机路径写 `${env.LLM_SERVING}`、`${env.LLM_MODELS}`、
   `${env.STRATA_RELEASE}`（取值在 `.local/serving.env`，`platform/llama-swap/env.sh` 读入）；容量与注释写测量依据，不写本机目录。
   新的机器取值加到 `tools/install_serving.py` 的取值表和 `serving.env` 模板里。
3. 新的公开文件加进 `public-paths.txt`，运行 `tools/gen_gitignore.py`，与文件一起提交（不用 `git add -f`）。
4. 依赖与 commit：Python fork 写进 `pyproject.toml` 的 `[tool.uv.sources]`（完整 commit），`uv lock` 更新 `uv.lock`；原生引擎写进
   `native.lock.json`。fork commit 必须已推送。
5. 模型清单 `models.required.json` 和权重准备步骤（下载、页对齐、转换）写进 `docs/`。
6. 文档：README 只放简介、模型清单和安装步骤；性能与资源数字统一写在 `docs/performance.md`（格式见“报告怎么写”）。
7. 检查：`.venv/bin/python -m unittest discover -s tests`、`tools/doctor.py`、`tools/install_serving.py --check`；提交时 pre-commit
   钩子跑 `tools/privacy_check.py`（暂存区，带 `.local/private-literals`），推送时 pre-push 再加 `--check-history`；再人工看一遍 `git diff --cached`。
8. 只提交自己的路径：`git status` 先看，`git commit -o -- <paths>`；不用 `git add -A`、`git add -f`、`git commit -a`。
   在 `~/llm-serving` 里**不要**用 `git clean`、`git reset --hard`、`git checkout .`／`git restore .`、`git stash -u/-a`、`git switch`：
   被忽略的线上数据没有别的保护。
9. 提交用 GitHub noreply 身份；推送 `main` 前问用户；WORKLOG 记 commit。

## 附：经验与技巧（Qwen-Image-2.1、H3 部署中总结，2026-11-16）

### 正确性
- 生成模型对比时把**同一份初始噪声**注入两边（numpy 生成后传入），否则终图无法比较。
- 多步扩散会把 BF16 舍入差异放大成细节分叉，终图 PSNR 只能做参考。更可靠的是比较**第一步速度场**和文本 embedding 的
  相对误差，并以"参考实现自己换一种半精度"的误差作为噪声基线：比基线小 = 实现一致。
- 带条件输入（参考图编辑）时两边会更接近，可单独验证。

### 内存测量（Apple Silicon / MLX）
- 以系统 `vm_stat` 的 wired 增量为准，并**按阶段**（文本编码 / 去噪 / VAE）分别采样（0.1 s 间隔）。
- `mx.get_peak_memory()` 在零拷贝时会把映射视图重复计数；`ru_maxrss` 不含 Metal 缓冲区。进程占用看 `phys_footprint`
  （`proc_pid_rusage`）。
- MLX 在 GPU 计算期间会让**所有存活缓冲区常驻（wired）**，空闲时释放。所以"加载后空闲内存小"不代表生成期间小。
- MLX 默认缓存上限接近整机内存：形状多变的中间结果（如分块 VAE）会把缓存涨到几十 GB 并引发交换。服务进程一律
  `mx.set_cache_limit`（Qwen-Image 用 2 GB，比 0 还快）。

### 权重加载：零拷贝与流式
- 统一内存上"常驻"并不比"从页缓存读"快：两者读的是同一块 DRAM。真正的开销是**拷贝**（每请求逐层复制），不是位置。
- 零拷贝：safetensors 重写为页对齐（整页大小的张量排在前面，小张量放末尾，**保持首尾相连**，仍是官方库可读的标准格式），
  只读 mmap + `mx.from_dlpack`（已验证无拷贝），逐字节校验后替换原文件，只留一份。
- 流式：每层只在计算时持有视图（`with streamed(layer)`，层内 `mx.eval` 后释放）。算力瓶颈的模型几乎不掉速
  （读 436 MB 权重 ~0.5 ms，对比该层计算 ~30 ms）；生成期 wired 从"全部权重 + 激活"降到"一层 + 激活"。
- 依赖页缓存：内存紧张时页被回收会改为从 SSD 读，流式才会明显变慢——共存测试要看压缩区 / 交换。

### 大分辨率激活（VAE 等卷积网络）
- 全分辨率 FP32 卷积的中间结果是大图内存的主要来源。只要后续层都是局部运算（卷积、最近邻上采样、逐像素归一化），
  就可按行分块，结果在数学上精确。
- 先用**扰动法在 CPU 上实测感受野**（改一行输入，看输出哪些行变化），重叠取略大于实测值。
- 分两段：低分辨率、通道多的前几级分块到一个中间层，存下中间结果，再对高分辨率级小块解码。只对后几级分块时，前几级
  整图计算反而成为瓶颈。CPU 上应逐位一致；GPU 上因 Metal 内核随形状选择不同，会有舍入级差异（如 PSNR 74 dB）。

### 量化与计算精度的选择（Qwen-Image W8A8、Flash-Next 格式讨论中总结，2026-11-17）

**先分清两件事：计算精度决定速度，存储格式决定内存。**
- 先判断瓶颈：有效吞吐 = FLOPs / 时间。扩散模型 DiT、LLM prefill 是算力瓶颈（Qwen-Image DiT BF16 ~74 TFLOPS）；
  LLM decode 是带宽瓶颈（Flash-Next 每 token 读 5.78 GB）。
- 只量化权重（W8A16 / Q8_0 / Q4_K / MXFP4 等，计算前还原成 BF16）：算力瓶颈的模型**不提速**，只省内存
  （mflux `-q 8` 实测不快，还改了 Turbo 构图）。带宽瓶颈的 decode 才因读的字节少而变快。
- 量化激活（W8A8）才让矩阵乘本身变快：M5 TensorOps INT8 145–183 TOPS，BF16 86–103 TFLOPS，约 1.7×。
  W4A8 计算仍是 INT8，只多省内存。FP16 计算在 M5 上几乎没收益（BF16 1024² 快 9%，2048² 无，W8A8 上反而慢）。
- M5 没有 FP8/FP4 矩阵单元（据我们所知）；MXFP4/MXFP8/NVFP4 在 Apple 上等同只量化权重。MX 的缩放是 2 的幂（E8M0），
  同位宽下事后量化比 FP16 缩放的 GGUF/INT8 更不准；只在模型按 MX 训练（gpt-oss）或有原生硬件时占优。
- 看 bpw 要分部件：AtomicChat "4.27 bpw" = 主体 3.57 + PLE 表 6.0（Q5_1），路由专家只有 3.35；把 PLE 降到 4.5
  正好变成他们实测掉 top-1/KLD 的 3.84 版——对精度敏感的大表别降。

**W8A8 怎么做才准**
- 用 Hadamard 旋转（ConvRot，256 一组）把离群值摊开，再按行量化权重和激活；非 block 层（调制、输入输出、归一化）保留 BF16。
- 第三方量化文件先**自己从官方权重重新转换一遍逐张量对比**（Comfy-Org 的 Qwen-Image int8_convrot：99.99999% 相同），
  确认是标准方法、没有偷偷改动，再用；转换脚本留着，以后官方更新或合并 LoRA 可自己转。
- 精度看三层：权重误差（反量化后余弦，0.99996）→ 第一步速度场 / embedding 相对误差，与"官方实现换一种半精度"
  的基线比（W8A8 DiT 4.6% vs 基线 10.9% 合格；INT8 文本编码器 embedding 6.2% 超过 4.9% 基线 → 交给用户决定，
  最终两种都提供）→ 同噪声终图 + 人工看图。编辑任务（有参考图）比纯文生图更稳定，单独验证。
- 量化路径的辅助运算会吃掉收益：激活旋转量化用 FP32 矩阵乘时占 GEMM 的 40–70%，改成快速 Hadamard Metal 内核
  （按 H4 的 4 点蝶形，读一次写一次）后 7.2 → 2.2 ms；同一输入的多个投影（QKV、gate/up）只量化一次、合并成一个 GEMM。
- 近似计算（INT8 注意力、稀疏、token 削减）只在收益大时作为可选档：H3 视频注意力占 75–90%，`fast` 档值得；
  Qwen-Image 2048² 注意力只占 41%、INT8 QK 单独只快约 8%（缓存带宽受限），不做。

**测速时的坑**
- 微基准和真实流程会不一致：同一 GEMM 微基准快、流程里慢，原因是 MLX 缓存上限太小（2 GB），2048² 每层几百 MB
  的中间结果每次重新申请 → 每 block 160 vs 119 ms。按阶段设缓存上限（去噪 6 GB、VAE 2 GB、带参考图 2 GB）。
- 单次测量波动可达 9%；结论前至少重复一次，同一轮里放对照组。
- 测试脚本要和线上用同样的缓存上限、分块和环境变量，否则会测出假的慢（缓存膨胀到几十 GB → 交换）。

**多精度版本的取舍**
- 精度有争议时可以并列提供（Qwen-Image：BF16 / INT8 文本编码器两个入口），但每多一个入口就多一份页缓存和磁盘、
  客户端多一层选择；能用一个参数或一个入口解决就不要拆。
- 长序列输入（多张参考图）要单独测内存：按张数缩小参考图分辨率（总面积约 3 MP），否则 10 张参考图 18 GB。

### 服务包装器
- MLX 的 GPU 流按线程绑定：HTTP 服务器每个请求一个线程时，要把模型加载和生成放在同一个工作线程里。

## 附：经验与技巧（重排、嵌入、决策模型部署中总结，2026-11-16）

### 移植到 MLX 的检查清单
- 加载一律 `strict=True`：键名对不上时 `strict=False` 会静默保留**随机初始化**的权重（Decider 的检查点按 `Qwen3_5Model`
  保存，没有外层 `model.` 前缀，mlx_lm 的 sanitize 认不出）。加载后核对参数量和 dtype。
- 自己构造的模型要 `model.eval()`：mlx 模块默认是训练模式，Qwen3.5 / Qwen3-Next 的线性注意力在训练模式下走逐 token 的
  Python 算子路径（慢 10 倍、内存随长度暴涨）。
- 惰性求值：没有依赖关系的分支会被一起调度。分块计算时每块算完立刻 `mx.eval`，否则所有块的中间结果同时存在，分块失效。
- 激活精度先试 FP16：嵌入和 Decider 上 FP16 激活比 BF16 准约 10 倍，速度相同。转换方式：加载时逐张量 `astype` 并立即
  `mx.eval`（Decider 峰值 15.7 → 7.9 GiB，不另存副本）；整体 `set_dtype` 会让两份权重同时存在。也可以预先转存 FP16 替换原文件。
- 头维度 256 没有融合的注意力内核，长序列会完整算出 L×L 分数矩阵（见上面的精确分块）。
- 注意第三方包的导入副作用：mlx-vlm 导入 `qwen3_5` 时会把 transformers 的 `AutoProcessor` 换成它自己的处理器；
  需要官方预处理时先建好处理器再导入。
- 进程 RSS 不计 Metal 缓冲区（`set_dtype` 之后的权重、激活都不计），内存以 MLX 统计和 wired 为准。

### 服务与安全
- 请求里的字段绝不能被当作本地路径：官方 Decider 的 `open_image` 会把非 data URL 字符串当文件打开；服务只收 base64 data URL。
- API key 只在 llama-swap 校验：网关对旁路服务用 `forward_auth`，翻译 / 包装服务不存 key，原样转发 `Authorization`
  和客户端 IP（`X-Forwarded-For` / `X-Real-IP`）。
- 能力不匹配要显式拒绝：文本模型会静默忽略不认识的字段（Drex 忽略 `images`），所以翻译服务按 llama-swap 登记的
  `capabilities.in` 判断能否带图，而不是把请求转过去碰运气。
- llama-swap 不认识的接口（`/v1/decisions`、`/v1/videos`）由网关路由；它认识的（`/v1/systemone`、`/v1/rerank`）直接按
  `model` 路由，新模型只需登记 `capabilities`。

### 脚本
- **回归检查**：每次升级或改动后跑 `~/llm-serving/bin/regress --quick`（每个线上入口一条固定请求，经 :8080 和黄金输出比对）；
  大改（MLX / Python 环境 / 引擎升级、目录整理）跑完整模式 `bin/regress`。结果在 `logs/regress/`，有 FAIL 就退出码 1。
  有意改变输出的改动（换量化、换内核）上线后用 `bin/regress --record --only <入口>` 重新录黄金输出，并在报告里写明。
- **服务状态**：LaunchAgent `com.local.health` 每 10 分钟检查一次，摘要在 `logs/health/STATUS`，明细在 `logs/health/health.jsonl`。
- **备份只放 `~/llm-serving/.backup/`**（用户 2026-11-17）：改文件前复制到 `.backup/<相同相对路径>.<日期>-<标签>`，不要在原文件旁边留 `.bak`。
- **测试进程一律用 `bin/memguard` 包起来**（2026-11-17 09:58 内核 panic：一个测速进程在同一进程里加载 8B 的多个变体，常驻 79.7 GB，
  空闲页耗尽、看门狗重启）：`memguard --max-gb <预约 GB> -- 命令`；一个进程只放一个大模型变体；MLX 脚本设 `mx.set_memory_limit()`。
- 健康检查超时必须报错退出（Strata 的脚本曾在超时后仍打印 ready）；`lsw-apply` 失败后不要继续跑后面的测试
  （`... || exit 1`，曾因漏传 NAME 没应用配置、后面的测试照跑）。
- 带后台心跳子进程的锁包装器不要接管道（`gpu-run.sh ... | grep`）：心跳进程占着管道，命令会挂住；输出先写文件再过滤。
- 等锁 / 等会话结束的循环按退出码或会话脚本自己写的结束标记（如 `DONE`）判断，不要 grep “refused”“denied”这类词：
  gpu-lock 每次都打印的说明横幅里就有 “refused”，2026-11-16/17 至少三个 agent 的监视器因此提前退出、之后无人唤醒。
  另外 Claude Code 的 Monitor 最长 30 分钟，到时退出也不会报结果（2026-11-17 B、I 的验证结果因此迟了 1 小时以上）：
  可能超过 30 分钟的等待改用后台 Bash（`run_in_background`，`until grep -q DONE log; do sleep 30; done`），它结束时一定会通知。
