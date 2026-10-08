# 进度日志

## 会话：2026-09-28

### 阶段 0：仓库可写化配置（fork 与远端重绑）
- **状态：** complete
- **开始时间：** 2026-09-28 11:17:43 +0800
- 执行的操作：
  - 检查 `gh` CLI 状态：v2.96.0，已登录账号 kevinlasnh，token 含 `repo` 等 scope
  - 确认原远端为 `https://github.com/jzhzhang/NaVid-VLN-CE.git`
  - 执行 `gh repo fork --remote=false`，在 kevinlasnh 账号下创建 fork
  - `git remote rename origin upstream`，保留原仓库为只读同步源
  - `git remote add origin https://github.com/kevinlasnh/NaVid-VLN-CE.git`
  - `git fetch origin` 拉取 fork 引用
  - `git branch --set-upstream-to=origin/main main` 重绑跟踪关系
  - 检查 fork 权限：`admin` + `push` 均为 true
  - 端到端写权限验证：推送临时分支 `tmp-verify-push-access` 成功后立即删除
  - 确认 fork 上仅剩 `main` 一个分支
- 创建/修改的文件：
  - 无（仅变更 Git 元数据，未改动工作区文件）

### 阶段 1：仓库侦察与全景地图
- **状态：** complete
- **开始时间：** 2026-09-28 11:17:43 +0800
- 执行的操作：
  - 清点仓库结构：112 个文件，Python 约 12041 行
  - 读取 `README.md`、`.gitignore`、`requirements.txt`
  - 阅读核心文件：`run.py`、`agent_navid.py`、`agent_uninavid.py`、`navid/model/builder.py`、`navid/model/navid_arch.py`、`navid/model/language_model/llava_navid.py`、`navid/mm_utils.py`、`navid/constants.py`、`eval_navid_vlnce.sh`、`navid_r2r.yaml`、`vlnce_task_navid_r2r.yaml`、`VLN_CE/vlnce_baselines/config/default.py`
  - 摸清主调用链并输出全景地图
  - 定位三个核心机制：视觉 token 压缩、历史帧增量复用、连续语言到离散动作的映射
  - 确认仓库可运行性缺口：`uninavid/` 与 `model_zoo/` 均不存在
- 创建/修改的文件：
  - 无（纯阅读）

### 阶段 2：主调用链精读
- **状态：** in_progress
- **开始时间：** 2026-09-28 11:17:43 +0800
- 执行的操作：
  - 建立 PWF 三件套（本文件、`task_plan.md`、`findings.md`）
  - 等待用户选定学习路线后展开逐段讲解
- 创建/修改的文件：
  - `task_plan.md`
  - `progress.md`
  - `findings.md`

## 会话：2026-09-29

### 阶段 2：主调用链精读（续）
- **状态：** in_progress
- **开始时间：** 2026-09-29 10:05:33 +0800
- 执行的操作：
  - 为用户在 `eval_navid_vlnce.sh` 中选中的 7 个命令行参数补充中文注释。因 bash 以 `\` 续行的命令中间不能插入注释行（`#` 会被当作普通参数传给 python），改为在命令上方集中写参数说明块，并用 `bash -n` + dry-run 验证命令行未被污染
  - 为 `VLN_CE/vlnce_baselines/config/default.py` 的 `get_config` 补充中文注释，标注五层合并顺序、`_C.clone()` 的防污染作用、`prev_task_config` 的用途、`freeze()` 的只读语义
  - 逐行精读 `run.py` 全文件：数据集切分与固定种子、8 进程分工、`Env` 与 `Agent` 的职责边界、episode 双层循环、早停逻辑、指标白名单与落盘
  - 精读 `navid/model/builder.py` 全文件：精度选择、三种加载形态、特殊 token 注册与词表扩容、视觉塔惰性加载、`context_len` 的来源与去向
  - 概念梳理：厘清 habitat / VLN-CE / NaVid 三者的分层职责
  - 联网核实 habitat-lab v0.1.7 的 `Config`（yacs `CfgNode`）、`Env.__init__`、`Dataset.get_splits` 实现细节
  - 联网核实 RxR 数据集背景（多语言、Guide/Follower 双标注、平均指令长度、3m 成功阈值）
  - 确认 NaVid 的 LLM 权重为全参数联合训练产物（README 明文 + `train.py` 默认值）
- 创建/修改的文件：
  - `eval_navid_vlnce.sh`（+24 行注释）
  - `VLN_CE/vlnce_baselines/config/default.py`（+26 行注释）
- 新发现（详见 `findings.md`）：
  - `navid/model/builder.py:122` 的 `context_len3` 拼写错误（本会话工作区新引入，会导致 `NameError`，评测无法启动）
  - `NaVid_Agent.__init__` 形参名与实参错位，导致 `require_map` 恒为 `True`
  - `agent_navid.py:317` 的 `require_data` 判定条件写错（该类未被 `run.py` 使用）
  - RxR 配置下 `EVAL.EARLY_STOP_STEPS` 与 habitat 的 `MAX_EPISODE_STEPS` 同为 500，该早停条件永不触发
  - `run.py` 中 `count`、`builder.py` 中 `context_len` 均为死值（只写不读）

### 阶段 2：模型侧精读——`llava_navid.py` 完成
- **记录时间：** 2026-09-29 17:56:39 +0800
- **状态：** 完成当前文件，准备进入 `navid/model/navid_arch.py`
- 本次完成的内容：
  - 理清 `LlavaConfig`、`LlavaAttLlamaModel`、`LlavaLlamaAttForCausalLM` 三层类结构，以及 Llama 基类与 NaVid mixin 的组合方式
  - 理清自定义 `__init__` 如何手动装配 `self.model` 与 `lm_head`，以及 `post_init()` 的位置
  - 读通 `forward()` 的执行顺序：设备处理 → 多模态输入拼装 → Llama 主干 → `lm_head` → causal language modeling loss → 返回结果
  - 理解 `logits[..., :-1, :]` 与 `labels[..., 1:]` 的错位对齐，用于让当前位置预测下一个 token
  - 区分两个“准备输入”函数：`prepare_inputs_for_generation()` 是 Hugging Face `generate()` 的逐步调度接口；`prepare_inputs_labels_for_multimodal()` 才负责把文本 token 与视觉特征拼成 Transformer 实际使用的 `inputs_embeds`
  - 确认推理时 `agent_navid.py` 通过 `self.model.generate(...)` 触发前者，再由 `forward()` 间接触发后者；直接调用 `model(...)` 时不会经过 `prepare_inputs_for_generation()`
- 下一步：精读 `navid/model/navid_arch.py` 中的 `prepare_inputs_labels_for_multimodal()`，先梳理输入输出与主分支，再拆解图像编码、特殊 token 替换、padding 和 label 对齐

## 会话：2026-10-08

### 阶段 2：`navid_arch.py` 精读（续）
- **状态：** in_progress
- **记录时间：** 2026-10-08 14:47:33 +0800
- 从 `prepare_inputs_labels_for_multimodal()` 开始继续精读。
- 已确认 `prompts` 的兜底链路：agent 先通过 `update_prompt()` 把当前提示词缓存到模型实例；`forward()` 未显式传入 `prompts` 时，本函数读取 `self.prompts`，再交给视觉编码路径判断是否为导航任务。
- **14:52:12 +0800：** 已读通函数的提前返回分支：首轮多模态输入会编码图片并建立 KV Cache；后续单 token 解码复用缓存，仅扩展 attention mask，不重复计算视觉 embedding。
- **16:59:25 +0800：** 已精读 `encode_images()`：确认它是统一视觉入口；普通像素输入先经过 vision tower，预计算特征由 `long_video=True` 分支直接复用，两条路径随后统一交给 `vlm_attention()` 完成样本分组、视觉 token 压缩、维度投影及输入类型标记。
- **17:14:48 +0800：** 已精读 `vlm_attention()`：读通按 `image_counts` 恢复样本边界、按 prompt 识别导航任务、去除 CLS、调用 `token_generation()` 生成“压缩历史 + 当前帧 64 token”，以及通过 `video_or_not` / `nav_or_not` 驱动后续三种视觉 token 拼装分支的完整流程。
- **21:41:57 +0800：** 已精读 `token_generation()` / `process_grid()`：确认它是"帧当批量维"的向量化实现（无逐帧循环，`reshape(shape[0], 16, 16, -1)` 只拆第 1 维、帧维全程保留，`avg_pool2d` 把帧维当 batch 独立池化）；每帧 256 个 patch token 按 `cur_shape // grid_size` 做不重叠分块平均，压成 `grid_size²` 个 token（grid:8→64、grid:4→16、grid:2→4），最后经 `mm_projector` 投影到 LLM 维度。导航分支唯一"单帧特判"是 `vis_embed[-1:]`（切片保持 3 维）。
- **21:41:57 +0800：** 打通「RGB 帧 → 视觉 token」完整链路：预处理（`agent_navid.py:63` → `(N,3,224,224)`）→ EVA-ViT 的 `PatchEmbed`（14×14 卷积把每帧切成 16×16=256 个 patch）→ 加 CLS、过 39 层 → `(N,257,1408)` → 丢 CLS（`% 2 == 1` 奇偶启发式）→ 网格池化 → `mm_projector`。并厘清角色边界：`encode_images()` 是观测编码入口，`prepare_inputs_labels_for_multimodal()` 是装配器（把视觉 token 拼进文本序列），不是编码器本身。
- **21:41:57 +0800：** 澄清 `image_counts` 的两种语义与来源：`None` = 每 prompt 一图、按下标位置对齐（`image_features[_idx, None]` 补帧维）；list = 帧已展平、用 `total_count` 前缀和游标切块。确认 `len(image_counts) == len(prompts) == input_ids.shape[0] == batch size`，且该列表仅在"images 为 list / 5 维张量"这条上层分支里被创建。
- **21:41:57 +0800：** 澄清 attention mask 的归属与推理期重建的必要性：该 mask 属于 **LLM**（非 ViT）；解码步重建为"长度 = KV cache + 1"的全 1 张量，是为了对齐 HF `generate` 按"占位符展开前"token 数维护的过期 mask 长度。
- **21:41:57 +0800：** 对 `navid/model/navid_arch.py` 做了一次**纯空白格式刷新**（类内 2 空行→1、3 空行→1、空白行去尾随空格，共 3 处），并以 AST 指纹 + token 流比对证明零源码改动。
- 创建/修改的文件：
  - `navid/model/navid_arch.py`（本次仅 4 行空白改动；其余为该文件随读随补的中文注释，共 870 行）
  - `agent_navid.py`、`navid/model/language_model/llava_navid.py`（随读随补的中文注释）

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| gh CLI 登录状态 | `gh auth status` | 已登录且具备 repo 权限 | 登录 kevinlasnh，token scopes 含 `repo` | 通过 |
| fork 写权限 | `git push origin HEAD:refs/heads/tmp-verify-push-access` | 推送成功 | 推送成功，远程提示可创建 PR | 通过 |
| 临时分支清理 | `git push origin --delete tmp-verify-push-access` | 删除成功 | 删除成功，fork 仅剩 main | 通过 |
| fork 权限查询 | `gh api repos/kevinlasnh/NaVid-VLN-CE` | push 权限为 true | `{"admin":true,"maintain":true,"pull":true,"push":true}` | 通过 |
| 凭据链路 | 检查 github.com credential helper | 存在可用 helper | `!gh auth git-credential` 已配置 | 通过 |
| eval 脚本注释后语法 | `bash -n eval_navid_vlnce.sh` | 语法无误 | 无报错 | 通过 |
| eval 脚本参数未被污染 | 将 `python run.py` 替换为 echo 后执行 dry-run | 8 个进程参数完整 | 8 行参数均完整正确 | 通过 |
| 配置注释后语法 | `python3 -m py_compile VLN_CE/vlnce_baselines/config/default.py` | 编译通过 | 通过（已清理 `__pycache__`） | 通过 |
| habitat 运行环境探测 | `python3 -c "import habitat"` / `find / -name habitat` | 确认是否可本地读源码 | `ModuleNotFoundError`，全盘无 habitat 包，本机未安装 | 未通过（环境缺失，改用联网核实） |
| `navid_arch.py` 格式刷新的语义一致性 | AST 指纹（`ast.dump` 的 sha256）+ 剔除空白 token 后逐 token 比对（含注释） | 源码零改动 | 前后 AST 指纹相同、3996 个有效 token 逐一相同、`py_compile` 通过 | 通过 |
| PWF 记录后源码可编译性 | `python3 -m py_compile navid/model/navid_arch.py navid/model/language_model/llava_navid.py` | 编译通过 | 通过 | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-09-28 11:17:43 +0800 | `gh repo fork` 报 `the --remote flag is unsupported when a repository argument is provided` | 1 | 去掉仓库参数，在仓库目录内执行 `gh repo fork --remote=false`，再手动配置 remote |
| 2026-09-28 11:17:43 +0800 | `git remote rename origin upstream` 自动把 main 的跟踪改为 `upstream/main` | 1 | 显式执行 `git branch --set-upstream-to=origin/main main` 重新绑定 |
| 2026-09-29 10:05:33 +0800 | 本机 `python3 -c "import habitat"` 报 `ModuleNotFoundError`，`find / -name habitat` 无结果 —— 无法本地读 habitat-lab 源码 | 1 | 改用联网核实 habitat-lab v0.1.7 源码（`Config`/`Env`/`Dataset`/`make_dataset`），并在结论中标注证据来源为联网而非本地 |
| 2026-09-29 10:05:33 +0800 | 发现 `navid/model/builder.py:122` 为 `context_len3`（本会话工作区新引入），会抛 `NameError` 导致评测无法启动 | 1 | **已修复**：用户自行改回 `context_len`，`python3 -m py_compile navid/model/builder.py` 通过 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 2/3 交界：`navid_arch.py` 的 `prepare_inputs_labels_for_multimodal`、`encode_images`、`vlm_attention`、`token_generation` 均已读通，视觉 token 压缩链路闭环 |
| 我要去哪里？ | 补装配循环剩余细节与 `eva_vit.py` 快扫，再回 `agent_navid.py` 把 `predict_inference` / `extract_result` 与模型侧对上 |
| 目标是什么？ | 系统读懂 NaVid-VLN-CE 的完整推理链路与核心机制 |
| 我学到了什么？ | 见 findings.md（配置五层合并、yacs CfgNode 体系、早停语义修正、代码缺陷、VLN-CE 三层职责、builder.py 实际执行路径、Llava/NaVid 模型调用链、RGB→视觉 token 三站链路、`image_counts` 双语义、attention mask 归属与重建原因、`token_generation` 向量化压缩语义） |
| 我做了什么？ | 完成 fork 与远端重绑并验证写权限；完成仓库侦察；建立 PWF 三件套；完成 `run.py`、`builder.py`、`llava_navid.py` 精读；厘清 habitat/VLN-CE/NaVid 分层职责；完成 `navid_arch.py` 多模态装配与视觉压缩主干精读；对 `navid_arch.py` 做纯空白格式刷新并验证零源码改动 |

---
*每个阶段完成后或遇到错误时更新此文件*
