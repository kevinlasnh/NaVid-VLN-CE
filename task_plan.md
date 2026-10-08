# 任务计划：NaVid-VLN-CE 源代码学习

将此文件作为任务的持久化路线图。在开始复杂工作前创建，并在阶段变化时及时更新。

## 目标

系统读懂 NaVid-VLN-CE 评估代码库的完整推理链路，理解 NaVid / Uni-NaVid 如何把视频 VLM 接入 Habitat 仿真器完成 VLN-CE 导航任务。

## 下一步

2026-10-08 更新：`navid_arch.py` 的模型侧主干已读通（`prepare_inputs_labels_for_multimodal` / `encode_images` / `vlm_attention` / `token_generation` 均已闭环），下一步：

1. 装配循环剩余细节：`new_input_embeds` 的拼装顺序，以及 labels / attention mask 的两条对齐分支（长度一致 vs 需 padding）
2. `navid/model/multimodal_encoder/eva_vit.py` 快扫（只看输入尺寸、patch 数、输出维度、惰性加载）
3. 回看 `agent_navid.py` 的 `predict_inference()` / `extract_result()` / 动作队列，把 agent 侧输入格式与模型侧拼装对上

## 当前阶段

阶段 2/3 交界：模型侧主干精读完成（`run.py`、`navid/model/builder.py`、`llava_navid.py`、`navid_arch.py` 的多模态装配与视觉压缩链路），agent 侧 `predict_inference` 细节待补

## 各阶段

将任务拆分为可验证的阶段。状态只能使用 `pending`、`in_progress` 或 `complete`，并在工作推进时更新。

### 阶段 0：仓库可写化配置（fork 与远端重绑）
- [x] Fork 仓库到 kevinlasnh 账号
- [x] 将原 origin 改名为 upstream
- [x] 新 origin 指向 fork
- [x] main 跟踪 origin/main
- [x] 端到端验证写权限
- **状态：** complete

### 阶段 1：仓库侦察与全景地图
- [x] 清点文件规模与目录结构
- [x] 读 README 与依赖清单
- [x] 读核心入口文件，摸清主调用链
- [x] 输出全景地图给用户
- **状态：** complete

### 阶段 2：主调用链精读
- [x] 精读 run.py（数据集切分、主循环、早停逻辑）
- [x] 精读 `navid/model/builder.py`（模型加载全流程）—— 计划外新增，因用户选定模型侧路线
- [x] 精读 `eval_navid_vlnce.sh` 的 7 个命令行参数并补注释
- [x] 精读 `VLN_CE/vlnce_baselines/config/default.py` 的 `get_config` 并补注释
- [x] 厘清 habitat / VLN-CE / NaVid 三层职责边界
- [x] 把发现记录到 findings.md
- [ ] 精读 agent_navid.py 的 act() 完整生命周期（骨架已看，`predict_inference` / `extract_result` 未细读）
- [x] 精读 predict_inference() 的 token 拼接顺序（2026-10-08：读通 `<video_special>/<image_sep>/<image>/</video_special>/<image_special></image_special>[Navigation]` 骨架与模型侧的展开顺序）
- [x] 精读 `llava_navid.py` 的 `forward()` 执行流程图（含 `generate()` 间接调用 `prepare_inputs_for_generation()` 的关系）
- **状态：** in_progress

### 阶段 3：模型侧核心机制
- [x] `prepare_inputs_labels_for_multimodal`（现从 341 行起，全文件核心）：主干、提前返回、两种输入形态归一、三分支装配、labels/mask 对齐均已读通
- [x] token 压缩的池化数学（token_generation + process_grid）：向量化语义、grid 与 nav_size 的对应关系已确认
- [ ] 历史帧增量复用机制（process_images + prepare_inputs_labels_for_multimodal）
- [ ] 特殊 token 在 input_ids 中的替换与跳过逻辑
- [ ] KV cache / 历史视觉 token 复用的真实边界
- [ ] `eva_vit.py` 快扫（输入尺寸、patch 数、输出维度、惰性加载）
- **状态：** in_progress

### 阶段 4：环境侧 VLN-CE 扩展
- [ ] habitat_extensions 的 task / measures / sensors / actions
- [x] get_config 的配置合并链路
- [ ] 指标计算（SR / SPL / NE / OSR）与 TOP_DOWN_MAP 来源
- **状态：** pending（配置合并链路一项已提前完成）

### 阶段 5：NaVid 与 Uni-NaVid 对比
- [ ] 对比两个 agent 的差异（历史帧处理、动作输出、缓存策略）
- [ ] 定位 uninavid 外部依赖的接口边界
- **状态：** pending

### 阶段 6：沉淀总结
- [ ] 形成完整学习笔记
- [ ] 判定哪些知识值得沉淀到长期记忆
- **状态：** pending

## 关键问题

记录需要解决的重要问题，并在获得答案后更新。

1. ~~`EVAL.EARLY_STOP_ROTATION` / `EVAL.EARLY_STOP_STEPS` 在 r2r yaml 中被设置，但在 `VLN_CE/vlnce_baselines/config/default.py` 的 EVAL 段中未见定义，需确认配置合并时是否通过。~~ **已解决**：无需在 `default.py` 预定义。yacs `CfgNode` 以 `new_allowed=True` 构造，允许运行时新增键；`merge_from_file` 会把 yaml 里的新键直接挂上。但这也意味着**拼错的键不会报错，只会静默多出一个无用字段**。
2. ~~`long_video` 的判定条件是 `images[0].shape[-1] > 1000`，此处 `shape[-1]` 究竟落在哪个维度，需结合 Uni-NaVid 的实际输入确认。~~ **已解决**（2026-10-08）：该启发式用"最后一维"区分两种输入形态——原始像素帧 `(N, 3, 224, 224)` 的 `shape[-1]` 是图像宽 224（走视觉塔编码）；预计算特征 `(T, 256, 1408)` 的 `shape[-1]` 是 EVA-ViT 隐藏维 1408（`long_video=True` 直通、跳过视觉塔）。1000 是两类数值量级之间的分隔阈值。
3. ~~用户尚未选定学习路线，这决定阶段 2 的展开方式。~~ **已解决**：用户选定模型侧路线（模型如何搭建），阶段 2 转为 `builder.py` → `llava_navid.py` → `navid_arch.py`。
4. ~~`navid/model/builder.py:122` 的 `context_len3` 缺陷是否修复？~~ **已解决**：用户已改回 `context_len`，`py_compile` 通过。
5. `agent_navid.py:317` 的 `require_data` 条件写错（`"video"` 应为 `"data"`），以及 `NaVid_Agent.__init__` 的 `require_map` 参数错位 —— 是否修复？**待用户决定**（两者都会改变 `EXP_SAVE="data"` 的行为）。

## 已做决策

记录重要决策及其理由。

| 决策 | 理由 |
|------|------|
| PWF 三件套放在仓库根目录 | 遵循全局 Agent Markdown 的 L2 约定，文件固定为仓库根 task_plan.md / progress.md / findings.md |
| 保留 upstream 远端而非直接删除原 origin | 保留同步上游新提交的能力，同时让裸 `git push` 默认打到自己的 fork |
| origin 使用 HTTPS 而非 SSH | 本机无 GitHub 专用 SSH key，且 github.com 已配置 `!gh auth git-credential` 专属 helper，HTTPS 链路更可靠 |
| 未创建 .planning 子目录 | 全局规则明确 PWF 固定放仓库根，且该仓库原本无 .planning 结构 |

## 遇到的错误

记录每个不同的错误、尝试次数和解决方案。操作失败后，先改变方法再重试。

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `gh repo fork` 报 `--remote flag is unsupported when a repository argument is provided` | 1 | 改为在仓库目录内运行 `gh repo fork --remote=false`，不带仓库参数，之后手动配置 remote |
| `git remote rename origin upstream` 会自动把 main 的跟踪改成 upstream/main | 1 | 显式执行 `git branch --set-upstream-to=origin/main main` 重新绑定 |

## 备注
- 随着工作推进，将阶段状态从 `pending` 更新为 `in_progress`，再更新为 `complete`。
- 做重大决策前，重新读取目标和下一步。
- 及时记录错误，避免重复失败的方法。
- 本仓库中 `VLN_CE/vlnce_baselines/models/` 下的大量 baseline 代码属于继承遗产，NaVid 评估链路不经过它们，学习时应主动跳过。
