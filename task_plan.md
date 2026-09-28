# 任务计划：NaVid-VLN-CE 源代码学习

将此文件作为任务的持久化路线图。在开始复杂工作前创建，并在阶段变化时及时更新。

## 目标

系统读懂 NaVid-VLN-CE 评估代码库的完整推理链路，理解 NaVid / Uni-NaVid 如何把视频 VLM 接入 Habitat 仿真器完成 VLN-CE 导航任务。

## 下一步

等待用户选定学习路线（沿数据流逐层精读 / 主题式深挖核心机制 / 先建全局索引 / 先跑通再理解），选定后进入阶段 2 的逐段讲解。

## 当前阶段

阶段 2：主调用链精读

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
- [ ] 精读 run.py（数据集切分、主循环、早停逻辑）
- [ ] 精读 agent_navid.py 的 act() 完整生命周期
- [ ] 精读 predict_inference() 的 token 拼接顺序
- [ ] 把发现记录到 findings.md
- **状态：** in_progress

### 阶段 3：模型侧核心机制
- [ ] token 压缩的池化数学（token_generation）
- [ ] 历史帧增量复用机制（process_images + prepare_inputs_labels_for_multimodal）
- [ ] 特殊 token 在 input_ids 中的替换与跳过逻辑
- [ ] KV cache / 历史视觉 token 复用的真实边界
- **状态：** pending

### 阶段 4：环境侧 VLN-CE 扩展
- [ ] habitat_extensions 的 task / measures / sensors / actions
- [ ] get_config 的配置合并链路
- [ ] 指标计算（SR / SPL / NE / OSR）与 TOP_DOWN_MAP 来源
- **状态：** pending

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

1. `EVAL.EARLY_STOP_ROTATION` / `EVAL.EARLY_STOP_STEPS` 在 r2r yaml 中被设置，但在 `VLN_CE/vlnce_baselines/config/default.py` 的 EVAL 段中未见定义，需确认配置合并时是否通过。
2. `long_video` 的判定条件是 `images[0].shape[-1] > 1000`，此处 `shape[-1]` 究竟落在哪个维度，需结合 Uni-NaVid 的实际输入确认。
3. 用户尚未选定学习路线，这决定阶段 2 的展开方式。

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
