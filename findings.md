# 发现与决策

## 需求

- 用户目标：系统学习 NaVid-VLN-CE 代码库源代码。
- 前置需求：仓库需可写化（fork + 远端重绑），以保证后续记录进度能正常 push。

## 研究发现

### 仓库基本面（2026-09-28 侦察）

- 全仓库 112 个文件，Python 代码约 12041 行。
- 结构上明确分成两个半边：
  - `navid/` + `agent_navid.py` + `agent_uninavid.py`：模型侧，负责 VLM 读视频、出动作。
  - `VLN_CE/`：环境侧，负责 Habitat 仿真、VLN-CE 任务定义、指标计算。
- `VLN_CE/vlnce_baselines/models/` 下的 seq2seq / CMA / waypoint 系列是继承自 VLN-CE 的遗产代码，**NaVid 评估链路不经过它们**，学习时应主动跳过。

### 主调用链

```
eval_navid_vlnce.sh              # 起 8 个进程，每 GPU 一个 chunk
  └─ run.py:73  run_exp()        # 数据集切分 + 建环境
       ├─ get_config()           # VLN_CE/vlnce_baselines/config/default.py:294
       ├─ make_dataset()         # habitat 原生
       └─ evaluate_agent()       # run.py:98
            ├─ NaVid_Agent(...)              # agent_navid.py:21
            └─ while not env.episode_over:   # run.py:130
                 action = agent.act(obs, info)   # agent_navid.py:244
                 obs = env.step(action)
```

### 核心机制一：视觉 token 压缩（NaVid 的核心贡献）

位置：`navid/model/navid_arch.py:200-225` 的 `token_generation`。

- EVA-ViT-G 每帧输出 257 个 token（1 个 cls + 256 个 patch），代码中先切掉 cls 得到 256 个（16x16 网格）。
- 用 `F.avg_pool2d` 在 patch 网格上做池化压缩：
  - 历史帧 → `grid:2` → 2x2 = **4 token/帧**
  - 当前帧 → 强制 8x8 = **64 token**
- 压缩倍率由 `config.compress_type` 控制，`grid:2` 对应 `nav_size = 4`。
- 这是「视频能塞进 2048 上下文还跑得动」的根本原因。

### 核心机制二：历史帧增量复用

位置：`agent_navid.py:53-69` 的 `process_images`。

- 用 `start_img_index = self.history_rgb_tensor.shape[0]` 只对**新增帧**做预处理和 ViT 前向。
- 结果通过 `torch.cat` 追加到 `history_rgb_tensor`，旧帧不重复计算。
- README 中宣称的 "reuse of historical visual tokens" 即指此段。

### 核心机制三：连续语言指令 → 离散动作

位置：`agent_navid.py:151-175`（解析）与 `agent_navid.py:278-294`（换算）。

- 模型输出自然语言，如 `turn left 30 degrees`，用正则 `-?\d+` 抠出数字。
- 按仿真器步长换算成离散动作次数：
  - 环境配置 `FORWARD_STEP_SIZE: 0.25`（25cm）、`TURN_ANGLE: 30`（30度）—— 见 `VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml:7-8`
  - 代码 `min(3, int(num/25))` 次 forward、`min(3, int(num/30))` 次 turn
- **硬上限 3 步**：模型要求转 180 度时只会执行 90 度，下一轮重新观察后再决定。

### NaVid 与 Uni-NaVid 的架构分野

| 维度 | NaVid（agent_navid.py） | Uni-NaVid（agent_uninavid.py） |
|------|------------------------|-------------------------------|
| 历史帧处理 | agent 侧累积 `history_rgb_tensor`，增量预处理 | 每步重传全量 `rgb_list` 后清空 |
| 模型侧缓存 | 无显式缓存调用 | `config.run_type="eval"` + `initialize_online_inference_nav_feat_cache()` + `new_frames` |
| 输出格式 | 带角度的自然语言 | 纯动作词序列 `forward left stop` |
| 动作队列 | 按角度算步数，`min(3, ...)` | 直接解析词序列，最多取 2 个 |
| 推理加速手段 | 历史帧 token 压缩 | 在线 token 合并（online token merging） |

### 特殊 token 体系

定义于 `navid/constants.py:16-22`：
`<video_special>`、`</video_special>`、`<image_special>`、`</image_special>`、`[Navigation]`、`<image_sep>`。

- agent 侧在 `agent_navid.py:100-116` 把 input_ids 中的单个 `-200`（IMAGE_TOKEN_INDEX）替换成一段特殊 token 序列。
- 模型侧 `navid/model/navid_arch.py:360,392` 会跳过其中部分 token 并插入 nav token。
- `is_navigation` 的判定依据是 `NAVIGATION_IDENTIFIER`（值为 `'a video of historical observations and an image of the current observation'`）是否出现在 prompt 中 —— 见 `navid/model/navid_arch.py:154`。
- 待深入：`cur_input_ids[image_token_start + 1:image_token_start + 3]` 取 2 个 token，但跳过 3 个（`+3`），需要逐 token 核对确认 `</image_special>` 的去留。

### 仓库当前的可运行性缺口

- **`uninavid/` 目录不存在**。README 要求执行 `ln -s Uni-NaVid/uninavid uninavid`，目前未链接，因此 `agent_uninavid.py` 当前**无法运行**（其 import 会失败），只能作为对比阅读材料。
- **`model_zoo/` 目录不存在**，模型权重未下载。
- `agent_navid.py:303-600` 存在一个 `UniNaVid_Agent` 类，但 `run.py:107` 实际 import 的是 `agent_uninavid`，**该份实现是死代码**，阅读时不要困惑。

### 配置与早停逻辑

- `run.py:143` 的早停：连续旋转计数超过 `EARLY_STOP_ROTATION`，或总步数超过 `EARLY_STOP_STEPS` 时，强制下发 `{"action": 0}`（停止）。
- 连续旋转的判定：比较当前 `distance_to_goal` 与上一步，若未变化则计数 +1，一旦变化则清零 —— 见 `run.py:134-138`。

### 仓库可写化配置（fork）

- Fork 地址：https://github.com/kevinlasnh/NaVid-VLN-CE
- 远端布局：
  - `origin` → `https://github.com/kevinlasnh/NaVid-VLN-CE.git`（本 fork，main 跟踪 origin/main）
  - `upstream` → `https://github.com/jzhzhang/NaVid-VLN-CE.git`（原仓库，只读同步源）
- 认证方式：github.com 使用专属 credential helper `!gh auth git-credential`。
- **注意**：本机全局 `credential.helper` 被设置为 `store --file /home/kevinlasnh/.cache/huggingface/git-credentials`（HuggingFace 用），与 github.com 的专属 helper 并存，互不干扰。
- 写权限已做端到端验证（推送临时分支成功后删除），fork 上目前仅 `main` 一个分支。

## 技术决策

| 决策 | 理由 |
|------|------|
| 保留 upstream 而非删除原 origin | 保留同步上游新提交的能力，同时让裸 `git push` 默认打到自己的 fork |
| origin 使用 HTTPS 而非 SSH | 本机无 GitHub 专用 SSH key，且 github.com 已有 `!gh auth git-credential` 专属 helper，HTTPS 链路更可靠 |
| PWF 三件套放仓库根 | 遵循全局 Agent Markdown 的 L2 约定 |

## 遇到的问题

| 问题 | 解决方案 |
|------|---------|
| `gh repo fork` 报 `--remote flag is unsupported when a repository argument is provided` | 改为在仓库目录内运行 `gh repo fork --remote=false`，不带仓库参数，之后手动配置 remote |
| `git remote rename origin upstream` 连带把 main 跟踪改成了 upstream/main | 显式执行 `git branch --set-upstream-to=origin/main main` 重新绑定 |
| 当前仓库无 PWF 三件套 | 本次会话首次创建，遵循模板并填入实际内容 |

## 资源

- NaVid 论文：https://arxiv.org/pdf/2402.15852
- Uni-NaVid 论文：https://arxiv.org/pdf/2412.06224
- NaVid 项目页：https://pku-epic.github.io/NaVid/
- Uni-NaVid 项目页：https://pku-epic.github.io/Uni-NaVid/
- Uni-NaVid 微调代码仓库：https://github.com/jzhzhang/Uni-NaVid
- 上游依赖：LLaMA-VID（https://github.com/dvlab-research/LLaMA-VID）、VLN-CE（https://github.com/jacobkrantz/VLN-CE）

## 视觉/浏览器发现

本会话未执行网页浏览或图像查看操作，本节暂无内容。

---

*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
