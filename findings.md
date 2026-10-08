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

### 配置体系：get_config 的五层合并（2026-09-29）

位置：`VLN_CE/vlnce_baselines/config/default.py:294`。

- 合并顺序即优先级（后盖前）：
  1. `habitat_baselines.config.default._C`（框架底座）
  2. `purge_keys` 删 `SIMULATOR_GPU_ID` / `TEST_EPISODE_COUNT` 并标记 deprecated
  3. 本文件 `_C.clone()`（VLN-CE 默认值）—— `.clone()` 不可省，否则会就地污染模块级全局 `_C`
  4. 实验 yaml（`merge_from_file`）
  5. `opts` 命令行覆盖（同时存进 `CMD_TRAILING_OPTS` 留档）
- 关键机制：yaml 里的 `BASE_TASK_CONFIG_PATH` 一旦变化，就递归调 `get_task_config`（真名 `get_extended_config`，`habitat_extensions/config/default.py:140`）把 task yaml 整体挂成 `config.TASK_CONFIG` 子树。`prev_task_config` 用于多配置文件共享同一 task 配置时避免重复加载。
- 因此最终 config 住着两层：外层实验级（EVAL/IL/RL/MODEL），内层 `TASK_CONFIG`（SIMULATOR/TASK/DATASET）。
- `config.freeze()` 返回只读对象；下游 `habitat.Env.__init__` 第一句就 assert 配置已冻结，所以 freeze 不是可选防御。需修改时必须 `defrost()`（参见同文件 `add_pano_sensors_to_config`）。
- **`config` 的实际类型是 yacs `CfgNode`，不是 OmegaConf**（联网核实 habitat-lab v0.1.7 源码：`class Config(yacs.config.CfgNode)`，`__init__` 传 `new_allowed=True`）。`merge_from_file` / `merge_from_list` / `freeze` / `defrost` / `clone` / `register_deprecated_key` 全是 yacs 自带方法，habitat 只加了 `new_allowed=True`。**注意**：新版 habitat-lab（v0.2+）已改用 OmegaConf，API 完全不同，查资料时勿混用。

### 早停逻辑的精确语义（修正此前理解）

- `continuse_rotation_count`（注意拼写错误 `continuse`）**统计的不是"旋转"，而是"连续多少步 `distance_to_goal` 未变化"**。`run.py` 全程未检查 action 类型，撞墙、来回走、连续 STOP 都会被计入。命名有误导性。
- 它是**连击计数器**（一旦有进展即清零），不是累计次数。
- 判断为 `>` 而非 `>=`，故 `EARLY_STOP_ROTATION = 25` 实际需连续 **26** 步无进展才触发。
- 命中后 `action = {"action": 0}` 覆盖 agent 决策（0=STOP，依据 task yaml 的 `POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]` 顺序）。这是**先决策后否决**，对 agent 透明。
- **RxR 配置下 `EVAL.EARLY_STOP_STEPS = 500` 与 habitat 的 `ENVIRONMENT.MAX_EPISODE_STEPS = 500` 相同**，而早停条件是严格大于，需要 `iter_step = 501` 才触发；但 habitat 会在第 500 步就把 `episode_over` 置真，循环先退出 —— **该条件在 RxR 下永不触发**，实际只有 `EARLY_STOP_ROTATION` 在起作用。R2R 配置为 400 < 500，才真正生效。

### 两个早停上限分属不同层

| 上限 | 定义位置 | 读取方 | 生效方式 |
|------|---------|--------|---------|
| `ENVIRONMENT.MAX_EPISODE_STEPS` | task yaml（habitat 体系） | `habitat.Env` 内部 | `episode_over` 自动置真 |
| `EVAL.EARLY_STOP_STEPS` | 实验 yaml（VLN-CE 脚本体系） | `run.py:163` 自数 `iter_step` | 强行改动作成 STOP |

- habitat 默认 `MAX_EPISODE_STEPS = 1000`（联网核实 v0.1.7 源码），本仓库 11 个 task yaml 中除 `vlnce_waypoint_task.yaml`（200）外全部覆盖为 500。

### 代码缺陷清单（2026-09-29 发现）

| 位置 | 问题 | 影响 | 状态 |
|------|------|------|------|
| `navid/model/builder.py:122` | `return tokenizer, model, image_processor, context_len3` —— 末尾多了 `3` | **`NameError`，`load_pretrained_model` 返回时抛异常，agent 初始化即崩，评测完全跑不起来** | 本会话工作区新引入（git diff 可证原始版本无此问题）；**已修复**，用户改回 `context_len`，`py_compile` 通过 |
| `agent_navid.py:22` vs `run.py:118` | `NaVid_Agent.__init__(self, model_path, result_path, require_map=True)`，但调用处传 `NaVid_Agent(model_path, result_path, exp_save)` —— 第三位实参是 `exp_save` 字符串，被塞进了 `require_map` | 任何非空字符串均为 truthy，故 `require_map` 恒为 `True`；`EXP_SAVE="data"` **达不到脚本注释宣称的加速效果**，仍会画俯视图、存 gif | 待定（需改 agent 侧签名，影响面稍大） |
| `agent_navid.py:317` | `self.require_data = True if "video" in exp_save else False` —— 应为 `"data"`（对比 `agent_uninavid.py:26` 的正确写法） | 若使用该类且 `exp_save="data"`：不建 `log` 目录，而 `run.py` 仍会写 json → `FileNotFoundError` | 该类未被 `run.py` 使用（`run.py:121` 导入的是 `agent_uninavid` 版本），属遗留隐患 |

### 死代码 / 死值清单

- `run.py:139` `count = 0` 与 `:174` `count += 1` —— 只写不读，从未参与判断或输出。
- `run.py:191` `result_dict = dict()` —— 紧接着被字典推导式整体覆盖，纯废行。
- `navid/model/builder.py` 的 `context_len` —— 全仓库 3 处赋值、**0 处读取**；`self.context_len` 存下后再未使用。
- `navid/model/builder.py` 大量不可达分支：8bit/4bit 量化（调用时不传开关）、`model_base is not None`（调用传 `None`）、`else` 整段（`model_name` 必含 "navid"）、`use_fast = False` 局部变量（赋值后未读）、`print('c')` 调试残留。**实际执行路径不到全文件 20%**：约 `:27, :40, :54, :55, :95-114, :116-119, :122`。
- `navid/model/navid_arch.py:89-92` 的 `initialize_attention_modules` 是**空壳函数**：前两行赋给局部变量后未使用，第三行仅把 `compress_type` 从 `model_args` 抄进 `self.config`，`for_eval` 参数完全未使用。名字与实现严重不符。
- `agent_navid.py:303-600` 的 `UniNaVid_Agent` 类为死代码（`run.py` 导入的是 `agent_uninavid` 版本）；其 `:382-393` 还重复了一遍特殊 token 的硬编码。

### `run.py` 主循环要点（精读补充）

- `evaluate_agent` 的结构：建 `Env` → 按 `model_name` 建 Agent → 读早停阈值与指标白名单 → 双层循环（外层 episode / 内层 step）→ 每 episode 收尾落盘。
- **`run.py:190` 的 `info = env.get_metrics()` 是必需的、不是重复**：循环体内的 `info`（`:159`）取于动作执行**之前**，退出循环时最后那次 `env.step()` 已改变世界但无人重读；直接用旧值会得到"停下之前"的 `distance_to_goal` 与 `success`，评测结果出错。
- `target_key` 白名单保留 5 项（`distance_to_goal`/`success`/`spl`/`path_length`/`oracle_success`），但 task yaml 声明了 8 个 MEASUREMENTS，被丢弃的是 `top_down_map_vlnce`、`ndtw`、`steps_taken`。注意 `top_down_map_vlnce` 虽不落盘，**agent 在 `require_map=True` 时仍会读取**（`agent_navid.py:251`），不是白算。
- `info` 被三方共用：`run.py` 用 `distance_to_goal` 做卡死检测；agent 用 `top_down_map_vlnce` 画图；白名单 5 项用于落盘。
- 落盘开关 `"data" in exp_save` 是**子串匹配**，`"data"` 与 `"video-data"` 均命中。`log` 目录由 **agent 的 `__init__`** 创建（`run.py` 内无 `makedirs`），两侧条件必须对齐，否则 `FileNotFoundError`。当前 NaVid 路径无条件建目录、Uni-NaVid 路径用 `require_data`（=`"data" in exp_save`）守卫，恰好一致，属**靠巧合对齐**。
- `dataset` 是**对象**（`RxRVLNCEDatasetV1` 实例），列表在其属性 `dataset.episodes`；`dataset.get_splits(n)` 返回的也是**对象列表**（`List[Dataset]`），故 `dataset_split` 仍是对象而非 episode 列表。`get_splits` 内部用 `np.random.choice` 随机抽下标，这正是 `np.random.seed(42)` 必须存在的原因。
- `make_dataset(id_dataset, config)` 是 habitat 的注册表工厂：字符串 → `registry.get_dataset()` 查表 → 类 → `_dataset(**kwargs)` 实例化。`"RxR-VLN-CE-v1"` 由 `VLN_CE/habitat_extensions/task.py:125` 的装饰器在 import 时注册；因此 `run.py:5` 那个看似无关的 import 是注册发生的必要条件。

### VLN-CE / habitat / NaVid 三层职责澄清（重要概念）

此前"VLN_CE 是环境提供方"的表述不准确，正确的是**分层**：

| 层 | 提供方 | 提供内容 |
|----|--------|---------|
| 引擎 | habitat-sim | 3D 渲染、物理、场景加载 |
| 框架 | habitat-lab | `Env` 循环、**注册表机制**、`Dataset`/`Sensor`/`Measure` 基类、**`VLN-v0` 任务基座本身**、标准导航指标 |
| VLN 内容 | `VLN_CE/habitat_extensions` | RxR/R2R 数据集解析、额外传感器、额外指标、额外动作 |
| 实验参数 | `VLN_CE/vlnce_baselines/config` | 实验级 yaml |

- 关键证据：全仓库搜 `VLN-v0` 只有 `base_il_trainer.py:40` 一处**字符串引用、无定义**；且 `task.py:13` 反过来 import habitat 的 VLN 基类（`from habitat.tasks.vln.vln import InstructionData, VLNEpisode`），`VLNExtendedEpisode(VLNEpisode)` 是继承扩展而非从零造。
- VLN-CE 通过 habitat 的注册表扩展点填入内容：`register_dataset`（`VLN-CE-v1`、`RxR-VLN-CE-v1`）、`register_sensor`（5 个）、`register_measure`（9 个）、`register_task_action`。
- navid 的 task yaml 逐项来源：`TASK.TYPE: VLN-v0`、`SUCCESS_DISTANCE`、`INSTRUCTION_SENSOR`、`RGB_SENSOR`、`DISTANCE_TO_GOAL`/`SUCCESS`/`SPL`、`POSSIBLE_ACTIONS` 均来自 **habitat**；`DATASET.TYPE`、`SHORTEST_PATH_SENSOR`、`PATH_LENGTH`/`ORACLE_SUCCESS`/`NDTW`/`STEPS_TAKEN`/`TOP_DOWN_MAP_VLNCE` 来自 **VLN-CE**。已核对 `measures.py` 注册的 9 个指标中**不含** `success`/`spl`/`distance_to_goal`。
- 推论：想读懂 `success`/`spl`/`Env.step()` 内部，需看 habitat 源码（本机未装）；想读懂 RxR 解析、NDTW、俯视图，看本仓库 `habitat_extensions/` 即可。
- `VLN_CE/vlnce_baselines/__init__.py:7` 的 models 导入被注释，实际只加载 `common.environments`；故 `models/`（CMA/Seq2Seq/Waypoint）与 `common/` 训练基础设施**评测链路完全不走**。

### `navid/model/builder.py` 精读要点

- `load_pretrained_model` 是**配置装配 + 权重加载**函数，不是核心逻辑；实际执行路径：精度选 `torch_dtype=float16` → 判定 `'navid' in model_name` → `model_base is None` 全量加载（2 行）→ 特殊 token 与视觉塔装配。
- `'navid' in model_name.lower()` 的判据来自 `get_model_name_from_path`（`mm_utils.py:207`，取路径末段）。**权重目录名参与逻辑判断**，改名会导致走错分支。
- 三种加载形态的差别（**都是加载完整模型，区别只在权重来源**）：

| 形态 | LLM 权重来源 | 投影层权重来源 |
|------|-------------|---------------|
| `model_base is not None` | base 目录（旧） | model_path（新，`mm_projector.bin`，`strict=False` 部分覆盖） |
| `else`（实际走这条） | model_path（新） | model_path（新） |
| 非 navid + `model_base` | LoRA：`PeftModel` + `merge_and_unload()`（`W' = W + BA` 融合） | 同左 |

- `strict=False` 在此是**必需项**：`mm_projector.bin` 只含投影层 key，其余层保持 base 原样，`strict=True` 会因缺 key 报错。
- 特殊 token 注册是**幂等**的：`add_tokens` 对已存在 token 返回 0，`resize_token_embeddings` 尺寸已对则不变。推理路径只加 `<im_patch>`（`mm_use_im_patch_token` 默认 True，`mm_use_im_start_end` 默认 False）。
- 训练时由 `initialize_vision_tokenizer`（`navid_arch.py:487`，**仅 `train.py:1269` 调用**）一次加 6 个特殊 token；它们随 tokenizer 一起保存（`train.py:1290` 把 tokenizer 传给 Trainer，HF `Trainer.save_model()` 会自动存 tokenizer），推理时由 `AutoTokenizer.from_pretrained` 读回。
- 视觉塔是**惰性加载**的（`navid_arch.py:42` 构造时传 `delay_load=True`），故需显式 `load_model()` 与手动 `.to(device, dtype)`；`image_processor` 挂在 vision_tower 上。**EVA-ViT 权重不在 NaVid 权重目录内**，需按 README 单独下载。

### `navid/model/language_model/llava_navid.py` 精读要点（2026-09-29）

- `LlavaConfig` 继承 `LlamaConfig`，只把 `model_type` 改成 `"llava"`；结尾的 `AutoConfig.register` 与 `AutoModelForCausalLM.register` 将配置类和模型类接入 Transformers 的 Auto 工厂。
- `LlavaAttLlamaModel(NaVidMetaModel, LlamaModel)` 把 NaVid 的视觉塔/投影层能力与 Llama Transformer 主干组合起来；`LlavaLlamaAttForCausalLM(LlamaForCausalLM, NaVidMetaForCausalLM)` 再把多模态能力与因果语言建模能力组合起来。
- `LlavaLlamaAttForCausalLM.__init__` 使用 `super(LlamaForCausalLM, self).__init__(config)` 跳过 `LlamaForCausalLM.__init__`，然后手动创建自定义的 `self.model = LlavaAttLlamaModel(config)` 和 `self.lm_head`，最后调用 `post_init()` 初始化完整模块树。
- `forward()` 的数据流为：推理模式下把输入搬到模型设备 → 调用 `prepare_inputs_labels_for_multimodal()` → 将得到的 `input_ids` / `inputs_embeds` 等交给 `self.model` → 取 `outputs[0]` 作为 hidden states → 经 `lm_head` 得到 `[batch, sequence, vocab]` 的 logits → 有 labels 时用 `logits[..., :-1, :]` 对齐 `labels[..., 1:]` 计算 next-token loss。
- `prepare_inputs_labels_for_multimodal()` 与 `prepare_inputs_for_generation()` 不是重复函数：前者在 `forward()` 内真正完成文本 token、视觉特征、特殊图像 token 的多模态拼装；后者是 Hugging Face `generate()` 使用的逐步输入调度接口，主要负责 `past_key_values` 存在时只保留最后一个 token，并把 `images` 等参数转交给 `forward()`。
- 当前仓库中 `agent_navid.py` 的 `self.model.generate(...)` 是 `prepare_inputs_for_generation()` 的实际间接入口；因此全仓库文本搜索可能只看到该函数定义，看不到显式调用，因为调用发生在 Transformers 的生成框架内部。
- 直接执行 `model(...)` 会走 PyTorch `nn.Module.__call__()` → `forward()`，不会自动经过 `prepare_inputs_for_generation()`；只有 `model.generate(...)` 的自回归生成路径才会经过该 hook。

### `prepare_inputs_labels_for_multimodal()` 的 prompt 兜底链路（2026-10-08）

- `if prompts is None and hasattr(self, 'prompts'):` 实现“显式实参优先、实例缓存兜底”：调用方传入非 `None` 的 `prompts` 时保持原值；未传时才尝试读取模型实例上的 `self.prompts`。
- `hasattr` 在读取 `self.prompts` 前检查属性是否存在，避免尚未调用 `update_prompt()` 时直接访问该属性引发 `AttributeError`。
- 实际推理链路为：`agent_navid.py` 在 `generate()` 前调用 `self.model.update_prompt([[cur_prompt]])` → `update_prompt()` 把提示词保存到 `self.prompts` → `forward()` 未显式传 `prompts` 时由这里取回 → 传给 `encode_images()` / `vlm_attention()`，后者据 `NAVIGATION_IDENTIFIER in prompt[0]` 判断是否为导航任务。

### `prepare_inputs_labels_for_multimodal()` 的提前返回与 KV Cache 分支（2026-10-08）

- 当视觉塔不存在、没有图像，或 `input_ids` 只有 1 个 token 时，函数不再编码图片和重建多模态 embedding，而是直接返回；其中单 token 情况对应 `generate()` 首轮之后的自回归解码步骤。
- 在“已有 `past_key_values` + 视觉塔和图片仍存在 + 当前仅 1 个 token”时，之前的文本与视觉信息已经保存在 KV Cache 中。代码把 `attention_mask` 重建为 `[batch_size, cached_sequence_length + 1]` 的全 1 张量，使当前 token 可以关注全部缓存位置；`+1` 代表本轮的新 token。
- `past_key_values[-1][-1]` 取最后一层的 value cache，其倒数第二维是缓存序列长度；返回元组第四项 `None` 对应 `inputs_embeds=None`，让下游 Llama 仅为当前 `input_ids` 做普通词嵌入，不重复插入视觉 embedding。

### `encode_images()` 的双路径视觉入口（2026-10-08）

- `encode_images()` 是像素/预计算特征的统一入口，不负责最终拼接文本 token。`long_video=False` 时，输入像素张量经 vision tower 从 `[总帧数, 3, 224, 224]` 编码为 `[总帧数, 257, 1408]`；`long_video=True` 时，`images` 已是这类视觉特征，直接赋给 `image_features`，避免重复执行视觉编码器。
- 两条分支都会调用 `vlm_attention()`。后者依据 `image_counts` 把展平的总帧重新分回各个 batch 样本，依据 `prompts` 是否含 `NAVIGATION_IDENTIFIER` 判断导航任务，再通过 `token_generation()` 压缩 patch token 并用 `mm_projector` 映射到 Llama 隐藏维度。
- 返回值不是一个裸张量，而是三个并行列表：压缩后的样本级视觉特征 `image_features`、控制后续单图/视频拼装分支的 `video_or_not`，以及导航任务当前帧的 64 个高分辨率 token `nav_or_not`（非导航样本为 `None`）。
- `long_video` 虽继续作为关键字参数传给 `vlm_attention()`，但当前函数体内没有读取它；它在现有代码里的实际作用仅是让 `encode_images()` 跳过 vision tower。变量名表达“长视频优化场景”，实际判据则是输入最后一维是否大于 1000，即借形状区分预计算特征与像素图。

### `vlm_attention()` 的样本分组与双分辨率导航 token（2026-10-08）

- 该函数名虽含 `attention`，实际没有 Q/K/V 注意力计算；主流程是：校验压缩配置与 batch 元数据 → 用 `image_counts` 从展平的总帧张量中恢复每个样本 → 根据 `NAVIGATION_IDENTIFIER` 识别导航 prompt → 去除 CLS token → 调用 `token_generation()` 池化和投影 → 返回视觉特征及后续拼装分支标记。
- `image_counts=None` 代表每个 `image_features[_idx]` 都是独立单图，代码用 `None` 索引补出帧维；提供 `image_counts` 时，`total_count` 是累计切片游标。例如 `[2, 3]` 会依次取得 `image_features[0:2]` 和 `[2:5]`。
- 假设某样本有 `F` 帧、内层 prompt 数为 `P`、Llama 隐藏维度为 `H`：池化投影后的 `[F, T, H]` 经 `[None] → expand(P, ...) → flatten(1, 2)` 变为 `[P, F×T, H]`，从而让同一份视觉内容可供该样本的多个 prompt 复用；导航任务强制 `P=1`。
- 普通单图固定采用 8×8 池化并输出 64 个 token，`video_or_not=False`；普通多帧视频每帧输出 `nav_size` 个 token，`video_or_not=True`；导航样本除保留所有帧的低分辨率 `nav_size` token 外，还从最后一帧额外生成 64 个高分辨率 token 放入 `nav_or_not`，即“压缩历史/全序列 + 精细当前观测”的双分辨率结构。
- 当前实现有四处代码现状需留意：形参 `long_video` 和局部变量 `final_token_length_lst` 均未使用；入口声称支持 `compress_type="mean"`，但 `token_generation()` 会执行 `int("mean")` 而报错；只校验 `len(prompts) == len(image_counts)`，没有校验 `sum(image_counts) == len(image_features)`；多项输入检查使用 `assert`，在 `python -O` 下会被移除。

### 模型架构归属（NaVid vs LLaVA）

- 文件头 `Copyright 2023 Haotian Liu`（LLaVA 作者）—— 本仓库模型代码是 **从 LLaVA fork 而来**，类名 `Llava*` 是未改净的痕迹。
- 三层继承（`llava_navid.py`）：
  - `LlavaConfig(LlamaConfig)`，`model_type = "llava"`
  - `LlavaAttLlamaModel(NaVidMetaModel, LlamaModel)`
  - `LlavaLlamaAttForCausalLM(LlamaForCausalLM, NaVidMetaForCausalLM)`
- `NaVidMetaForCausalLM` 是 **mixin**（能力包），与 `LlamaForCausalLM` 多继承组合。
- `LlavaLlamaAttForCausalLM.__init__` 用 `super(LlamaForCausalLM, self).__init__(config)` **故意跳过** `LlamaForCausalLM`，随后手动把 `self.model` 换成 `LlavaAttLlamaModel` —— LLaVA 招牌手法。
- 结尾两行 `AutoConfig.register("llava", LlavaConfig)` + `AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)` 把自定义类注册进 transformers 的 Auto 映射表。
- **`forward()`（`llava_navid.py:56-124`）是整个模型最好的地图**：① 设备搬运 → ② `prepare_inputs_labels_for_multimodal`（多模态拼装）→ ③ `self.model(...)` 过 Llama 主干 → ④ `lm_head` 出 logits → ⑤ 训练时算 loss。建议以此为导航图再读 `navid_arch.py`。
- 分辨归属的方法：`prepare_inputs_labels_for_multimodal` / `encode_images` / `mm_projector` 属 **LLaVA 标准**（有公开资料）；`vlm_attention` / `token_generation` / `update_prompt` / `process_grid` 属 **NaVid 自研**（必须读代码）。故 `navid_arch.py` 中真正需要啃的只是名字陌生的那几个方法。

### `navid/` 包的调用关系

- `agent_navid.py` **只直接 import 4 个模块**（`:14-17`）：`navid.constants`、`navid.conversation`、`navid.model.builder`、`navid.mm_utils`。
- 模型内核（`navid_arch.py`、`llava_navid.py`、`eva_vit.py`、`multimodal_projector/`）**agent 从未直接引用**，经 `builder.py:23` 的 `from navid.model import *` → `model/__init__.py` → `llava_navid.py:27` 间接加载。
- agent 之后只调模型对象的方法：`model.generate()`、`model.update_prompt()`、`image_processor.preprocess()`、`tokenizer()`。其中 **`update_prompt` 是 agent 唯一直接触碰 NaVid 自研代码的地方**。
- `navid/train/` 无任何外部调用者（仅 train 内部互相 import），推理链路完全不走。

### NaVid 权重的训练方式

- README:89 明文：权重在 **VLN-CE R2R 与 RxR 的 training split** 上训练，follow Uni-NaVid 的训练策略。
- `train.py` 默认值支持全参数微调：`freeze_backbone=False`、`lora_enable=False`、`tune_vision_encoder=False`、`tune_mm_mlp_adapter=False`。即 **LLM 全参数参与训练、视觉编码器冻结**（相当于 LLaVA 两阶段的第二阶段）。
- 目录名 `navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split` 编码了配置：7B / 224 分辨率 / video / 1 fps / grid-2 / R2R+RxR / training split。`full` 按 LLaVA 惯例指 full fine-tuning。
- **证据强度**：README 明文 + 代码默认值可靠；"实际训练确实用默认参数"属**推断**（训练命令行不在本仓库，README 指向 Uni-NaVid 另一仓库）。

### RxR 数据集背景（联网核实）

- RxR = **Room-across-Room**，Google Research EMNLP 2020，R2R 的续作，基于 Matterport3D。
- 126K 条指令（英/印地/泰卢固三种语言，母语者**独立撰写**而非互译）+ 126K 条 Follower 示范路径；平均指令 **78 词**（R2R 仅 29 词），约 R2R 的 10 倍规模。
- **Guide/Follower 双人标注**：Guide 边走边口述；Follower 仅凭音频去走，走通才算指令合格，失败则重标。成功标准为"走到最后一个全景点 **3m** 以内"——这正是 task yaml 中 `SUCCESS_DISTANCE: 3.0` 的出处。
- 在本仓库的对应：`ROLES: [guide]`（`task.py:131` `annotation_roles = ["guide", "follower"]`）、`LANGUAGES: [en-US, en-IN]`（`task.py:132` 含 hi-IN/te-IN）、`DATA_PATH` 中的 `{role}` 占位、`GT_PATH`（NDTW 用）。
- 对评测的影响：当前跑 `val_unseen`（未见过场景，VLN 论文标准口径）；只启用英语；指令长度是 R2R 的两倍多，对 VLM 上下文构成实际压力，也是 RxR 指标通常低于 R2R 的原因之一。
- 来源：https://github.com/google-research-datasets/RxR 、论文 https://ar5iv.labs.arxiv.org/html/2010.07954

### 本机运行环境缺口

- **habitat 未安装**：`python3 -c "import habitat"` 报 `ModuleNotFoundError`，`find / -maxdepth 8 -name habitat` 无结果，无虚拟环境。故 habitat 相关结论均来自**联网核实 v0.1.7 源码**，非本地读取。
- `uninavid/` 不存在（README:71 要求 `ln -s Uni-NaVid/uninavid uninavid`）。
- `model_zoo/` 不存在，权重未下载。
- 结论：**当前状态可读代码，不可跑评测**。
- 附带：本机 `python` 命令不存在，需用 `python3`。

### RGB 帧 → 视觉 token 的完整链路（2026-10-08）

三站式的转换，`encode_images()` 只是"观测编码入口"，真正把像素变成 token 的是它调用的视觉塔内部：

| 站 | 位置 | 动作 | 形状 |
| ------ | ------ | ------ | ------ |
| ① 预处理（纯图像库） | `agent_navid.py:63` `image_processor.preprocess` | 缩放 + 归一化 | → `(N, 3, 224, 224)` |
| ② ViT patch embedding | `eva_vit.py` 的 `PatchEmbed.forward`（`self.proj = nn.Conv2d(3, 1408, kernel_size=14, stride=14)`） | 14×14 卷积滑过 224×224 → 16×16=256 个 patch，每 patch 投成 1408 维 | → `(N, 256, 1408)` |
| ②' 加 CLS + 39 层 transformer | `eva_vit.py` `forward_features` | 拼 CLS、加位置编码、过 39 层 | → `(N, 257, 1408)` |
| ③ 丢 CLS | `navid_arch.py` `mm_vision_select_feature=='patch' and shape[1]%2==1` → `[:, 1:]` | 切掉首位 CLS | → `(N, 256, 1408)` |
| ④ 网格池化压缩 | `navid_arch.py` `process_grid` | 见下节 | 每帧 → 4/16/64 个 token |
| ⑤ 投影 | `navid_arch.py` `mm_projector`（`nn.Linear(1408 → 4096)`） | 投到 LLM 隐藏维度 | → `(N, T, 4096)` |
| ⑥ 拼进输入序列 | `prepare_inputs_labels_for_multimodal` 装配循环 | 作为 `inputs_embeds` 的一段 | 与文本 embedding 同维 |

- **丢 CLS 是硬前提，不是美化**：`process_grid` 要用 `int(shape[1] ** 0.5)` 反推网格边长再 `reshape` 成正方形，257 开方得 16 但 16×16=256≠257，不切必报形状错误。
- **丢 CLS 用的是奇偶启发式**（`% 2 == 1`），一箭双雕：既判断"CLS 还在"，又对**已切过 CLS 的预计算特征**（`train.py:996` 的 `video_info['feats'][:, 1:]`，256 为偶数）保持幂等、不切第二次。代价是默认了"CLS 必在首位 + patch 网格数为偶数"，网格若变奇数（如 15×15）或 pkl 未带 CLS 会静默误切。
- 训练/预计算特征近路：`long_video=True` 时 ①②③ 全部跳过（特征在数据准备阶段已算好），只在 `encode_images()` 里以 `if long_video: image_features = images` 体现。
- 角色澄清：`prepare_inputs_labels_for_multimodal` 是**装配器**（把视觉 token 拼进文本 token 序列、同步 labels/mask），不是观察编码器；观察编码器是 `encode_images` → 视觉塔 + 网格压缩 + `mm_projector` 这条链。

### `token_generation()` / `process_grid()` 的向量化压缩语义（2026-10-08）

- **没有逐帧循环**：进来的 `vis_embed` 是 `(帧数, 256, 1408)`，`reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)` 只把第 1 维（256 个 patch）拆成 16×16 网格，**第 0 维（帧）原样保留**；`F.avg_pool2d` 把第 0 维当 batch 维，逐帧独立池化，帧与帧之间不混合。所以"每帧压成 N 个 token"是靠维度语义天然实现的，不需要循环。
- 池化参数：`grid_stride = cur_shape // grid_size`，且 `kernel_size = stride = grid_stride` → **不重叠分块平均**，输出每帧恰好 `grid_size²` 个 token。实测对应：`grid:8`→64（当前帧/单图）、`grid:4`→16、`grid:2`→4（本仓库实际配置，即 `nav_size=4`）。
- 两步 `permute` 是配套的：进去时 `(0,3,1,2)` 把特征维挪到"通道"位以适配 `avg_pool2d` 的 `(N,C,H,W)` 约定，出来时 `(0,2,3,1)` 换回"最后一维是特征"并 `flatten(1,2)` 合并空间两维。
- **唯一"单帧特判"**是导航分支的 `vis_embed[-1:]`——用**切片**取末帧以保持 3 维（写成 `[-1]` 会降成 2 维，后续 reshape 网格会错），让它单独走 `grid:8` 拿到 64 个高分辨率 token。
- 三分支的判据是 `image_counts` 的**数值**而不只是有无：`None` 或（`==1` 且非导航）→ 单图模式 `grid:8`；`navigation` → 末帧 64 + 其余粗网格；否则 → 全部粗网格。

### `image_counts` 的两种对齐语义（2026-10-08）

- 它是**"每个样本各有多少帧"的账本**，创建于 `image_counts = [image.shape[0] for image in images]`，前置条件是该分支已把每个元素补成 4 维 `(帧数, 3, H, W)`，所以 `shape[0]` 就是帧数。名字里的 image 指"一个样本的图像输入"（可能是一段多帧视频），数的是**帧数**不是图片张数。
- 它是函数**默认参数**（`None`）：当上层传入的是"一整块张量"（走 `else` 分支）时，创建语句根本不执行，于是保持 `None` 传下来。故两种取值对应两条互斥的输入形态，而非同一件事的前后阶段：
  - `None` = 每 prompt 一条特征，**位置对齐**（`image_features[_idx, None]`，`None` 只为补回帧维使下游 reshape 成立）；
  - `list` = 帧已被 `torch.cat(dim=0)` 拍平，**区间对齐**（`total_count` 前缀和游标 + `image_counts[_idx]` 长度切片，循环不变式：进入第 `_idx` 轮时 `total_count == sum(image_counts[: _idx])`）。
- **长度契约 = batch size**：`assert len(prompts) == len(image_counts)`，三本账（`input_ids.shape[0]` / `prompts` / `image_counts`）同长，靠"都按 instances 顺序组装"对齐。推理时恒为 1（agent 一次一条对话）。
- 该数值还被另两处消费：传入 `token_generation` 决定压缩模式；以及在 `vlm_attention` 尾部决定 `video_or_not` 标记（`==1` 且非导航 → `False` 单图分支，否则 `True` 视频分支）。
- 期望的 `sum(image_counts) == len(image_features)` **没有断言**（只校验了 `len` 而非总和），是当前实现的校验缺口。

### attention mask 的归属与推理期重建（2026-10-08）

- **归属是 LLM 的**，不是 ViT 的：证据有三——(1) 它由 `forward()` 传入本函数、重建后作为返回值第 2 位喂给 `self.model`（`LlavaAttLlamaModel` = Llama 主干）；(2) 视觉塔那条路完全不接收 mask（`eva_vit` 只吃 `images`）；(3) 它的长度被断言等于"文本 + 展开后视觉 token"的序列长度，而该长度只存在于 LLM 输入层。
- **语义**：1 = 可见、0 = 屏蔽（padding）。训练/prefill 时它真正干活——屏蔽 batch 内右 padding、以及在序列前部为展开出来的视觉段补 `True`（因为图像展开发生在序列前部而 tokenizer 是右 padding）。
- **解码步为什么还要重建**：重建的是**全 1** 张量（数学上等价于无掩码，证明"不是模型需要掩码，而是接口要求长度合法"）。原因是 HF `generate` 自己维护 `model_kwargs["attention_mask"]`，其长度按**占位符展开前**的 token 数（agent 未显式传 mask，由 HF 自动造）逐step增长；而模型内部把 1 个 `-200` 展开成了几十上百个 token，于是 mask 长度 ≠ KV cache 长度，加到 attention 分数上会形状不匹配。由于 forward 的返回值不会回流进 generate 循环，**只能在 forward 内按 `past_key_values[-1][-1].shape[-2] + 1` 重建**。

### `navid/model/navid_arch.py` 行号基准更新（2026-10-08）

该文件在 2026-10-08 经历"随读补中文注释 + 一次纯空白格式刷新"，行号整体位移（现 870 行），本节之前各节引用的 `navid_arch.py` 行号可能已过期。**最新基准**：

| 符号 | 行号 |
|------|------|
| `class NaVidMetaModel` | 49 |
| `class NaVidMetaForCausalLM` | 109 |
| `encode_images` | 118 |
| `vlm_attention` | 144 |
| `token_generation`（含 `process_grid`） | 293（295） |
| `update_prompt` | 336 |
| `prepare_inputs_labels_for_multimodal` | 341 |
| `initialize_vision_tokenizer` | 805 |

## 技术决策

| 决策 | 理由 |
|------|------|
| 保留 upstream 而非删除原 origin | 保留同步上游新提交的能力，同时让裸 `git push` 默认打到自己的 fork |
| origin 使用 HTTPS 而非 SSH | 本机无 GitHub 专用 SSH key，且 github.com 已有 `!gh auth git-credential` 专属 helper，HTTPS 链路更可靠 |
| PWF 三件套放仓库根 | 遵循全局 Agent Markdown 的 L2 约定 |
| `eval_navid_vlnce.sh` 的 7 个参数注释写在命令上方而非逐行上方 | bash 以 `\` 续行的命令中间不能插入注释行，`#` 会被当作普通参数传给 python；续行状态下无法逐行注释 |
| `builder.py:122` 的 `context_len3` 缺陷先报告不直接改 | 用户正在编辑器中编辑该文件，直接改可能与未保存内容冲突 |
| habitat 相关结论标注为"联网核实" | 本机未安装 habitat，无法本地读源码，需明确区分证据来源 |
| 对 habitat 源码的版本判断锚定 v0.1.7 | README 明确要求该版本；新版已从 yacs 迁到 OmegaConf，API 不兼容，混用会得出错误结论 |
| 纯空白格式刷新用「AST 指纹 + token 流比对」做零源码改动证明 | 空白改动的风险是误伤源码；`ast.dump(ast.parse(src))` 的 sha256 相同保证语法树一致，剔除 INDENT/NEWLINE 类空白 token 后逐 `(type, string)` 比对（含 COMMENT）保证连注释都未变，再叠加 `py_compile` 三重保险 |
| PWF 中的源码行号以「最新基准表」为准 | 文件补注释/格式刷新会使旧行号整体位移，逐个回改历史条目等于覆盖历史；改为在 findings 追加一张最新行号基准表，并声明此前引用可能过期 |

## 遇到的问题

| 问题 | 解决方案 |
|------|---------|
| `gh repo fork` 报 `--remote flag is unsupported when a repository argument is provided` | 改为在仓库目录内运行 `gh repo fork --remote=false`，不带仓库参数，之后手动配置 remote |
| `git remote rename origin upstream` 连带把 main 跟踪改成了 upstream/main | 显式执行 `git branch --set-upstream-to=origin/main main` 重新绑定 |
| 当前仓库无 PWF 三件套 | 本次会话首次创建，遵循模板并填入实际内容 |
| 本机未安装 habitat，无法读源码验证结论 | 改用联网核实 habitat-lab v0.1.7 对应文件（`config/default.py`、`core/env.py`、`core/dataset.py`、`datasets/registration.py`），并在 findings 中标注来源 |
| 本机无 `python` 命令 | 改用 `python3`（`python3 -m py_compile` 验证语法） |
| 编辑 `findings.md` 时误删 `## 技术决策` 标题 | 立即用 grep 检查章节结构发现，恢复标题并补入本会话决策 |

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
