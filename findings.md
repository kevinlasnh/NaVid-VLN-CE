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
