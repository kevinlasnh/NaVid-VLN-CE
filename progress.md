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
| 我在哪里？ | 阶段 2（主调用链精读）进行中：已完成 `run.py` 与 `navid/model/builder.py` 的精读，下一步进入 `llava_navid.py` 与 `navid_arch.py` |
| 我要去哪里？ | 阶段 2 收尾（llava_navid.py → navid_arch.py）→ 阶段 3 模型机制 → 阶段 4 环境侧 → 阶段 5 对比 → 阶段 6 沉淀 |
| 目标是什么？ | 系统读懂 NaVid-VLN-CE 的完整推理链路与核心机制 |
| 我学到了什么？ | 见 findings.md（配置五层合并、yacs CfgNode 体系、早停语义修正、三处代码缺陷、VLN-CE 三层职责澄清、RxR 背景、builder.py 实际执行路径） |
| 我做了什么？ | 完成 fork 与远端重绑并验证写权限；完成仓库侦察；建立 PWF 三件套；完成 `run.py` 与 `builder.py` 逐行精读并补注释；厘清 habitat/VLN-CE/NaVid 分层职责 |

---
*每个阶段完成后或遇到错误时更新此文件*
