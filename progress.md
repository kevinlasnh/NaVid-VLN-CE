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

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| gh CLI 登录状态 | `gh auth status` | 已登录且具备 repo 权限 | 登录 kevinlasnh，token scopes 含 `repo` | 通过 |
| fork 写权限 | `git push origin HEAD:refs/heads/tmp-verify-push-access` | 推送成功 | 推送成功，远程提示可创建 PR | 通过 |
| 临时分支清理 | `git push origin --delete tmp-verify-push-access` | 删除成功 | 删除成功，fork 仅剩 main | 通过 |
| fork 权限查询 | `gh api repos/kevinlasnh/NaVid-VLN-CE` | push 权限为 true | `{"admin":true,"maintain":true,"pull":true,"push":true}` | 通过 |
| 凭据链路 | 检查 github.com credential helper | 存在可用 helper | `!gh auth git-credential` 已配置 | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-09-28 11:17:43 +0800 | `gh repo fork` 报 `the --remote flag is unsupported when a repository argument is provided` | 1 | 去掉仓库参数，在仓库目录内执行 `gh repo fork --remote=false`，再手动配置 remote |
| 2026-09-28 11:17:43 +0800 | `git remote rename origin upstream` 自动把 main 的跟踪改为 `upstream/main` | 1 | 显式执行 `git branch --set-upstream-to=origin/main main` 重新绑定 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 2（主调用链精读）刚起步，已完成阶段 0 与阶段 1 |
| 我要去哪里？ | 阶段 2 精读 → 阶段 3 模型机制 → 阶段 4 环境侧 → 阶段 5 对比 → 阶段 6 沉淀 |
| 目标是什么？ | 系统读懂 NaVid-VLN-CE 的完整推理链路与核心机制 |
| 我学到了什么？ | 见 findings.md（token 压缩、历史帧复用、动作离散化、仓库运行缺口、fork 配置） |
| 我做了什么？ | 完成 fork 与远端重绑并验证写权限；完成仓库侦察；建立 PWF 三件套 |

---
*每个阶段完成后或遇到错误时更新此文件*
