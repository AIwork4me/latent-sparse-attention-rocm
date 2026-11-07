# SESSION_REPORT — 2026-11-06（F10/F11/F12 收口 + PR #1 独立终审）

**分支/HEAD：** `v0.2.1-scientific-validity` @ `e243ba4e03f996e8650b4eb143dd678eff13c643`（已推送）
**PR：** #1（Draft，共 14 commits，65 文件）· main `5f2918e3` · 上游 pin `5ebea732`（== 上游 main）
**主环境：** ROCm 10.1.0 / torch 2.14.0+rocm10.1.0 / gfx1151（Radeon 8060S, 32GiB UMA）

## 已完成（全部有真机/线上证据）

1. **F10（P0）公共仓库 runner 安全** — 修复前 `pull_request` 触发器可把任意 fork PR
   代码路由到持久 gfx1151 主机（该机 runner 以交互用户 `amd` 运行，`$HOME` 内有
   gh CLI 活凭据）。修复：GPU workflow 删除 `pull_request`，仅 push(main)/确认式
   dispatch；新增 GitHub-hosted 纯 CPU 的 `cpu-pr.yml`（零密钥、contents:read、GPU
   测试响亮跳过）；actions 全部 SHA 锚定；`check_workflow_security.py`（R1–R5）进
   CI 防回退（它自身当场抓到 upstream-watch 未锚定的 checkout）；新增
   `docs/SECURITY_RUNNER_POLICY.md`（威胁模型 + 主机实测 + 所有者加固清单）。
   **线上验证：** 本次 push 的 PR 事件只跑了 CPU 工作流；受信 dispatch 在 PR 分支
   成功（run 42139297335，回执本地重审计 APPROVED）；`confirm_ref` 不匹配的
   dispatch 20 秒内失败关闭（run 42139650668）。
2. **F11（P0）战役失败关闭** — 已配置且已执行的 Kernel/E2E 适配器只要不是"已验证
   PASS"（含 BLOCKED/exit 78、崩溃、超时、回执缺失/截断/跳过门/上游 SHA 缺失、
   执行前失败）一律使战役退出码非零；无适配器却记 PASS 判为伪造证据；
   `campaign.sh` 此前"打印 rc 后永远退出 0"，现逐阶段聚合并传播。新增 53 项测试
   （30 状态参数化 + 15 execute_stage 负路径 + 8 shell 传播，沙箱化）。
3. **F12（P0）独立回执审计器** — `lsa_rocm_lab/evidence/auditor.py` + CLI：不信
   适配器任何断言，从原始 stage 行**重算全部 speedup**（1e-9 容差），复核 16 门
   清单/上游+仓库 SHA/gfx1156 设备/可追溯 run_id/精确 CLI、行级校验（有限正数、
   ≥10 样本、p50∈[min,max]、单一 seed/dtype/几何、稀疏预算、dense decode
   query_len==1），并拒绝：改写加速比、调换基线、重复/缺失/歧义行、NaN/Inf、重放
   run_id、损坏 SHA256SUMS、"预构建索引冒充完整 LSA"。30 项对抗样本测试。
   **对 6 份真实回执全部 APPROVED（重算 111 组对比）**；`render_tables.py` 与 GPU
   CI 现在拒绝未 APPROVED 的回执；BENCHMARKS.md/FINAL_REPORT.md 依此重建。
4. **评审中新发现并修复（MEDIUM）** — `checkout_upstream.sh` 原默认浮动到上游
   main，上游一动 CI 就悄悄测未评审 SHA；现默认 pin `tracking/UPSTREAM_BASELINE.json`
   的已评审 SHA。CPU-PR 首跑缺 PYTHONPATH/triton/上游 checkout，已修复（绿，59s）。
5. **本地真机回归（HEAD 2139758）** — 16/16 门 PASS；上游测试 203 通过/7 分类/0
   未分类（rc=0）；pytest **133 全过**；新产出 2048 回执经校验+审计 APPROVED
   （`evidence/receipt-2139758-postfix-verification.json`）。
6. **独立终审** — `docs/reviews/PR1_FINAL_INDEPENDENT_REVIEW.md`：核心数值路径
   （dense_read 修复、bench 匹配、稀疏读内核的掩码/越界/int64、审计器、runner）
   逐一重推导；结论 **GO（可合并，限 v0.2.1 有效性范围）**，遗留 L1–L5（皆 LOW/
   记录项，不阻塞）。**未合并，等待所有者。**
7. **持久化台账** — `tracking/PROJECT_STATE.json`、`tracking/MILESTONES.yaml`、
   `docs/CONTINUATION.md`（含唯一下一步）、`docs/DECISION_LOG.md`、
   `docs/EVIDENCE_INDEX.md`。

## 测试计数

- 单元/负向/对抗：**133 passed / 0 failed**（本机 GPU 全量）
- 真机门：16/16；上游：203/7/0；CI：CPU 绿 + 受信 GPU 绿 + 守卫负例按设计失败

## 性能口径（审计器重算，非转述）

32k bf16 decode select+read（预构建索引，非 TTFT/E2E）三次独立进程中位数：
block-sparse ≈**3.9×**、flat ≈**2.1×**（跨次 1.5–2.4）、tree ≈**0.9×（不快于
dense）**；prefill ≈**0.19×**。v0.2.0 的 5.04×/7.86×/2.18× 已作废归档。
已知局限：三次重复共用同一 seed（L3），v0.3.0 首项即多种子研究。

## 安全态势

- 不可信 PR→持久 GPU 通路：**已消除并线上验证**（正向+反向）。
- 所有者待办（不阻塞本 PR）：runner 迁出交互用户、轮换 gh 凭据、
  `gpu-regression` 环境加审批人、可选 repo 级 SHA-pin 强制（策略 §4 给出可回滚命令）。

## AWAITING_OWNER_APPROVAL

- 合并 PR #1（保持 Draft 直至所有者决定）；合并后在 main 新 SHA 上跑受信 GPU
  工作流，绿了才可打 `v0.2.1` 标签。

## 下一步（唯一最重要动作）

开 `v0.3.0-kernel-optimization` 分支写 `docs/OPTIMIZATION_PLAN_v0.3.md`（首任务：
多种子研究 + 读内核剖析），对 main 开 Draft PR 并注明 staged behind PR #1。
详见 `docs/CONTINUATION.md`。
