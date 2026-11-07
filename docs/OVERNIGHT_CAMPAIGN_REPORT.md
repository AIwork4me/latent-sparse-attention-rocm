# 夜间自治工程战役报告 — campaign-20261106T231500Z

## A. 执行摘要

- **开始/结束时间**:2026-11-06 23:12 → 2026-11-07 05:30(本地),实际约 6.3 小时(硬上限 10 小时,未触顶)
- **硬件/软件**:AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S(gfx1151)/ ROCm 10.1.0 / PyTorch 2.14.0+rocm10.1.0 / Triton 3.8.0+git669b31ac / Python 3.12.3;全程 GPU 温度 49–55°C,无限流
- **起始 GitHub 状态**:PR #1(v0.2.1)Draft @ 79412f8;PR #5(v0.3.0 计划)Draft @ ca8bc70
- **结束 HEAD**:v0.3.0-kernel-optimization @ 47e63f7(代码头 60c004b,其后均为纯文档/证据提交)
- **实际完成的最高里程碑**:v0.3.0 全部(T1/T2/T3/T4)+ v0.3.1(topk 选择器)——**REVIEW_READY**
- **正在实施的下一里程碑**:v0.3.1 后续(fused pack→expand)或 v0.4.0 长上下文阶梯
- **总体判定**:**GO**(作为 PR #5 内容提交 owner 评审;未合并、未打 tag——按策略等待 owner)

## B. 已完成工作

| 任务 | 实际实现 | 提交 | 结果 |
|---|---|---|---|
| 战役前置 | 杀掉前夜遗留的孤儿 GPU 进程(100% CPU × 25h,违反 10h 上限,污染 UMA 带宽),完整取证 | 见 MANIFEST/LEDGER | 主机干净后再开测 |
| T1 多种子研究(issue #2,已关闭) | 3 种子 × 3 独立进程 × **3 轮研究 = 27 张回执**,全部审计 APPROVED(每比较独立重算);诚实方差带取代单种子标题;聚合器+运行器+失败即败测试 | 2daaebf, 265d97e 等 | **32k 解码:flat 1.94×(带 1.85–2.06),block-sparse 4.29×(4.24–4.38,稳健),tree 0.93×(仍慢于稠密)**;v0.2.1 的"2.1×"确证为种子运气 |
| T2 剖析归因 | 冷/热分离 + torch.profiler 内核表 + 内核 ID;剖析器虚高缺陷被识别并写明方法 | 3e093a8 | sparse_read 预填充 299ms = flat 步 77%;tree select 19,940 launches/iter;select 排序+拷贝链 ≈ 48% |
| T3 平铺稀疏读(已入产) | 行平铺 Triton 内核(TML4/CN32 预填充,TM1/CN64 解码)成为生产路径;新 gate H(清单 16→17);零长度 NaN 契约修复;同输入 A/B 每配置先过正确性 | 4988672 + 修复 | 预填充 1.24/1.18/1.14×(8k/16k/32k);**32k 预填充 0.19×→0.210×**;解码按设计持平 |
| T4 树束内核化(诚实负结果) | 页批处理 beam(修复了再测量时暴露的 picked-有效性 bug,集合同等);承诺工件 scripts/ab_tree_select.py | 1e73c3a | 批处理 **1.16–1.86× 更慢** → launch 开销不是成本;生产保持逐页;Triton 移植为已测量的后续杠杆 |
| 独立对抗评审 | 全新上下文代理评审,自跑负面测试+重审计全部回执;GO-WITH-FINDINGS;F1–F7 全部关闭并以最终 SHA 重发 11/11 回执 | 1e73c3a, 25c7bbc | 见 docs/reviews/V0_3_INDEPENDENT_REVIEW.md |
| v0.3.1 topk 选择器 | best_first_keys:打包键列唯一 ⇒ topk ≡ sort+slice(5 个等价测试钉死);同进程 A/B | 60c004b | **解码 select 0.266→0.214ms(1.24×)**;预填充持平;证据循环全量重发 |

## C. 性能与质量(仅限已验证证据)

- **正确性门**:17/17 PASS(默认与 LSA_TREE_IMPL=batched 双路径);pytest **181 通过**(战役前 133)
- **解码 @32k bf16(带,非单值)**:flat **1.94×(1.85–2.06)**;block-sparse **4.29×(4.24–4.38)**;tree **0.93×(0.92–0.99)**
- **预填充 @32k**:flat **0.210×**(对稠密仍慢——如实声明;较 0.19× 提升来自平铺读内核)
- **多种子方差**:3 轮研究一致显示 dense 基线对种子稳定(±2.5%),离散度来自稀疏路径本身
- **审计**:战役累计 38 张回执、4 项研究,全部独立审计 APPROVED;BENCHMARKS.md 全部由已审计回执生成
- **内存**:32k 全程无 OOM;发现并规避了页批处理 einsum 的多 GiB 中间量(UMA 抖动)
- 未测量 E2E/模型级指标——不声明(v0.5.0+/v0.7.0 范围)

## D. 剩余阻塞

| 类别 | 项 |
|---|---|
| 需 owner 批准 | PR #1 合并 + v0.2.1 tag;PR #5 评审(合并条件:在最终合并 SHA 重发 17/17 回执) |
| 需 owner 操作 | Runner 加固(docs/SECURITY_RUNNER_POLICY.md §4,issue #3) |
| 外部依赖 | SGLang gfx1151 通道(v0.6.0,维持 BLOCKED_ON_REASSESS) |
| 工程(已测量排序) | select 拷贝链融合(pack→expand);Triton beam 移植;解码 select 融合;共享块去重(研究级) |
| 科学不确定性 | flat 解码加速的种子敏感性已量化为带;16k 处 flat≈0.8–1.3× 波动大 |

## E. 下一夜恢复指令

1. `cd ~/Desktop/lsa-rocm-eval-starter && git log --oneline -3`(应为 47e63f7 文档链,代码头 60c004b)
2. `gh pr view 1 --json state,isDraft` — owner 是否已合并 PR #1?若已:按任务书 §1 校验合并 SHA、验证 main GPU 门、按 §13 处理 PR #5 基
3. 未合并则继续独立工作:从 `v0.3.0-kernel-optimization` 切 `v0.3.1-select-fusion`(fused pack→expand,docs/OPTIMIZATION_v0.3.md §h-1)或开 `v0.4.0-long-context`(内存可行性预检 → 64k 阶梯)
4. 基线校验:`pytest 181 通过`、`gates 17/17`、`check_workflow_security.py` 绿、`gh run list` CI 绿
5. 证据入口:docs/EVIDENCE_INDEX.md(最新在最上)、docs/CONTINUATION.md(完整恢复点与本夜教训)

**本夜未做任何合并、tag、发布或特权操作。所有性能声明可由 evidence/ 下 SHA256 校验的回执独立复算。**

## F. 战役后修订(2026-11-07 发布就绪批次,原文以上保持不可变)

- **SHA 台账明确化**:生产代码 SHA 为 `60c004b`(全部性能回执钉在此);`0908842` = `60c004b` + `_tree.py` 纯 docstring 修订(证明无行为变化:双路径 17/17 门 + 186 单测重验);`5a13bf0` = 仅测试可移植性修复;其余均为纯文档提交;**最终合并 SHA 仍未定**(合并条件:在该 SHA 重发 17/17 回执)。
- **GitHub CPU CI 修复**:运行 42260195840 / 42260187512 在 `2f3002c` 失败——`tests/test_select_topk.py` 硬编码 `device="cuda"`,在 CPU-only torch 上分配即抛错,恰 5 例。`5a13bf0` 设备参数化(cpu 恒跑 + ROCm 可见时加跑);无 skip/xfail/放宽断言;并列分数、EMPTY、k==width、参差行、生产形状覆盖全部保留。本地 CPU-wheel venv 完整复现(修前 5 failed/130 passed/46 skipped 与 CI 一致;修后 **135 passed / 46 skipped / 0 failed**);GitHub 绿:运行 42371335910 / 42371339224。gfx1151 上套件 181 → **186**(每用例 cpu+cuda 双跑)。
- **本会话未做任何新性能测量**:v0.3.1 回执核验为 11/11 APPROVED、153 比较独立重算(索引 EV-2026-11-07-J);`evidence/SHA256SUMS` 修复 2 行错误路径前缀(哈希未动)后 **71/71 OK**,历史证据字节不变。
- **T4 注释纠正**:`_tree.py` docstring 曾写"0.81-0.87x"(早期被否决的测量),已按审计工件 `evidence/receipts-v0.3.0/ab-tree-select.json` 更正为批处理 1.16–1.29×/1.78–1.86× 更慢。
- PR #5 标题与描述已改写为实现态(正/负结果、测试配置、证据链接、已知限制、owner 合并条件);仍为 Draft。§E 的恢复指令以 docs/CONTINUATION.md(2026-11-07 版)为准。
