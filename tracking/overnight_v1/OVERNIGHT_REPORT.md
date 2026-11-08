# 夜间自主工程报告 — LSA ROCm gfx1151 overnight_v1（2026-11-08 → 11-09）

执行者：ZCode（owner 指示以本机 ZCode 会话替代 codex exec；MISSION 边界全程遵守）
任务书：`lsa_gfx1151_overnight_v1_kit.zip / CODEX_GFX1151_OVERNIGHT_V1_MISSION.md`
工作树：`~/lsa-v1-overnight` · 分支 `exp/gfx1151-v1-night-20261108-2155` · 基线 main `83e410b`（= 任务书起点，无漂移）· upstream pin `5ebea732`
夜内提交链：`614becc`（P0 ledger）→ `bf36917`（M1）→ `9c1a431`（M2/M3 harness + M4）→ `1465e5f`（M2 findings）→ `e2b6ab5`（M3 pilot）→（收尾提交见文末）
未推送（push 未执行；分支保留在本地，随时可推）。**未动 main、未动 PR#8（仍 DRAFT）、未 sudo、未发布任何 tag/Release。**

## 总览：计划 vs 执行

| 工作流 | 计划 | 执行结果 |
|---|---|---|
| P0 初始证明与 ledger | 强制 | ✅ 完成（起点无漂移、evidence 28/28、GPU 独占确认） |
| M1 合并树 GPU 资格验证 | 最高优先 | ✅ **PASS**（详见 §2） |
| M2 64k→256k fail-closed 扩展 | 核心 | ✅ **PASS**（含重要负发现，§3） |
| M3 真实模型质量（v0.5 目标） | 目标 | 🟡 **PILOT_DONE / CAMPAIGN_NOT_RUN**（集成已证明，§4） |
| M4 SGLang 可行性（v0.6 目标） | 只读矩阵 | ⛔ **BLOCKED_WITH_ROOT_CAUSE**（§5） |
| M5 E2E 服务优化 | 依赖 M3/M4 | ⛔ 未运行（M4 阻断；M3 标注 MODEL_ONLY_NOT_SERVING） |
| M6 可靠性/复现/DX | 按时间 | ✅ soak + 负控制 PASS（§6） |
| M7 v1.0.0 资格 | 只出包不发布 | ✅ 清单交付，结论 **NO-GO**（§7） |

## 1. 环境（全部实测验证）

- GPU：AMD Radeon 8060S（gfx1151）· torch 2.14.0+rocm10.1.0 · hip 7.16.26385 · Triton 3.8.0+git669b31ac
- 内存 94 GiB（起跑 88 GiB available）/ 磁盘 76 GiB 空闲 / GTT 上限 100 GiB
- GitHub GPU run `42483727525`：`pending` = **未运行，绝不计 PASS**；未尝试审批、未建第二监听
- 变通记录：git HTTPS fetch 间歇失败（GnuTLS -110）→ `http.version=HTTP/1.1` 一次成功（DECISIONS D002）；pip 默认源 stall → aliyun 镜像（D005：transformers/pyarrow 装入既有用户 venv，非全局，torch/numpy 未动，验证通过）

## 2. M1 — v0.4.0 合并树物理资格（全绿）

| 检查 | 结果 |
|---|---|
| 全量 GPU 测试套件 | **314/314 PASS**（0 skip；PR9 新增 56 个 release-integrity 用例 = 258+56；GPU 真实执行有运行时证据：Triton fused wide-fallback UserWarning 源自 `pipeline.py:305`，CPU 不可能触发） |
| 17-gate × {default, `LSA_TREE_IMPL=batched`, `LSA_SELECT_IMPL=fused`} | **17/17 × 3 PASS**（receipts `m1-gates-*.txt`） |
| evidence 完整性 | `SHA256SUMS` **28/28 OK** |
| 聚合独立重算 | **60/60 cells** 从 9 个 per-seed 原始 receipt 二次实现重算完全复现（容差 5e-4）；全部 correctness pre+post = true；全表最小 pooled speedup **1.0469**（无 <0.98 回格） |
| 16k/32k bf16 A/B 单 seed 确认 | **12/12 rows byte-exact pre+post**；32k prefill select **2.951×**（带 2.952–2.969）、32k prefill s+r **1.226×**（带 1.225–1.228）、16k prefill s+r **1.180×**（带内）；decode 各格 1.16–1.26×（run-to-run 噪声内，无 <1.0 格）；launch 计数与 campaign 一致。receipt sha256 `9b2026b2…`。**明确标注：单 seed 单进程确认，非新 3×3 campaign** |
| workflow 安全 guard | PASS |

v0.4.0 发布判定（`V040_RELEASE_GO_NOGO.md`）：**工程 GO / 执行仍 owner 门控**（run 42483727525 pending + runner 迁移未做 + tag/Release 需 owner）。未发布。

### 2a. Fresh 3×3 campaign（后半夜追加，M1 的最强证据）

在合并树上新跑**真正的 3 seeds × 3 独立进程 campaign**（`m1-fresh-s{seed}-rep{1,2,3}.json` ×9，16k+32k bf16，warmup 5 / reps 20，全部 row byte-exact pre+post）：

| cell | fresh pooled | 已发表 pooled | 判定 |
|---|---:|---:|---|
| 32k prefill select | **2.964×** | 2.962× | in |
| 32k prefill s+r | **1.227×** | 1.226× | in |
| 16k prefill select | 3.568× | 3.545× | in（更快） |
| 16k prefill s+r | 1.179× | 1.178× | in |
| 32k decode b1 s+r | 1.208× | 1.199× | in |
| 其余 7 cells | 1.17–1.37× | 同量级 | 4 in / 3 NEAR（≤±3% run-to-run 噪声，其一为更快方向 outlier） |

**12/12 cells 无回归（min pooled 1.1714 > 0.98 线）；headline 精确复现。** 今晚的 M1 结论由"单 run 确认"升级为"合并树上的真 campaign 复现"。

## 3. M2 — 64k→256k fail-closed 阶梯（receipt `d0bf6477…`）

方法：`scripts/ctx_scale_ladder.py`（投影预检拒跑、chunked reference oracle 且 8k 处断言等价、逐臂 OOM 分类、逐形状内存包络）；seed 424243、warmup 5 / reps 20、交错双臂、生产入口。

**正确性**：15/15 实测 phase 行 fused == reference **byte-exact + 3 次重复确定性**（64k–256k 全部 decode + 64/96/128k prefill）——wide/sampled 回退契约在真实尺寸成立。

**性能（负发现，诚实上报）**：ctx ≥ 65536 进入 wide path（columns > 4·top_p，kernel A 不适用，fused 走回退）后 **fused 系统性慢于 reference，且随 context 恶化**：

| ctx | prefill select | prefill s+r | decode b1 s+r | decode ragged s+r |
|---:|---:|---:|---:|---:|
| 64k | 0.335× | 0.669× | 1.045× | 0.878× |
| 96k | 0.256× | 0.610× | 0.938× | 0.843× |
| 128k | 0.212× | 0.560× | 0.916× | 0.828× |
| 192k | — OOM | — | 0.843× | 0.786× |
| 256k | — OOM | — | 0.815× | 0.699× |

**生产结论（证据支持）：fused 保持 opt-in 且限定 ≤32k 阶梯（那里 1.18–1.23× 收益）；≥64k 保持 reference。** 独立支持 PR#8 的窄范围。今晚未改内核（任务书：只优化 profiled 瓶颈——wide 回退成本现为已测、已登记的后续实验候选）。

**内存包络**（UMA，GTT 100 GiB 上限）：decode 全阶梯 <1 GiB（长上下文 decode 今天就可行）；prefill 双臂峰值 64k 18.3 GiB / 96k 40.1 GiB / 128k 70.3 GiB；**192k/256k prefill OOM（已分类：19–38 GiB 请求 vs 3–14 GiB 空闲；256k 连 fused 的 fp32 score 矩阵都放不下）→ >128k prefill 只能走 chunked/streamed 设计**（任务书所述独立车道）。

## 4. M3 — 真实模型集成（TinyLlama-1.1B-Chat，Apache-2.0，本地缓存）

- **可接通性：已证明。** 全部 22 层 attention 经 transformers 5.22 官方 custom-attention 接口运行本 repo 的 `flat_select`+`lsa_read`；prefill 与逐 token decode（真 KV cache，`model.generate`）均工作。接口数学：select-everything vs dense 因果 **cosine 0.9999973 / max_abs 0.00195**（bf16）。模型内 reference vs fused select：层 7 输出 **byte-equal**。
- **质量（pilot 扩样后，未训练 index）**：ppl（wikitext-2 test，**48×2048 = 98,304 tokens**）：dense 7.97592 / lsa_full 7.97635（+0.0054%）/ lsa_1024 7.97650（+0.0072%）/ lsa_256 7.97638（+0.0058%）——**稀疏损失 ≤0.007%**（12 窗口版与其一致；window 512+sink 保底近期性，wikitext 局部性强，不可外推长程任务）。NIAH **15 例/臂**：dense 10/15 = lsa_1024 10/15 = lsa_256 10/15，且**逐例命中集合三臂完全相同**（错的都是模型自身错例，稀疏化零检索行为改变；PILOT_ONLY）。**v0.5 不判 PASS**：预注册的 ≥100 例/上下文 RULER/NIAH campaign 未运行。
- **集成成本（MODEL_ONLY_NOT_SERVING）**：12 窗口 wall 13.9s（lsa_1024）vs 9.8s（dense）——每层每 forward 重建 bucket（p50 11.1ms）；峰值 2.78 vs 3.89 GiB。
- **产品缺陷发现**（跑真模型的价值）：① `build_buckets_gpu` 要求 T ≥ 512（transformers 分块 prefill 的 256 尾块触发 Ward 0 块崩溃；真服务早期 decode 也会踩）→ 今晚 adapter 层数学等价回退+计数，**工程后续项：port 短序列支持**；② Ward kernel index_dim ≤256（pow2 16–256）：零训练 full-dim regroup 仅适用 Hkv·head_dim ≤256（TinyLlama 4×64=256 恰好）；**Qwen3-0.6B（8×128=1024）被阻断**，需 upstream 的训练投影路线（与其 Qwen3.5 设计一致）。
- 模型选择记录：Qwen3-0.6B 原选（0.5–3B 带内）因 ② 改为 TinyLlama-1.1B（同带内、Apache-2.0、本地缓存）。

## 5. M4 — SGLang（BLOCKED_WITH_ROOT_CAUSE，宿主未动）

根因四连：① Docker 29.4.3 在但**零本地镜像**——无预批准 gfx1151 sgl-dev 容器，拉多 GB 镜像被任务书禁止；② 官方 gfx1151 车道（ROCm 7.2.4 镜像系）与本机 pip ROCm 10.1 venv 是**不同容器车道**，禁止混装；③ 本 repo port **无 SGLang 集成点**（仅 docstring 提及）；④ upstream `lsa/sglang` 面向 Qwen3.5 级 partial-rotary 模型 + SGLang 服务器栈（RadixAttention KV 池/KeySlot），非今晚可达。M3 的 transformers 前向集成**不隐含 serving**。

## 6. M6 — 可靠性/负控制/复现

- 负控制 PASS：`LSA_SELECT_IMPL=bogus` → loud ValueError（无静默 PASS）；`HIP_VISIBLE_DEVICES=""` → loud RuntimeError（无假 GPU 收据）。skip-to-PASS 防御已由 314 套件中的 release-integrity 测试钉住。
- 10 分钟有界 soak（`m6-soak.json`，sha256 `64ed5426…`）：**148/148 轮全绿**（每轮 17/17 gates，~4s/轮），allocator 零增长（每轮 empty_cache 后 reserved 0.0 MiB，leak_ratio 0），GPU 温度稳定 55–56°C，VRAM free 恒定 90.92 GiB。
- 复现手册 `HOW_TO_REPRODUCE_V1.md` 已交付（含全部命令、预期数字、跨硬件 NOT_TESTED 声明）。

## 7. M7 — v1.0.0 资格判定（`V100_QUALIFICATION_CHECKLIST.md`，仅判定包，未发布）

**v1.0.0：NO-GO。** 已达标：合并树+物理 GPU 正确性、证据哈希、长上下文包络、SGLang 如实 BLOCKED、来源 pin、无安全回归、回滚文档。未达标（每项都阻断）：预注册质量门（需 ≥100 例 campaign + 理想为训练 index 投影）、诚实 E2E serving 数字（M5 未跑）、owner 门（runner 迁移 + post-merge GPU run 42483727525 仍 pending + v0.4.0 发布授权）。最小转 GO 路径见 checklist 四步。

## 8. 失败/负实验与工具缺陷（全记录于 EXPERIMENT_REGISTRY + 各 FINDINGS）

1. wide path（≥64k）fused 全面退化（M2，核心负发现）
2. 192k/256k prefill OOM（内存上限的诚实边界）
3. `build_buckets_gpu` T<512 崩溃（M3 发现，adapter 规避，未修 port）
4. Ward index_dim ≤256 阻断 Qwen3 零训练路线（M3）
5. ctx_scale_ladder 的 lane 关闭判定缺陷（OOM 行被标 FAILED，192k 后 256k prefill 多试一次 ~30s，无正确性影响——已记录）
6. NIAH 首版 harness 3 个 bug（parquet 读取、decode 行切片、Tq=0 guard）——全部修复后重跑
7. git HTTPS/TLS 间歇失败、pip 默认源 stall——变通方案记录于 DECISIONS

## 9. 证据索引（全部落盘 `tracking/overnight_v1/receipts/`）

| 收据 | sha256（前 8） |
|---|---|
| m1-gpu-suite / m1-gates-{default,batched,fused} | 见文件（17/17×3 + 314/314） |
| m1-ab-confirm-s424243.json | `9b2026b2` |
| m2-ladder-bf16.json | `d0bf6477` |
| m3-model-integration.json（probe/consistency/ppl） | 见文件 |
| m3-niah-{dense,lsa_1024,lsa_256}.json（5 例 pilot） | `b3d22e61` / `813d0c43` / `75244b31` |
| m3-ppl-extended.json（48 窗口 98,304 tokens） | `33143c4a` |
| m3-niah-*-ext.json（15 例 ×3 臂） | `7185d5f8` / `a43aefa9` / `aae368d1` |
| m6-soak.json（148 轮） | `64ed5426` |
| m1-fresh-s{seed}-rep{n}.json（3×3 campaign ×9） | 见 receipts/ 与 §2a |

## 10. Owner 专属下一步（不变 + 新增）

1. （原有）sudo 执行 `docs/r1.1/ops/migrate-runner.sh` → 批准 post-merge GPU run（42483727525 现为验证载体）→ 退役旧装 → 轮换 amd 凭据 → 应用 P1–P4。
2. （原有）审阅 v0.4.0 RC → 授权 tag/Release（工程 GO 已具备，见 §2）。
3. （新增，来自今晚证据）决定 fused 的生产范围声明：≤32k 阶梯内 opt-in（M2 证明 ≥64k 应保持 reference）。
4. （新增）若要 v0.5：投入 upstream 蒸馏路线训练 index 投影（Qwen3 类大 head_dim 模型的唯一路线）+ 预注册 RULER/NIAH ≥100 例 campaign。
5. （新增）port 工程项：短序列（T<512）bucket 支持；chunked prefill 设计（>128k 唯一路径）。
6. （新增）SGLang：先决定是否预拉批准的 gfx1151 容器车道，再谈 v0.6。
7. 本分支处置：`exp/gfx1151-v1-night-20261108-2155`（本地，含全部收据与 FINDINGS）——owner 决定推送/开 Draft PR（今晚未推送）。

## 11. 结论

一夜之间完成了 M1 全绿、M2 的 64k–256k 诚实包络（含 fused 负发现）、M3 的模型级集成证明（含两个产品缺陷发现）、M4 的根因阻断、M6 稳定性与负控制、M7 资格包。**没有虚构任何里程碑**：v0.5 记 PILOT_DONE，v0.6 BLOCKED，v1.0.0 NO-GO，v0.4.0 发布与 PR#8 归 owner。GPU 全程单实验独占，OOM 全部分类，未虚报任何 PASS。
