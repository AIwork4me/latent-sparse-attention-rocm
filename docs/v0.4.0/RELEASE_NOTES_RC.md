# v0.4.0 — Measured Select Fusion on ROCm gfx1151

**STATUS: RELEASE CANDIDATE — UNRELEASED. No tag or GitHub Release created.**

Reviewed source chain:
- v0.3.1 baseline main: `1ab0e0bd32679b11f51015f6a2db8c811292fefd`
- R1.1 security/ops merge: PR #6, `518ad3031f244c105cc2f1afacd23e75025f3b26`
- Select-fusion integration merge: PR #7, `6d17a7deb1d0537a938b341410772fc379bcbf06`
- Final physical-gfx1151 numerical-code confirmation: `45a0ed12201b3a301712815dab32c0dac9c8fee4`
- Evidence and adversarial review: `docs/reviews/V0_4_INDEPENDENT_REVIEW.md`; `evidence/v0.4.0/`

## Summary

This release candidate integrates two ROCm Triton optimizations for latent sparse
attention's flat selection: (1) scoring-row pack and sorted top-k within a
single kernel, avoiding materialization of the large int64 listing-key matrix,
and (2) unpacking selected packed keys as part of the upstream-equivalent
expansion walk. The upsteam pin is `5ebea732d639576d2a7579b0de40e6252f9001f5`.

**Compatibility-preserving default:** with `LSA_SELECT_IMPL` unset, the
v0.3.1 `reference` implementation remains active. To opt into fusion set
`LSA_SELECT_IMPL=fused`; `LSA_SELECT_IMPL=fused_unpack` preserves the
unpack-only alternative. PR #8 is the **separate, still-Draft, owner-gated**
proposal for making fused the default.

## Accuracy and correctness

- Prior 17-gate correctness manifest remains the mandatory validation gate.
- Author-reported physical gfx1151 suite: 258/258 PASS; 17/17 gates in
  reference, batched-tree and fused-select modes. GitHub-hosted CPU CI passed
  on PR #7's reviewed head; post-merge gfx1151 CI requires owner migration.
- Byte-exact ordered read lists and lengths match the reference in the
  tested configurations, including sampled-wide paths and pow2 bucket capacities
  8/16/32/64. The previously unmasked sampled-listing path was fixed.
- Existing v0.3.1 whole-listing operator sequence retained; a no-op mask
  skip avoids a separately measured ~28% prefill-select slowdown.

## Measured outcomes (Radeon 8060S/gfx1151, BF16, 32k)

The multi-seed campaign compares **fused to v0.3.1 reference select+read**,
not to Dense SDPA or to end-to-end model serving.

| Stage | Reference | Fused | Observed improvement |
|---|---:|---:|---:|
| Prefill select | 98.0 ms | 33.1 ms | 2.96x |
| Prefill select+read | 349.4 ms | 284.9 ms | 1.226x |
| Decode B=1 select+read | 0.317 ms | 0.266 ms | 1.199x |

Method: 3 seeds × 3 independent processes, 5 warmups and 20 timing
samples, interleaved A/B; observed min/max are not confidence intervals.
No >2% measured regression across 120 aggregate cells (minimum ratio
0.9986x, hence *not* literally every cell was faster).

Clean-tree final confirmation on a single seed recorded 2.977x prefill select,
1.225x prefill select+read and 1.224x decode select+read.

The 32k prefill-select **PyTorch max_memory_allocated delta** decreased
from 2304.2 to 1344.9 MiB (**959.4 MiB / 41.6%**). This is not the
total system UMA memory or an end-to-end model peak. The five-sample
clean-tree receipt is `evidence/v0.4.0/ab/memory-probe.json`; the superseded
dirty-provenance receipt remains preserved separately.

## Evidence integrity

All **25 v0.4.0 evidence files** (baseline, A/B raw samples, aggregates,
final verification and archived superseded receipt) are sealed with SHA256
in `evidence/v0.4.0/SHA256SUMS`. Verify from a checkout:

```bash
cd evidence/v0.4.0 && sha256sum -c SHA256SUMS
python -m pytest tests/test_v040_release_integrity.py -q  # run at repo root
```

The test suite checks precise coverage and immutable evidence bytes on the
GitHub-hosted CPU CI. Existing historical `evidence/SHA256SUMS` and
v0.2.1/v0.3.1 Release tags were not rewritten.

## Exclusions and deployment state

- **NOT TESTED:** W7900/gfx1100.
- **NOT CLAIMED:** SGLang end-to-end serving, real-model quality retention,
  full inference latency/throughput improvement, 64k–256k performance.
- **PENDING OWNER:** dedicated runner systemd migration, real-world
  device sandbox checks and post-merge GitHub GPU CI; separate default
  promotion and GPU regression via PR #8.
- **NO-GO** for assigning an unconditional production default improvement
  or for promoting model-level performance based on these receipts.

## Publication gate

Do not tag or publish until the owner verifies the hosted-CPU release
integrity checks, reads the independent review, and explicitly authorizes
v0.4.0 publication. If the project chooses to require the new migrated
runner's post-merge GPU workflow as a release gate, leave publication
blocked until that job succeeds on the exact main SHA.

The RELEASE NOTES may be pasted into an official GitHub Release only
after the status, gate wording and target tag SHA are reconciled against
the actual final `main` commit; remove this UNRELEASED header at that time.
