# DECISION LOG — immutable decisions (date · SHA · reason)

Format: one entry per decision; append only, never rewrite.

| Date | SHA | Decision | Reason |
|---|---|---|---|
| 2026-11-06 | 2139758 | Public PRs run only GitHub-hosted CPU checks; persistent gfx1151 runner reachable only via trusted push/confirmed dispatch; `pull_request_target` prohibited (guard R1–R5) | Public repo + runner on interactive user with live gh credentials: untrusted PR code = credential compromise. Human review does not sandbox code. |
| 2026-11-06 | 2139758 | Campaign exit fails closed on any attempted-but-unverified kernel/E2E adapter; PASS-without-attempt is fabricated evidence | v0.2.1 review found BLOCKED (exit 78) and E2E failures never failed the campaign; campaign.sh exited 0 after echoing failures. |
| 2026-11-06 | 2139758 | Every published table requires auditor-APPROVED receipts (recomputed speedups, provenance, gate manifest); render + CI refuse the rest | Adapter-asserted status/speedups are claims, not evidence; independent recompute closes the fabrication class (F12). |
| 2026-11-06 | e243ba4 | Upstream checkout defaults to the reviewed SHA in tracking/UPSTREAM_BASELINE.json; `UPSTREAM_REF` is the only override | Floating upstream main in CI would silently test unreviewed code while receipts tracked the new SHA. |
| 2026-11-06 | e243ba4 | Seed study scheduled before v0.3.0 optimization claims | The three v0.2.1 decode repetitions share one seed; they bound timing noise, not seed sensitivity (review L3). |
| 2026-11-07 | 2daaebf | Multi-seed bands are the only publishable decode claim shape; the v0.2.1 single-seed 2.1× flat figure is corrected to the cross-seed band | T1 measured seed 20261105 as the favorable outlier (2.79× vs 1.69/1.74×); dense baseline itself is seed-stable, so the spread is genuine sparse-path input sensitivity |
| 2026-11-07 | 4988672 | Tiled sparse read becomes production behind a new gate H (tiled == v1 + zero-length contract); manifest 16→17 gates, v0.2.1 receipts grandfathered as time-valid but non-validating against the current manifest | A kernel swap without a per-receipt equivalence gate would let numeric drift ride a speedup; the zero-length NaN was a latent contract hole |
| 2026-11-07 | pending | Tree beam stays on the per-page walk; page-batched torch kept experimental (LSA_TREE_IMPL=batched) after measuring 0.81–0.87× | Launch-tax hypothesis from T2 was wrong at these shapes: the beam is op-bound, not launch-bound; a Triton per-level port (upstream layout) is the only remaining kernelization lever |
