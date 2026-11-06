# DECISION LOG — immutable decisions (date · SHA · reason)

Format: one entry per decision; append only, never rewrite.

| Date | SHA | Decision | Reason |
|---|---|---|---|
| 2026-11-06 | 2139758 | Public PRs run only GitHub-hosted CPU checks; persistent gfx1151 runner reachable only via trusted push/confirmed dispatch; `pull_request_target` prohibited (guard R1–R5) | Public repo + runner on interactive user with live gh credentials: untrusted PR code = credential compromise. Human review does not sandbox code. |
| 2026-11-06 | 2139758 | Campaign exit fails closed on any attempted-but-unverified kernel/E2E adapter; PASS-without-attempt is fabricated evidence | v0.2.1 review found BLOCKED (exit 78) and E2E failures never failed the campaign; campaign.sh exited 0 after echoing failures. |
| 2026-11-06 | 2139758 | Every published table requires auditor-APPROVED receipts (recomputed speedups, provenance, gate manifest); render + CI refuse the rest | Adapter-asserted status/speedups are claims, not evidence; independent recompute closes the fabrication class (F12). |
| 2026-11-06 | e243ba4 | Upstream checkout defaults to the reviewed SHA in tracking/UPSTREAM_BASELINE.json; `UPSTREAM_REF` is the only override | Floating upstream main in CI would silently test unreviewed code while receipts tracked the new SHA. |
| 2026-11-06 | e243ba4 | Seed study scheduled before v0.3.0 optimization claims | The three v0.2.1 decode repetitions share one seed; they bound timing noise, not seed sensitivity (review L3). |
