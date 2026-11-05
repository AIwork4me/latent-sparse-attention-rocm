from lsa_rocm_lab.run import campaign_exit_code, stage_command

def test_missing_port_is_blocked_not_faked(monkeypatch):
    monkeypatch.delenv('LSA_ROCM_KERNEL_CMD',raising=False)
    monkeypatch.delenv('LSA_ROCM_E2E_CMD',raising=False)
    assert stage_command('lsa_rocm_kernel',True) is None
    assert stage_command('lsa_rocm_end_to_end',True) is None


def _statuses(**over):
    """The happy path of a campaign with no E2E adapter (kernel PASS, end blocked)."""
    base = [
        {'stage': 'preflight', 'status': 'PASS', 'command': ['x']},
        {'stage': 'reference_correctness', 'status': 'PASS', 'command': ['x']},
        {'stage': 'triton_hip_probe', 'status': 'PASS', 'command': ['x']},
        {'stage': 'dense_prefill_baseline', 'status': 'PASS', 'command': ['x']},
        {'stage': 'lsa_rocm_kernel', 'status': 'PASS', 'command': ['x']},
        {'stage': 'lsa_rocm_end_to_end', 'status': 'BLOCKED', 'command': None},
    ]
    for stage, patch in over.items():
        for row in base:
            if row['stage'] == stage:
                row.update(patch)
    return base


class TestCampaignExitCode:
    def test_all_pass_with_blocked_e2e_exits_zero(self):
        assert campaign_exit_code(_statuses()) == 0

    def test_kernel_failure_fails_the_campaign(self):
        """v0.2.0 hole: kernel FAIL left the campaign exit code at 0."""
        assert campaign_exit_code(_statuses(lsa_rocm_kernel={'status': 'FAIL'})) == 1

    def test_kernel_timeout_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(lsa_rocm_kernel={'status': 'TIMEOUT'})) == 1

    def test_kernel_blocked_without_adapter_is_not_a_failure(self):
        assert campaign_exit_code(
            _statuses(lsa_rocm_kernel={'status': 'BLOCKED', 'command': None})
        ) == 0

    def test_preflight_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(preflight={'status': 'FAIL'})) == 1

    def test_correctness_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(reference_correctness={'status': 'FAIL'})) == 1

    def test_dense_baseline_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(dense_prefill_baseline={'status': 'FAIL'})) == 1

    def test_missing_stage_record_fails_the_campaign(self):
        assert campaign_exit_code(_statuses()[:4]) == 1

    def test_e2e_blocked_or_failed_adapter_does_not_fail_mandatory_exit(self):
        statuses = _statuses(lsa_rocm_end_to_end={'status': 'FAIL', 'command': ['x']})
        assert campaign_exit_code(statuses) == 0  # E2E is optional evidence, never silently green
