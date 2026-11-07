import pytest

from lsa_rocm_lab.run import MANDATORY_STAGES, campaign_exit_code, stage_command

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

    @pytest.mark.parametrize('bad_status', [
        'FAIL', 'TIMEOUT', 'BLOCKED', 'ERROR', 'OOM', 'CRASHED', 'UNVERIFIED', '',
    ])
    def test_attempted_kernel_adapter_anything_but_pass_fails(self, bad_status):
        """F11: configured+attempted adapter, anything other than verified PASS
        (BLOCKED from exit 78 included) must fail the campaign."""
        statuses = _statuses(lsa_rocm_kernel={'status': bad_status, 'command': ['x'],
                                              'exit_code': 78 if bad_status == 'BLOCKED' else 1})
        assert campaign_exit_code(statuses) == 1

    def test_kernel_blocked_without_adapter_is_not_a_failure(self):
        assert campaign_exit_code(
            _statuses(lsa_rocm_kernel={'status': 'BLOCKED', 'command': None})
        ) == 0

    def test_kernel_pass_without_command_is_fabricated_evidence(self):
        """A PASS record with no adapter run can never exit zero."""
        assert campaign_exit_code(
            _statuses(lsa_rocm_kernel={'status': 'PASS', 'command': None})
        ) == 1

    def test_kernel_record_with_no_status_key_fails_closed(self):
        statuses = _statuses()
        del statuses[4]['status']  # a truncated record must not be treated as PASS
        assert campaign_exit_code(statuses) == 1

    def test_preflight_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(preflight={'status': 'FAIL'})) == 1

    def test_correctness_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(reference_correctness={'status': 'FAIL'})) == 1

    def test_dense_baseline_failure_fails_the_campaign(self):
        assert campaign_exit_code(_statuses(dense_prefill_baseline={'status': 'FAIL'})) == 1

    @pytest.mark.parametrize('stage', list(MANDATORY_STAGES[:-1]))
    def test_any_mandatory_stage_missing_fails(self, stage):
        kept = [s for s in _statuses() if s['stage'] != stage]
        assert campaign_exit_code(kept) == 1

    def test_kernel_stage_missing_entirely_fails(self):
        assert campaign_exit_code(_statuses()[:4]) == 1

    @pytest.mark.parametrize('bad_status', ['FAIL', 'TIMEOUT', 'BLOCKED', 'ERROR', 'OOM'])
    def test_attempted_e2e_adapter_failure_fails_the_campaign(self, bad_status):
        """F11: an explicitly configured and attempted E2E adapter may not
        quietly green its own failure."""
        statuses = _statuses(lsa_rocm_end_to_end={'status': bad_status, 'command': ['x']})
        assert campaign_exit_code(statuses) == 1

    def test_e2e_not_requested_stays_a_legitimate_block(self):
        assert campaign_exit_code(
            _statuses(lsa_rocm_end_to_end={'status': 'BLOCKED', 'command': None})
        ) == 0

    def test_absent_e2e_record_is_not_a_failure(self):
        assert campaign_exit_code(_statuses()[:5]) == 0
