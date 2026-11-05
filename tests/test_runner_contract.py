from lsa_rocm_lab.run import stage_command

def test_missing_port_is_blocked_not_faked(monkeypatch):
    monkeypatch.delenv('LSA_ROCM_KERNEL_CMD',raising=False)
    monkeypatch.delenv('LSA_ROCM_E2E_CMD',raising=False)
    assert stage_command('lsa_rocm_kernel',True) is None
    assert stage_command('lsa_rocm_end_to_end',True) is None
