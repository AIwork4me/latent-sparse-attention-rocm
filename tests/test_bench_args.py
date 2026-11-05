"""Adapter CLI filters for repetition runs - matched shapes only, loudly."""

import pytest

from lsa_rocm_lab.kernel_adapter import CONTEXTS, parse_contexts


def test_parse_contexts_none_means_full_ladder():
    assert parse_contexts(None) is None
    assert parse_contexts("") is None


def test_parse_contexts_comma_list_sorted_unique():
    assert parse_contexts("16384,8192,8192") == [8192, 16384]


def test_parse_contexts_rejects_off_ladder_shapes():
    with pytest.raises(ValueError, match="ladder"):
        parse_contexts("8192,6000")


def test_ladder_is_the_campaign_ladder():
    assert CONTEXTS == [512, 1024, 2048, 4096, 8192, 16384, 32768]
