"""Pure helpers of the T2 stage profiler: kernel-row attribution and formatting.

The profiler itself needs the GPU; these tests pin the attribution contract -
device-side rows only, shares of the GPU total, deterministic markdown - so a
profile JSON can never attribute python/launch time as kernel time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from profile_stages import format_kernel_table, kernel_rows  # noqa: E402


class _Evt:
    def __init__(self, key, count, cpu_us, dev_us):
        self.key = key
        self.count = count
        self.self_cpu_time_total = cpu_us
        self.self_device_time_total = dev_us


def test_kernel_rows_keeps_device_rows_with_shares():
    rows = kernel_rows([_Evt("_sparse_read_kernel", 8, 100, 800),
                        _Evt("aten::mm", 8, 50, 200),
                        _Evt("Memcpy DtoH", 1, 10, 0)])  # 0us device -> dropped
    assert [r["name"] for r in rows] == ["_sparse_read_kernel", "aten::mm"]
    assert rows[0]["total_us"] == 800 and rows[0]["count"] == 8
    # shares of the GPU total (1000us): 80% / 20%, cumulative ordering
    assert rows[0]["share"] == 0.80 and rows[1]["share"] == 0.20
    assert rows[0]["cumulative"] == 0.80 and rows[1]["cumulative"] == 1.0


def test_kernel_rows_empty_is_empty():
    assert kernel_rows([]) == []


def test_format_table_top_n_and_totals():
    rows = kernel_rows([_Evt("k1", 8, 100, 900), _Evt("k2", 8, 50, 100)])
    table = format_kernel_table(rows, top=1)
    assert "| kernel | count | total us | share |" in table
    assert "k1" in table and "k2" not in table
    rows2 = kernel_rows([_Evt("only", 3, 1, 42)])
    assert "only" in format_kernel_table(rows2, top=5)
