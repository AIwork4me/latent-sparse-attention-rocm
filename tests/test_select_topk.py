"""Best-first key selection (v0.3.1): topk replaces the full-width sort + slice.

Upstream packs each candidate as (score_bits << 32) | (0x7FFFFFFF - column) -
UNIQUE per column, so no two keys ever tie and ``torch.topk(sorted=True)``
returns exactly what ``torch.sort(...).values[:, :k]`` returns. The sort it
replaces cost ~19% of the prefill select (plus the strided-slice copies the
profiled copy chain) sorting up to 4x more keys than needed. These tests pin
the equivalence - with tied SCORES, EMPTY keys, narrow rows and k >= width -
so the swap can never change which buckets the expansion walks.

The contract is over int64 keys with pure-torch ops, so it holds on any
device: every case runs on CPU everywhere (the CPU CI must execute them,
never skip), and the visible ROCm device is parametrized in when present so
production-shape coverage also runs on gfx1151.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa_rocm_lab.port.pipeline import best_first_keys  # noqa: E402


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _gpu_available() else [])


def _packed(scores: torch.Tensor) -> torch.Tensor:
    """Upstream's pack: ordered float bits high, 0x7FFFFFFF - column low."""
    ordered = scores.view(torch.int32) ^ torch.tensor(-0x80000000, dtype=torch.int32)
    columns = torch.arange(scores.shape[1], device=scores.device)
    return (ordered.to(torch.int64) << 32) | (0x7FFFFFFF - columns).to(torch.int64)


def _old_way(keys: torch.Tensor, top_p: int) -> torch.Tensor:
    return torch.sort(keys, dim=1, descending=True).values[:, :top_p]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,width,top_p", [
    (64, 4096, 1024),   # the wide-row production shape (listed superset 4x top_p)
    (64, 1024, 1024),   # k == width
    (7, 300, 128),      # ragged narrow rows
])
def test_topk_equals_sort_slice_on_production_shapes(rows, width, top_p, device):
    torch.manual_seed(20261107)
    scores = torch.randn(rows, width, device=device)
    keys = _packed(scores)
    assert torch.equal(best_first_keys(keys, top_p), _old_way(keys, top_p))


@pytest.mark.parametrize("device", DEVICES)
def test_tied_scores_resolve_by_column_both_ways(device):
    # equal scores everywhere: packed keys stay unique via the column bits
    torch.manual_seed(1)
    scores = torch.zeros(16, 512, device=device)
    keys = _packed(scores)
    assert torch.equal(best_first_keys(keys, 64), _old_way(keys, 64))


@pytest.mark.parametrize("device", DEVICES)
def test_empty_keys_sort_last_both_ways(device):
    torch.manual_seed(2)
    scores = torch.randn(8, 2048, device=device)
    keys = _packed(scores)
    keys[:, ::3] = -1  # EMPTY keys scattered in
    a = best_first_keys(keys, 256)
    b = _old_way(keys, 256)
    assert torch.equal(a, b)
    assert (a != -1).all()  # EMPTY never reaches the top 256 of random scores
