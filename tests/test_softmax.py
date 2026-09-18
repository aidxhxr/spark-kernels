import pytest
import torch

from conftest import TOL, dtype_id

DTYPES = [torch.float32, torch.bfloat16]
ROWS = [1, 7, 4096]
COLS = [256, 1024, 4096, 11008]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("softmax")))
    except Exception:
        return [0]


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_softmax_matches_reference(sk, dtype, rows, cols, variant):
    torch.manual_seed(0)
    # Wide dynamic range so a non-online / non-max-subtracted implementation would overflow.
    x = torch.randn(rows, cols, device="cuda", dtype=dtype) * 20.0
    got = sk.softmax(x, variant)
    ref = sk.reference.softmax(x)
    assert got.shape == x.shape and got.dtype == dtype
    torch.testing.assert_close(got, ref, **TOL[dtype])
    # Rows sum to one (in fp32).
    sums = got.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=5e-3, rtol=0)


@pytest.mark.parametrize("variant", _variants())
def test_softmax_extreme_values(sk, variant):
    x = torch.tensor([[1e4, -1e4, 0.0, 1e4]], device="cuda")
    got = sk.softmax(x, variant)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, torch.softmax(x, dim=-1), **TOL[torch.float32])


def test_softmax_flattens_leading_dims(sk):
    x = torch.randn(2, 4, 8, 512, device="cuda")
    torch.testing.assert_close(sk.softmax(x), torch.softmax(x, dim=-1), **TOL[torch.float32])
