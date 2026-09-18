import pytest
import torch

from conftest import TOL, dtype_id

DTYPES = [torch.float32, torch.bfloat16]
ROWS = [1, 7, 4096]
COLS = [256, 1024, 4096, 11008]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("rmsnorm")))
    except Exception:  # extension not built / no GPU: tests get skipped anyway
        return [0]


def _vector_width(dtype):
    return 4 if dtype == torch.float32 else 8


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_rmsnorm_matches_reference(sk, dtype, rows, cols, variant):
    if cols % _vector_width(dtype) != 0 and variant >= 2:
        pytest.skip("vectorized variant needs aligned cols")
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = (1.0 + 0.1 * torch.randn(cols, device="cuda")).to(dtype)
    got = sk.rmsnorm(x, w, 1e-6, variant)
    ref = sk.reference.rmsnorm(x, w, 1e-6)
    assert got.shape == x.shape and got.dtype == dtype
    torch.testing.assert_close(got, ref, **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
def test_rmsnorm_flattens_leading_dims(sk, dtype):
    x = torch.randn(2, 3, 5, 1024, device="cuda", dtype=dtype)
    w = torch.ones(1024, device="cuda", dtype=dtype)
    got = sk.rmsnorm(x, w)
    torch.testing.assert_close(got, sk.reference.rmsnorm(x, w), **TOL[dtype])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", [256, 4096, 11008])
def test_add_rmsnorm_inplace(sk, rows, cols):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    resid = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = (1.0 + 0.1 * torch.randn(cols, device="cuda")).to(dtype)
    ref_resid, ref_out = sk.reference.add_rmsnorm(x, resid.clone(), w)
    out = sk.add_rmsnorm_(x, resid, w)
    torch.testing.assert_close(resid, ref_resid, **TOL[dtype])
    torch.testing.assert_close(out, ref_out, **TOL[dtype])


def test_rmsnorm_rejects_bad_weight(sk):
    x = torch.randn(4, 256, device="cuda")
    w = torch.ones(128, device="cuda")
    with pytest.raises(RuntimeError):
        sk.rmsnorm(x, w)
