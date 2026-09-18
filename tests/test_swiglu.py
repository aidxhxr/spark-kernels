import pytest
import torch

from conftest import TOL, dtype_id

DTYPES = [torch.float32, torch.bfloat16]
SHAPES = [(1, 256), (7, 1024), (4096, 4096), (4096, 11008), (3, 5, 1000)]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("swiglu")))
    except Exception:
        return [0]


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_swiglu_matches_reference(sk, dtype, shape, variant):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)
    got = sk.swiglu(gate, up, variant)
    ref = sk.reference.swiglu(gate, up)
    assert got.shape == gate.shape and got.dtype == dtype
    torch.testing.assert_close(got, ref, **TOL[dtype])


def test_swiglu_shape_mismatch(sk):
    g = torch.randn(4, 256, device="cuda")
    u = torch.randn(4, 128, device="cuda")
    with pytest.raises(RuntimeError):
        sk.swiglu(g, u)
