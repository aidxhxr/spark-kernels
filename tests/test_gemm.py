import pytest
import torch

from conftest import TOL

SGEMM_SHAPES = [(64, 64, 64), (129, 257, 65), (1024, 1024, 1024), (1, 4096, 4096)]
HGEMM_SHAPES_16 = [(256, 256, 256), (272, 144, 48), (16, 16, 16)]
HGEMM_SHAPES_128 = [(256, 384, 256), (128, 128, 32), (512, 512, 4096)]
HGEMM_TOL = dict(atol=3e-2, rtol=3e-2)


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants(name)))
    except Exception:
        return [0]


@pytest.mark.parametrize("variant", _variants("sgemm"))
@pytest.mark.parametrize("shape", SGEMM_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_sgemm_matches_torch(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    got = sk.sgemm(a, b, variant)
    ref = a @ b
    assert got.shape == (M, N)
    # fp32 accumulation order differs from cuBLAS; scale tolerance with K.
    torch.testing.assert_close(got, ref, atol=1e-3 * (K**0.5), rtol=1e-4)


def _hgemm_case(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    got = sk.hgemm(a, b, variant)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


@pytest.mark.parametrize("variant", [v for v in _variants("hgemm") if v < 2])
@pytest.mark.parametrize("shape", HGEMM_SHAPES_16, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_multiples_of_16(sk, shape, variant):
    _hgemm_case(sk, shape, variant)


@pytest.mark.parametrize("variant", _variants("hgemm"))
@pytest.mark.parametrize("shape", HGEMM_SHAPES_128, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_multiples_of_128(sk, shape, variant):
    _hgemm_case(sk, shape, variant)


def test_hgemm_rejects_unaligned(sk):
    a = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        sk.hgemm(a, b)


def test_sgemm_rejects_inner_mismatch(sk):
    a = torch.randn(8, 16, device="cuda")
    b = torch.randn(8, 16, device="cuda")
    with pytest.raises(RuntimeError):
        sk.sgemm(a, b)


def test_tolerance_dict_present():
    assert torch.float32 in TOL


def test_hgemm_default_variant_takes_any_multiple_of_16(sk):
    M, N, K = 272, 144, 48  # multiples of 16 but not of the 128x128x32 tile of variant 2
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    got = sk.hgemm(a, b)  # steps down to variant 1
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.hgemm(a, b, 2)
