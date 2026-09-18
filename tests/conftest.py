import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    # Only tests that touch the compiled extension (via the `sk` fixture) need a GPU; the
    # rest (e.g. the results-script helpers) run anywhere, including CI.
    skip = pytest.mark.skip(reason="CUDA GPU required")
    for item in items:
        if "sk" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-4),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


@pytest.fixture(scope="session")
def sk():
    import spark_kernels

    return spark_kernels


def dtype_id(dt):
    return {torch.float32: "f32", torch.bfloat16: "bf16"}[dt]
